from sequential_tasks import EchoData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# By taking away these seeds, we see that even performance here is very stochastic.

# torch.manual_seed(1)
# np.random.seed(3)

batch_size = 5
echo_step = 5
series_length = 20_000
BPTT_T = 100

train_size = -1
test_size = -1
total_values_in_one_chunck = -1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

feature_dim = 1 #since we have a scalar series
h_units = 20

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_data():
    global total_values_in_one_chunck, train_size, test_size, BPTT_T, series_length, echo_step, batch_size

    train_data = EchoData(
        echo_step=echo_step,
        batch_size=batch_size,
        series_length=series_length,
        truncated_length=BPTT_T
    )
    total_values_in_one_chunck = batch_size * BPTT_T
    train_size = len(train_data)

    test_data = EchoData(
        echo_step=echo_step,
        batch_size=batch_size,
        series_length=series_length,
        truncated_length=BPTT_T,
    )
    test_data.generate_new_series()
    test_data.prepare_batches()
    test_size = len(test_data)

    return train_data, test_data

class SimpleRNN(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, output_size):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_cell = torch.nn.RNNCell(
            input_size=input_size, 
            hidden_size=rnn_hidden_size, 
            nonlinearity='relu',
            # batch_first = True?
        )
        self.linear = torch.nn.Linear( # This is the decoder
            in_features=rnn_hidden_size,
            out_features=output_size
        )

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.rnn_hidden_size).to(x.device)
            
        hidden_states_list = []
        for t in range(seq_len):
            hidden = self.rnn_cell(x[:, t, :], hidden)
            if self.training:
                hidden.retain_grad() # This .grad attribute will be filled during .backward()
                
            hidden_states_list.append(hidden)

        # Stack to look like the old output (batch, seq, hidden)
        # Note: We stack them, but we will use the LIST for gradient inspection
        # because the list holds the original graph nodes.
        all_hidden = torch.stack(hidden_states_list, dim=1)
        out = self.linear(all_hidden)
        return out, hidden, all_hidden, hidden_states_list
    

# Probably don't even use train
def train():
    model.train()
    
    # New epoch --> fresh hidden state
    hidden = None   
    correct = 0
    for batch_idx in range(train_size):
        data, target = train_data[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        optimizer.zero_grad()
        if hidden is not None: hidden.detach_()
        logits, hidden, _, _ = model(data, hidden)

        # RNN has a bijection between 
        # print(data.shape, target.shape)

        loss = criterion(logits, target) # It doesn't do anything really special
        # It just has a 1-1 mapping between data and target, all at once.
        # And then it later on does a topological sort, rooted at loss.
        loss.backward() # Calculates all gradients involved (anything that has autograd=True)
        # And adds the grad dL/dw_i
        optimizer.step() # This just steps all of those things, but carefully (i.e. following
        # a stepping algorithm, like AdamW)

        
        pred = (torch.sigmoid(logits) > 0.5)
        correct += (pred == target.byte()).int().sum().item()/total_values_in_one_chunck
        
    return correct, loss.item()

def test(model, test_data):
    global total_values_in_one_chunck, test_size
    model.eval()   
    correct = 0
    # New epoch --> fresh hidden state
    hidden = None
    with torch.no_grad():
        for batch_idx in range(test_size):
            data, target = test_data[batch_idx]
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
            logits, hidden, _, _ = model(data, hidden)
            
            pred = (torch.sigmoid(logits) > 0.5)
            correct += (pred == target.byte()).int().sum().item()/total_values_in_one_chunck

    return correct

def modified_train_regular(model, optimizer, criterion, train_data, test_data):
    global total_values_in_one_chunck, train_size
    model.train()

    # New epoch --> fresh hidden state
    hidden = None   
    iterations = min(10000000, train_size)

    total_loss = 0
    total_acc = 0
    loss_arr = []

    grad_arr = []
    test_acc = []

    for batch_idx in (range(iterations)):
        data, target = train_data[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        
        optimizer.zero_grad()
        if hidden is not None: hidden.detach_()
        logits, hidden, hidden_stack, hidden_list = model(data, hidden)

        loss = criterion(logits, target) # loss for this batch only
        loss.backward()

        model_dict = {}
        for name, p in model.named_parameters():
            model_dict[name] = {
                'param': p.detach().clone(),
                'grad': p.grad.detach().clone() if p.grad is not None else None
            }

        grad_arr.append(model_dict)

        optimizer.step()
        
        loss_arr.append(loss.detach().clone().cpu().item())

        pred = (torch.sigmoid(logits) > 0.5)
        correct = (pred == target.byte()).int().sum().item()/total_values_in_one_chunck # acc for this batch only
        
        if batch_idx % 100 == 0:
            test_acc.append(test(model, test_data))      

        with torch.no_grad():
            total_loss += loss.detach().cpu().item()
            total_acc += correct


    avg_acc = float(total_acc)*100/iterations
    avg_loss = total_loss/iterations
    # (f'Train Epoch: {epoch}/{n_epochs}, loss: {avg_loss:.3f}, accuracy {avg_acc:.1f}%')
    return avg_acc, avg_loss, {"loss": loss_arr, "grad": grad_arr, "test_acc": test_acc}

def compute_dE_dh(model, criterion, data, target, hidden=None): # returns an array
    model.train()
    seq_len = target.shape[1]

    if hidden is not None: hidden.detach_()
    logits, hidden, hidden_stack, hidden_list = model(data, hidden)
    arr_dE_dh = []
    sum_so_far = None
    for i in range(seq_len):
        dEi = []

        loss = criterion(logits[:, i:i+1,:], target[:, i:i+1,:]) * torch.tensor(1/seq_len)
        loss.backward(retain_graph=True)

        if i == 0:
            sum_so_far = [torch.zeros_like(hidden_list[0])] * len(hidden_list)
        dEi = [x.grad.clone() for x in hidden_list]

        for j in range(seq_len):
            temp = dEi[j]
            dEi[j] = dEi[j] - sum_so_far[j]
            sum_so_far[j] = temp
        arr_dE_dh.append(dEi)
    
    return arr_dE_dh, hidden_list

# CHAT-GPTed code, might be wrong
def compute_dE_dh_fast(model, criterion, data, target, hidden=None):
    model.train()
    seq_len = target.shape[1]

    if hidden is not None:
        hidden = hidden.detach()

    logits, hidden, hidden_stack, hidden_list = model(data, hidden)

    arr_dE_dh = []

    for i in range(seq_len):
        loss_i = criterion(
            logits[:, i:i+1, :],
            target[:, i:i+1, :]
        ) / seq_len

        grads = torch.autograd.grad(
            loss_i,
            hidden_list,
            retain_graph=True,
            allow_unused=True
        )

        arr_dE_dh.append([
            g.clone() if g is not None else torch.zeros_like(h)
            for g, h in zip(grads, hidden_list)
        ])

    return arr_dE_dh, hidden_list

def populate_grad(model, dEi, start_idx, h_prev, model_dict, data, target, hidden=None): 
    # Uses global data, target, seq_len, model

    # Need to zero before calling if this is the only thing you want
    # Need to pass in h_{-1} if you want to (probably not a huge deal, especially if the sequence is long)

    # adjusts the model's parameters
    model.train()

    igates = torch.mm(data[:,start_idx,:].clone(), model_dict["rnn_cell.weight_ih"].t()) + model_dict["rnn_cell.bias_ih"] # These are the same p in our model
    hgates = torch.mm(h_prev, model_dict["rnn_cell.weight_hh"].t()) + model_dict["rnn_cell.bias_hh"] # And when we call activation.backward, it
    activation = torch.relu(igates + hgates)

    # Wait, I can just compute a vector jacobian product, because dEt_dhk is a vector. And dhk+/dtheta is the Jacobian.
    activation.backward(dEi[start_idx])

def enhanced_train_log(model, train_data, test_data, ll_optimizer, optimizer, criterion):
    global total_values_in_one_chunck, train_size
    hidden = None
    loss_arr = []
    grad_arr = []
    optim_arr = []
    test_acc = []

    total_loss, total_acc = 0, 0

    iterations = min(1000, train_size)
    # for batch in range(train_size):
    # for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches")): 
    for batch_idx in range(iterations):
        data, target = train_data[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        seq_len = target.shape[1]    
        
        # for diff in range(seq_len):
        assert BPTT_T == seq_len
        
        grad_arr_diff = []
        optim_arr_diff = [] # per batch computations over each diff
        print(batch_idx, "/", iterations)


        for diff in (range(seq_len)):

            ### OPTIONAL
            if diff > echo_step * 2:
                continue
            ###

            optimizer[diff].zero_grad()
            # arr_dE_dh, arr_h = compute_dE_dh(model, criterion, data, target, hidden)
            arr_dE_dh, arr_h = compute_dE_dh_fast(model, criterion, data, target, hidden)

            model_dict = {}
            for name, p in model.named_parameters():
                # print(name, p.shape)
                # print(name, " grad ", p.grad)
                # print(name, " param ", p)
                model_dict[name] = p # These are the same p in our model. Need to do this after every optimizer step.
                p.grad = None # Wait, so I'm resetting it every single time? And not letting them accumulate?

            for i in range(seq_len - diff):
                h_prev = -1 # h_{j-1}
                if i == 0:
                    h_prev = torch.zeros_like(arr_h[0]).to(device)
                else:
                    h_prev = arr_h[i-1].detach().clone()
                hidden = hidden.detach() if hidden is not None else None
                hidden = None
                populate_grad(model, arr_dE_dh[i+diff], i, h_prev, model_dict, data, target, hidden) # look at dE_{i+diff}/dh_i * dh_i+/dtheta


            # Two things to try, do gradient clipping of 1
            # It looks like the gradient norms grow together, suggesting some oscillatory behavior
            # It probably doesn't hurt to do both
            max_norm = 1.0  # or any threshold you choose
            torch.nn.utils.clip_grad_norm_(model.rnn_cell.parameters(), max_norm)

            for name, p in model.named_parameters():
                model_dict[name] = {
                    'param': p.detach().clone(),
                    'grad': p.grad.detach().clone() if p.grad is not None else None
                }

            grad_arr_diff.append(model_dict)
            # so grad_arr_diff now holds the grads

            # don't step yet
        for diff in range(seq_len):
            if diff > echo_step * 2:
                continue
            optimizer[diff].zero_grad()
            for name, p in model.named_parameters():
                p.grad = grad_arr_diff[diff][name]['grad']

            optimizer[diff].step()

            step_info = {"diff": diff}
            for name, param in model.rnn_cell.named_parameters():

                # Access optimizer state for this parameter
                state = optimizer[diff].state[param]
                if 'exp_avg' in state and 'exp_avg_sq' in state:
                    step_info[name] = {
                        'step': state.get('step', 0),
                        'norm_exp_avg': state['exp_avg'].norm().item(),
                        'norm_exp_avg_sq': state['exp_avg_sq'].norm().item()
                    }
            optim_arr_diff.append(step_info)

        ll_optimizer.zero_grad()
        logits, hidden, hidden_stack, hidden_list = model(data, hidden)
        loss = criterion(logits, target)

        loss.backward()
        ll_optimizer.step()
        loss_arr.append(loss.detach().clone().cpu().item())

        optim_arr.append(optim_arr_diff)
        grad_arr.append(grad_arr_diff)
        pred = (torch.sigmoid(logits) > 0.5)
        correct = (pred == target.byte()).int().sum().item()/total_values_in_one_chunck # acc for this batch only
        with torch.no_grad():
            total_loss += loss.detach().cpu().item()
            total_acc += correct

        if batch_idx % 100 == 0:
            test_acc.append(test(model, test_data))  

    avg_acc = float(total_acc)*100/iterations
    avg_loss = total_loss/iterations
    # (f'Train Epoch: {epoch}/{n_epochs}, loss: {avg_loss:.3f}, accuracy {avg_acc:.1f}%')
    return avg_acc, avg_loss, {"loss": loss_arr, "grad": grad_arr, "optim": optim_arr, "test_acc": test_acc}

def run_trial(seed, standard,
              
              # Need dummy to reassign global values
              batch_size_d=5, 
              echo_step_d=5, 
              series_length_d=20_000, 
              BPTT_T_d=100,

              h_units=20,
              optimizer_args={"lr":0.001, "betas":(0.95, 0.95)},
              ll_optimizer_args={"lr": 0.001, "betas": (0.95, 0.95)},
              n_epochs=10):
    set_seed(seed)
    global BPTT_T, series_length, echo_step, batch_size, feature_dim


    batch_size=batch_size_d
    echo_step=echo_step_d
    series_length=series_length_d
    BPTT_T=BPTT_T_d

    train_data, test_data = generate_data()
    model = SimpleRNN(input_size=1, rnn_hidden_size=h_units, output_size=feature_dim)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = None
    ll_optimizer = None

    if standard:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=optimizer_args["lr"], 
                                        betas=optimizer_args["betas"])
    else:
        optimizer = [torch.optim.AdamW(model.rnn_cell.parameters(), lr=optimizer_args['lr'], betas=optimizer_args['betas']) for i in range(BPTT_T)]
        ll_optimizer = torch.optim.AdamW(model.linear.parameters(), lr=ll_optimizer_args['lr'], betas=ll_optimizer_args['betas'])

    epoch_dict = {"standard": standard, 
                  "train_acc": [], 
                  "train_loss": [], 
                  "test_acc": [], 
                  "train_loss_arr": [],
                  "ll_optimizer": None,
                  "optimizer": None
                  }


    for epoch in range(n_epochs):
        acc, loss, loss_arr = 0, 0, []
        print(epoch, epoch, epoch)
        if standard:
            acc, loss, loss_arr = modified_train_regular(model, optimizer, criterion, train_data, test_data)
        else:
            # acc, loss, loss_arr = enhanced_train(model, train_data, ll_optimizer, optimizer, criterion)
            acc, loss, loss_arr = enhanced_train_log(model, train_data, test_data, ll_optimizer, optimizer, criterion)
        epoch_dict["train_acc"].append(acc)
        epoch_dict["train_loss"].append(loss)
        epoch_dict["test_acc"].append(float(test(model, test_data))*100/test_size)
        epoch_dict["train_loss_arr"].append(loss_arr)
    

    epoch_dict["ll_optimizer"] = ll_optimizer
    epoch_dict["optimizer"] = optimizer
    # for key, value in epoch_dict.items():
    #     print(key)
    #     print(value)
    #     print()
    return epoch_dict