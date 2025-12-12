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
    global total_values_in_one_chunck, train_size, test_size

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

def test(model):
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

def modified_train_regular():
    model.train()

    # New epoch --> fresh hidden state
    hidden = None   
    iterations = min(10000000, train_size)

    total_loss = 0
    total_acc = 0

    for batch_idx in (range(iterations)):
        data, target = train_data[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        
        optimizer.zero_grad()
        if hidden is not None: hidden.detach_()
        logits, hidden, hidden_stack, hidden_list = model(data, hidden)

        loss = criterion(logits, target) # loss for this batch only
        loss.backward()
        optimizer.step()
        
        pred = (torch.sigmoid(logits) > 0.5)
        correct = (pred == target.byte()).int().sum().item()/total_values_in_one_chunck # acc for this batch only
        with torch.no_grad():
            total_loss += loss.detach().cpu().item()
            total_acc += correct


    avg_acc = float(correct)*100/iterations
    avg_loss = total_loss/iterations
    # (f'Train Epoch: {epoch}/{n_epochs}, loss: {avg_loss:.3f}, accuracy {avg_acc:.1f}%')
    return avg_acc, avg_loss

def run_trial(seed, standard,
              batch_size=5, 
              echo_step=5, 
              series_length=20_000, 
              BPTT_T=100,
              h_units=20,
              optimizer_args={"lr":0.001, "betas":(0.95, 0.95)},
              n_epochs=10):
    set_seed(seed)

    batch_size=batch_size
    echo_step=echo_step
    series_length=series_length
    BPTT_T=BPTT_T
    h_units=h_units

    train_data, test_data = generate_data()
    model = SimpleRNN(input_size=1, rnn_hidden_size=h_units, output_size=feature_dim)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=optimizer_args["lr"], 
                                    betas=optimizer_args["betas"])

    epoch_dict = {"train_acc": [], "train_loss": [], "test_acc": []}

    for epoch in range(n_epochs):
        acc, loss = 0, 0
        if standard:
            acc, loss = modified_train_regular()
        else:
            acc, loss = modified_train_different()
        epoch_dict["train_acc"].append(acc)
        epoch_dict["train_loss"].append(loss)
        epoch_dict["test_acc"].append(float(test(model))*100/test_size)
