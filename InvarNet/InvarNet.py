import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import grad
from dataloader import generate_sequence, SequenceDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.fc(x[:,-1,:])
        return out

class Transformer(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads=2, dropout=0.2):
    super(Transformer, self).__init__()
    self.embedding = nn.Linear(input_size, hidden_size)
    encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.embedding(x)
    x = x.permute(1,0,2)
    x = self.encoder(x)
    x = x.permute(1,0,2)
    x = self.fc(x[:,-1,:])
    return x

def create_envs(name='bj', env=0, lookback=6, future=1):
    DATAPATH = f"..\InvarNet\envs_{name}\env_{env}.csv"
    df = pd.read_csv(DATAPATH)
    target = f'PM25_Concentration_{env}'
    sequence = generate_sequence(df, lookback=lookback, target=target, future=future)
    dataset = SequenceDataset(sequence)
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def mean_nll(logits, y):
    return nn.functional.mse_loss(logits, y)

def penalty(logits, y):
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    g = grad(loss, [scale], create_graph=True)[0]
    return torch.sum(g ** 2)

def train(input_size=6, hidden_size=48, num_layers=16, output_size=3, train_n_env=2, test_n_env=3, lr=0.001, n_epochs=500, lookback=12, loss_type ='ERM', l_e=1.0, l_2=1.0, l_p=1.0, l_rex=1.0, annealed_epoch=50):

    model = LSTM(input_size, hidden_size, num_layers, output_size).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Create val envs
    val_envs = [create_envs(name='bj', env=0, lookback=lookback, future=output_size)]

    train_loader1 = create_envs(name='gz', env=2, lookback=lookback, future=output_size)
    train_loader2 = create_envs(name='sz', env=2,lookback=lookback,future=output_size)

    train_size = int(0.2 * len(train_loader1.dataset))
    val_size = len(train_loader1.dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(train_loader1.dataset, [train_size, val_size])

    test_dataloader = DataLoader(train_dataset, batch_size=8567, shuffle=True)
    train_envs = [test_dataloader,create_envs(name='sz', env=2, lookback=lookback, future=output_size)]

    # only one for training
    combined_dataset = ConcatDataset([train_dataset, train_loader2.dataset])
    combined_train_loader = DataLoader(combined_dataset, batch_size=8567, shuffle=True)
    train_envs_erm = [combined_train_loader]

    '''
    test_envs = []
    for i in range(n_env):
        test_loader = create_envs(name='bj',env=i + 1, lookback=lookback, future=output_size)
        test_envs.append(test_loader)
    print("Environments Have Been Created...")
    '''
    # Training loop
    print(f"Environments Have Been Created...Start Training({loss_type})...")
    train_info = []
    #loss_info = []
    for epoch in range(n_epochs):
        total_loss = 0
        total_penalty = 0
        for train_loader in train_envs:
            for X, y in train_loader:
                X = X.cuda()
                y = y.cuda()
                logits = model(X)
                train_nll = mean_nll(logits, y)
                train_penalty = penalty(logits, y)
            total_loss += train_nll
            total_penalty += train_penalty

        total_loss = total_loss
        total_penalty = total_penalty
        #print(total_loss, total_penalty)

        annealed_penalty_weight = (l_p if epoch > annealed_epoch else 1e-10)
        if loss_type == 'ERM':
            loss = total_loss
        elif loss_type == 'IRM':
            loss = total_loss + annealed_penalty_weight * total_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 != 0:
            continue
        with torch.no_grad():

            train_error = 0.0
            mae_t_error = 0.0
            for train_loader in train_envs:
                for x_train, y_train in train_loader:
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()
                    y_train_pred = model(x_train)
                train_error += loss_fn(y_train_pred, y_train) / 8567

            val_error = 0.0
            mae_v_error = 0.0
            for val_loader in val_envs:
                for x_val, y_val in val_loader:
                    x_val = x_val.cuda()
                    y_val = y_val.cuda()
                    y_val_pred = model(x_val)
                val_error += loss_fn(y_val_pred, y_val) / 8567
            print("Epoch %d: train MSE %.4f, train MAE %.4f, validation MSE %.4f, validation MAE %.4f" % (epoch, train_error, mae_t_error, val_error, mae_v_error))
        train_info.append([train_error.cpu(), val_error.cpu()])

def main():
    parser = argparse.ArgumentParser(description='LSTM-IRM')
    parser.add_argument('--input_size', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=28)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--output_size', type=int, default=3)
    parser.add_argument('--train_n_env', type=int, default=3)
    parser.add_argument('--test_n_env', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lookback', type=int, default=7)
    parser.add_argument('--loss_type', type=str, default='ERM')
    parser.add_argument('--l_e', type=float, default=0.001)
    parser.add_argument('--l_2', type=float, default=0.001)
    parser.add_argument('--l_p', type=float, default=2e-3)
    parser.add_argument('--l_rex', type=float, default=1e-5)
    parser.add_argument('--annealed_epoch', type=int, default=50)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()