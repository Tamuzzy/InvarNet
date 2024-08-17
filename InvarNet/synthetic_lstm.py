import torch
import torch.nn as nn
from torch.autograd import grad
import argparse

torch.manual_seed(0)

def synthetic(n=1000, env=1.0, d=1):
  x = torch.zeros(n,d)
  y = torch.zeros(n,d)
  z = torch.zeros(n,d)
  x[0] = torch.randn(1) * env
  for i in range(1, n):
    x[i] = 0.1 * x[i-1] + torch.randn(1) * env
  y[0] = torch.randn(1) * env
  for j in range(1, n):
    y[j] = 0.1 * y[j-1] + x[j-1] + torch.randn(1) * env
  z[0] = torch.randn(1)
  for k in range(1, n):
    z[k] = 0.1 * z[k-1] + y[k-1] + torch.randn(1)
  feature = torch.cat((x,y,z), dim=1)
  train_ts = []
  label = []
  for t in range(n - 5 - 1):
    train_ts.append(feature[t + 1: t + 5 + 1])
    label.append(y[t + 5 + 1])
  return torch.stack(train_ts), torch.stack(label)


class LSTM(nn.Module):
  def __init__(self, input_size=3, hidden_size=3, num_layer=1, output_size=1):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layer, proj_size=output_size, batch_first=True)

  def forward(self, x):
    out,_ = self.lstm(x)
    return out[:,-1,:]

def compute_penalty (losses, dummy_w):
  g1 = grad(losses.mean(), dummy_w, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
  return (g1 * g2).sum()

model = LSTM(input_size=3,output_size=1).cuda()
dummy_w = torch.nn.Parameter(torch.Tensor ([1.0])).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
mse = torch.nn.MSELoss(reduction="none")
mae = torch.nn.L1Loss()

a1,b1 = synthetic(n=15000, env=0.1)
a2,b2 = synthetic(n=5000, env=1.0)

train_envs = [[torch.cat((a1,a2),0), torch.cat((b1,b2),0)]]

environments=[synthetic(n=15000, env=2.0)]
test_envs = [synthetic(n=10000, env=2.0)]

def train(n=50000, loss_type='ERM'):
  for iteration in range(n):
    error = 0
    penalty = 0
    rex = []
    for x_e, y_e in environments:
      p = torch.randperm(len(x_e))
      error_e = mse(model(x_e[p].cuda()) * dummy_w, y_e[p].cuda())
      penalty += compute_penalty(error_e, dummy_w)
      error += error_e.mean()
      rex.append(error_e.mean())
    rex = torch.stack(rex)
    opt.zero_grad ()
    if loss_type == 'IRM':
      (error + 1000 * penalty).backward()
    elif loss_type == 'ERM':
      error.backward()
    elif loss_type == 'REx':
      (error + 10 * rex.var()).backward()
    opt.step()
    if iteration % 1000 == 0:
      with torch.no_grad():
        error_sum = 0.0
        mae_sum = 0.0
        for x_t, y_t in test_envs:
          error_t = mse(model(x_t.cuda()), y_t.cuda())
          mae_t = mae(model(x_t.cuda()), y_t.cuda())
          error_sum += error_t.mean()
          mae_sum += mae_t.mean()
        print("Test RMSE:",error_sum,"Test MAE:",mae_sum)

def main():
  parser = argparse.ArgumentParser(description='InvarLSTM-Synthetic')
  parser.add_argument('--n', type=int, default=50000)
  parser.add_argument('--loss_type', default='ERM')
  args = parser.parse_args()
  train(**vars(args))

if __name__ == "__main__":
  main()