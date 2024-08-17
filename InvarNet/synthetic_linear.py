import torch
from torch.autograd import grad

torch.manual_seed(0)

def synthetic(n=10000, env=1.0, d=1):
  x = torch.zeros(n,d)
  y = torch.zeros(n,d)
  z = torch.zeros(n,d)

  x[0] = torch.randn(1)
  for i in range(1, n):
    x[i] = 0.2 * x[i-1] + torch.randn(1) * env
  y[0] = torch.randn(1)
  for j in range(1, n):
    y[j] = 0.2 * y[j-1] + x[j-1] + torch.randn(1) * env
  z[0] = torch.randn(1)
  for k in range(1, n):
    z[k] = 0.2 * z[k-1] + y[k-1] + torch.randn(1)

  feature = torch.cat((x,y,z), dim=1)
  target = torch.roll(y,1)
  return feature[1:], target[1:]

def compute_penalty (losses, dummy_w):
  g1 = grad(losses.mean(), dummy_w, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
  return (g1 * g2).sum()

phi = torch.nn.Parameter(torch.ones(3, 1))
dummy_w = torch.nn.Parameter(torch.Tensor ([1.0])).cuda()
opt = torch.optim.SGD([phi], lr=0.05)
mse = torch.nn.MSELoss(reduction="none")

a1,b1 = synthetic(n=15000, env=0.1)
a2,b2 = synthetic(n=10000, env=0.01)
a3,b3 = synthetic(n=5000, env=1.0)

train_envs = [[torch.cat((a1,a2),0), torch.cat((b1,b2),0)]]

environments=[synthetic(n=15000, env=0.1),synthetic(n=10000, env=0.01),synthetic(n=5000, env=1.0)]
test_envs = [synthetic(env=2.0)]

def train(n=50000, loss_type='ERM'):
  res = []
  for iteration in range(n):
    error = 0
    penalty = 0
    rex = []
    for x_e, y_e in environments:
      p = torch.randperm(len(x_e))
      error_e = mse(x_e[p].cuda() @ phi.cuda() * dummy_w, y_e[p].cuda())
      penalty += compute_penalty(error_e, dummy_w)
      error += error_e.mean()
      rex.append(error_e.mean())
    rex = torch.stack(rex)
    opt.zero_grad ()
    if loss_type == 'IRM':
      (error + 0.1 * penalty).backward()
    elif loss_type == 'ERM':
      error.backward()
    elif loss_type == 'REx':
      (1e-1 * error + rex.var()).backward()
    opt.step()
    if iteration % 1000 == 0:
      for x_t, y_t in test_envs:
        error_t = mse(x_t.cuda() @ phi.cuda(), y_t.cuda())
        print("Train Error:",error_t.mean())
      res.append(error_t.cpu())
  return res

if __name__ == '__main__':
  train(loss_type='ERM')


