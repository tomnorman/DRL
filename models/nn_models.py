import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


def layer_init(layer,func): #layer acts as pointer
  param_dict = layer.state_dict()
  for key in param_dict.keys():
    func(param_dict[key])
  # layer.load_state_dict(param_dict)


class sequential_model(nn.Module):
  def __init__(self, layers_sizes, device):
    super().__init__()
    self.device = device
    self.layers_n = len(layers_sizes)-1
    self.fc = nn.ModuleList()
    for i in range(self.layers_n):
      self.fc.append(nn.Linear(layers_sizes[i],layers_sizes[i+1]).to(self.device))

  def forward(self, x, srl=False):
    x = x.to(self.device)
    for i, layer in enumerate(self.fc):
      last_layer = (i == (self.layers_n-1))
      if srl and last_layer:
        return x
      x = layer(x)
      if not last_layer:
        x = F.relu(x)
    return x


  def get_last(self):
    return self.fc[-1].state_dict()


  def load_last(self, src):
    self.fc[-1].load_state_dict(src)



class probs_sequential_model(nn.Module):
  def __init__(self, in_dim, h1, h2, out_dim, atoms, v, device):
    super().__init__()
    self.device = device
    self.atoms = atoms
    self.support = torch.linspace(min(v), max(v), self.atoms).to(self.device)
    self.out_dim = out_dim
    self.h1 = nn.Linear(in_dim, h1).to(self.device)
    self.h2 = nn.Linear(h1, h2).to(self.device)
    self.out = nn.Linear(h2, atoms * out_dim).to(self.device)

  def forward(self, x):
    probs = self.probs(x)
    Q = (probs * self.support).sum(dim=-1) #batch x out_dim
    return Q #expectation


  def probs(self, x):
    x = F.relu(self.h1(x))
    x = F.relu(self.h2(x))
    x = self.out(x).view(-1,self.out_dim,self.atoms) #batch x out_dim x atoms
    return F.softmax(x, dim=-1) #batch x out_dim x atoms


  def get_last(self):
    return self.out.state_dict()


  def load_last(self, src):
    self.out.load_state_dict(src.state_dict())


class actor_model(nn.Module):
  def __init__(self, in_dim, h1, h2, out_dim, lim, device):
    super().__init__()
    self.device = device
    self.h1 = nn.Linear(in_dim, h1).to(self.device)
    # self.h1_bn = nn.BatchNorm1d(in_dim)
    self.h2 = nn.Linear(h1, h2).to(self.device)
    # self.h2_bn = nn.BatchNorm1d(h1)
    self.out = nn.Linear(h2, out_dim).to(self.device)
    self.out.weight.data.uniform_(-lim,lim)
    self.out.bias.data.uniform_(-lim,lim)


  def forward(self, x):
    x = x.to(self.device)
    # x = self.h1_bn(x)
    x = F.relu(self.h1(x))
    # x = self.h2_bn(x)
    x = F.relu(self.h2(x))
    return self.out(x).tanh()


class critic_model(nn.Module):
  def __init__(self, state_dim, action_dim, h1, h2, lim, device):
    super().__init__()
    self.device = device
    self.h1 = nn.Linear(state_dim+action_dim, h1).to(self.device)
    self.h2 = nn.Linear(h1, h2).to(self.device)
    self.out = nn.Linear(h2, 1).to(self.device)

    self.out.weight.data.uniform_(-lim,lim)
    self.out.bias.data.uniform_(-lim,lim)


  def forward(self, states, actions, srl=False):
    states = states.to(self.device)
    actions = actions.to(self.device)
    x = torch.cat((states,actions), dim=1)
    x = F.relu(self.h1(x))
    x = F.relu(self.h2(x))
    if srl:
      return x
    return self.out(x)


  def get_last(self):
    return self.out.state_dict()


  def load_last(self, src):
    self.out.load_state_dict(src)



# class probability_critic_model(nn.Module):
#   def __init__(self, state_dim, action_dim, h1, h2, atoms, lim, device):
#     super().__init__()
#     self.device = device
#     self.state_seq = sequential_model([state_dim, h1, h2, atoms], self.device)
#     self.action_seq = sequential_model([action_dim, h1, atoms], self.device)


#   def forward(self, states, actions):
#     states = states.to(self.device)
#     actions = actions.to(self.device)
#     states = self.state_seq(states)
#     actions = self.action_seq(actions)
#     w = states + actions
#     return F.softmax(w, dim=1)



# class duel_model(nn.Module):
#   def __init__(self, seq_layers, duel_layers, D_out, device):
#     #D_out = number of actions
#     super().__init__()
#     self.device = device
#     seq_out = seq_layers[-1]
#     self.seq_model = sequential_model(seq_layers, True)
#     duel_layers = [seq_out] + duel_layers #add sequential model last layer output as duel model first layer input
#     self.V_model = sequential_model(duel_layers + [1]) #make duel model output size == 1 (value function)
#     self.A_model = sequential_model(duel_layers + [D_out]) #same as above (advantage function)
    
#   def forward(self, x):
#     x = x.to(self.device)
#     mid = self.seq_model(x)
#     V = self.V_model(mid)
#     A = self.A_model(mid)
#     A_mean = torch.mean(A,dim=1)[:,None] #make column vector
#     return V + (A - A_mean)