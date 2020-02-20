from algorithms.dqn import *

class CDQN(DQN):
  """docstring for Categorical_DQN"""
  def __init__(self, D_in, D_out, atoms, v, nn_model, rb_model, device, name):
    super().__init__(D_in,D_out,nn_model,rb_model,device,name)
    self.atoms = atoms
    self.v_min = min(v)
    self.v_max = max(v)
    self.dz = (self.v_max - self.v_min) / (self.atoms - 1)
    self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)


  def target_calc(self, next_states, step_rewards, dones, discount):
    with torch.no_grad():
      next_actions = self.Q_hat(next_states).argmax(dim=1).long()
      batch = len(next_actions)
      probs = self.Q_hat.probs(next_states) #batch x D_out x atoms
      probs = probs[range(batch), next_actions] #batch x atoms

      t_z = step_rewards + (1 - dones) * discount * self.support
      t_z = t_z.clamp(min=self.v_min, max=self.v_max)
      b = (t_z - self.v_min) / self.dz
      l = b.floor().long()
      u = b.ceil().long()

      offset = (torch.linspace(0, (batch - 1) * self.atoms, batch).long()
               .unsqueeze(1).expand(batch, self.atoms).to(self.device))

      proj_probs = torch.zeros(probs.size(), device=self.device)
      proj_probs.view(-1).index_add_(0, (l + offset).view(-1), (probs * (u.float() - b)).view(-1))
      proj_probs.view(-1).index_add_(0, (u + offset).view(-1), (probs * (b - l.float())).view(-1))
    return proj_probs

  def loss(self, loss_fn, batch, discount):
    current_states, actions, step_rewards, dones, next_states = self.er_glance(batch)
    target = self.target_calc(next_states,step_rewards,dones,discount)
    prediction = self.Q.probs(current_states)[range(batch), actions.view(-1).long()]
    loss = -(target * prediction.log2()).sum(dim=1).mean() #cross entropy
    return loss