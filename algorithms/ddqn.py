from algorithms.dqn import *

class DDQN(DQN):
  def __init__(self, D_in, D_out, nn_model, rb_model, device, name):
    super().__init__(D_in,D_out,nn_model,rb_model,device,name)

  def target_calc(self, next_states, step_rewards, dones, discount):
    batch = next_states.shape[0]
    with torch.no_grad():
      mask = 1 - dones
      maximise_actions_Q = self.Q(next_states).argmax(dim=1)
      max_Q_hat = self.Q_hat(next_states)[range(batch),maximise_actions_Q.long()][:,None]
      target = step_rewards + discount * mask * max_Q_hat
    return target

  def loss(self, loss_fn, batch, discount):
    current_states, actions, step_rewards, dones, next_states = self.er_glance(batch)
    target = self.target_calc(next_states,step_rewards,dones,discount)
    prediction = self.Q(current_states)[range(batch), actions.long().view(-1)]
    return loss_fn(prediction.view(-1),target.view(-1))