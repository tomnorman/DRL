from algorithms.ddpg import *
import multiprocessing as mp
import torch.multiprocessing as t_mp
import numpy as np
import time



def actor(i, env, reward_logic, noise_prcs, rb_model, agent_net, run, write=None,reward_w=None, device="cpu"):
  done = True
  reward = 0
  step = 0
  while run.value:
    if done:
      if write:
        write.value=1
        reward_w.value=reward

      noise = noise_prcs.reset()
      current_state = torch.Tensor([env.reset()]).view(1,-1)
      reward = 0

    with torch.no_grad():
      action = agent_net(current_state.to(device)).to("cpu").numpy() + noise
    noise = noise_prcs.sample()

    next_state, step_reward, done, info = env.step(action.reshape(-1))
    next_state = torch.Tensor(next_state).view(1,-1)
    #reward_logic works with lists and numbers
    step_reward = reward_logic(current_state.tolist()[0], action, reward, step_reward, done, next_state.tolist()[0])
    reward += step_reward
    # store, uses numpy
    current_state = current_state.numpy().reshape(1,-1)
    action = np.array([action]).reshape(1,-1)
    step_reward = np.array([step_reward]).reshape(1,-1)
    done = np.array([done]).reshape(1,-1)

    rb_model.acquire()
    rb_model.push(current_state,action,step_reward,done,next_state.numpy())
    rb_model.release()

    current_state = next_state #env returns a vector
    step += 1

  env.close()



class LS_DDDPG(DDPG):
  def __init__(self, state_dim, action_dim, actor_nn_model, critic_nn_model, rb_model, noise_model, agents, past, name, device):
    super().__init__(state_dim, action_dim, actor_nn_model, critic_nn_model, rb_model, noise_model, name, device)
    self.agents = agents
    self.past = past
    # locks_init(self.agents)
    self.U.share_memory()


  def er_glance(self, sample):
    self.q.acquire()

    batch = self.q.l() if sample == -1 else sample
    current_states, actions, step_rewards, dones, next_states = self.q.glance(int(batch))

    self.q.release()

    current_states = torch.Tensor(current_states).to(self.device)
    actions = torch.Tensor(actions).to(self.device)
    step_rewards = torch.Tensor(step_rewards).to(self.device)
    dones = torch.Tensor(dones).to(self.device)
    next_states = torch.Tensor(next_states).to(self.device)
    return current_states, actions, step_rewards, dones, next_states


  def train(self, reward_logic, env, hypers, writer):
    self.writer = writer
    # parsing
    self.critic_optimizer = hypers["critic_optimizer"](self.Q.parameters(), lr=hypers["critic_lr"], weight_decay=hypers["critic_wd"])
    self.actor_optimizer = hypers["actor_optimizer"](self.U.parameters(), lr=hypers["actor_lr"])
    self.loss_fn = hypers["loss_fn"]
    self.batch = hypers["batch"]
    self.drl_steps = hypers["drl_steps"]
    self.srl_steps = hypers["srl_steps"]
    self.discount = hypers["discount"]
    self.tau = hypers["tau"]
    self.regu = hypers["regu"]
    self.files_path = hypers["files_path"]

    # actors
    run = mp.Value('i',1)
    self.agent_write = mp.Value('b',0)
    self.agent_reward = mp.Value('f',0.0)
    processes = []
    for i in range(self.agents):
      if i==0:
        p = t_mp.Process(target=actor, args=(i,env[i],reward_logic,self.noise_prcs[i],self.q,self.U,run,self.agent_write,self.agent_reward))
      else:
        p = t_mp.Process(target=actor, args=(i,env[i],reward_logic,self.noise_prcs[i],self.q,self.U,run))
      p.start()
      processes.append(p)

    while True:
      self.q.acquire()
      can = self.q.can_sample(self.batch)
      self.q.release()
      if can:
        break

    # start training
    self.episode = 0
    self.ex_rp = 0
    for srl_step in range(self.srl_steps):
      self.drl_train()
      self.srl_train()
      string = '{}: shallow rl update done, srl step: {}'.format(self.name,srl_step+1)
      print(string)

    run.value = 0

    for p in processes:
      p.join()


  def srl_train(self):
    current_states, actions, step_rewards, dones, next_states = self.er_glance(-1)
    self.generate_features(current_states, actions)
    target = self.target_calc(next_states,step_rewards,dones,self.discount)
    self.srl_update(target)


  def generate_features(self, current_states, actions):
    batch = current_states.shape[0] #batch
    with torch.no_grad():
      self.phi_s_a = self.Q(current_states,actions,srl=True) #batch x f

    # bias
    self.phi_s_a_bias = torch.ones(batch, 1, device=self.device)


  def srl_update(self, target):
    samples = target.shape[0] #amount of samples
    param_dict = self.Q.get_last() #last layer
    # weight
    A = (self.phi_s_a.T @ self.phi_s_a) / samples #[f x f]
    b = (self.phi_s_a.T @ target) / samples #[f x 1]
    first = (A + self.regu * torch.eye(A.shape[0],device=self.device)).inverse()
    w_last = param_dict['weight'].T
    second = (b + self.regu * w_last)
    w_last_new = first @ second
    param_dict['weight'] = w_last_new.T
    # bias
    A = (self.phi_s_a_bias.T @ self.phi_s_a_bias) / samples #[f x f]
    b = (self.phi_s_a_bias.T @ target) / samples #[f x 1]
    first = (A + self.regu * torch.eye(A.shape[0],device=self.device)).inverse()
    w_last = param_dict['bias'][:,None]
    second = (b + self.regu * w_last)
    w_last_new = (first @ second).view(-1)
    param_dict['bias'] = w_last_new

    self.Q.load_last(param_dict)


  def drl_train(self):
    for step in range(self.drl_steps):
      # experience replay
      # sample
      batch = self.batch // self.agents
      current_states, actions, step_rewards, dones, next_states = self.er_glance(batch)

      # create "supervised" y from Q_hat
      target = self.target_calc(next_states,step_rewards,dones,self.discount)
      # create reward prediction from Q
      prediction = self.Q(current_states,actions)

      ## train critic
      critic_loss = self.loss_fn(prediction, target).to(self.device)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # train actor
      actor_actions = self.U(current_states)
      actor_loss_tmp = self.Q(current_states,actor_actions)
      actor_loss = -torch.mean(actor_loss_tmp).to(self.device) #minus because ascension
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # update hats
      self.soft_update(self.Q, self.Q_hat, self.tau)
      self.soft_update(self.U, self.U_hat, self.tau)

      self.ex_rp += 1
      # plot
      if self.agent_write.value == 1:
        self.writer.add_scalar("Reward", self.agent_reward.value,self.ex_rp)
        self.agent_write.value = 0

      if self.ex_rp % 100 == 0:
        self.writer.add_scalar("Critic Loss", critic_loss.item(),self.ex_rp)
        self.writer.add_scalar("Actor Loss", actor_loss.item(),self.ex_rp)

      
      if self.ex_rp % 20000 == 0:
        self.save_nn_params(self.files_path+"_step_{}".format(self.ex_rp))
