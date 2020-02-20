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



class DDDPG(DDPG):
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
    self.steps = hypers["steps"]
    self.discount = hypers["discount"]
    self.tau = hypers["tau"]
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
    self.drl_train()

    run.value = 0

    for p in processes:
      p.join()


  def drl_train(self):
    ex_rp = 0
    for step in range(self.steps):
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

      ex_rp += 1
      # plot
      if self.agent_write.value == 1:
        self.writer.add_scalar("Reward", self.agent_reward.value,ex_rp)
        self.agent_write.value = 0

      if ex_rp % 100 == 0:
        self.writer.add_scalar("Critic Loss", critic_loss.item(),ex_rp)
        self.writer.add_scalar("Actor Loss", actor_loss.item(),ex_rp)

      
      if ex_rp % 20000 == 0:
        self.save_nn_params(self.files_path+"_step_{}".format(ex_rp))
