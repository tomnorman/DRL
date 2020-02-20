from algorithms.ddqn import *



class LS_DDQN(DDQN):
  def __init__(self, state_dim, action_dim, nn_model, rb_model, device, name):
    super().__init__(state_dim,action_dim,nn_model,rb_model,device,name)


  def train(self, reward_logic, env, hypers, writer):
    self.writer = writer
    self.reward_logic = reward_logic
    self.env = env

    # parsing
    self.optimizer = hypers['optimizer'](self.Q.parameters(), lr=hypers['lr'])
    self.loss_fn = hypers['loss_fn']
    self.epsilon_steps = hypers['epsilon_steps']
    epsilon_start,epsilon_end = hypers['epsilon_start'],hypers['epsilon_end']
    self.epsilon = epsilon_start
    self.epsilon_delta = (epsilon_start - epsilon_end) / self.epsilon_steps
    self.batch = hypers['batch']
    self.C = hypers['C']
    self.drl_steps = hypers['drl_steps']
    self.srl_steps = hypers['srl_steps']
    self.discount = hypers['discount']
    self.regu = hypers['regu']

    # start training
    self.done = True
    self.episode = 0
    self.ex_rp = 0
    for srl_step in range(self.srl_steps):
      self.drl_train()
      self.srl_train()
      string = '{}: shallow rl update done, srl step: {}'.format(self.name,srl_step+1)
      print(string)

    self.env.close()


  def srl_train(self):
    data_len = len(self.q)
    current_states, actions, step_rewards, dones, next_states = self.er_glance(data_len)
    self.generate_features(current_states, actions)
    target = self.target_calc(next_states,step_rewards,dones,self.discount)
    self.srl_update(target)


  def srl_update(self, target):
    samples = target.shape[0] #amount of samples
    param_dict = self.Q.get_last() #last layer

    # weight
    A = (self.phi_s_a.T @ self.phi_s_a) / samples #[f*|A| x f*|A|]
    b = (self.phi_s_a.T @ target) / samples #[f*|A| x 1]
    first = (A + self.regu * torch.eye(A.shape[0],device=self.device)).inverse()
    w_last = param_dict['weight'].T.reshape(-1,1)
    second = (b + self.regu * w_last)
    w_last_new = first @ second
    param_dict['weight'] = w_last_new.reshape(param_dict['weight'].T.shape).T.detach()
    #complicated way to get to the same row,column order the original weight layer was organized

    # bias
    biases = param_dict['bias'][None,:]
    A = (self.phi_s_a_bias.T @ self.phi_s_a_bias) / samples #[f*|A| x f*|A|]
    b = (self.phi_s_a_bias.T @ target) / samples #[f*|A| x 1]
    first = (A + self.regu * torch.eye(A.shape[0],device=self.device)).inverse()
    w_last = biases.T
    second = (b + self.regu * w_last)
    w_last_new = (first @ second).view(-1)
    param_dict['bias'] = w_last_new

    self.Q.load_last(param_dict)


  def generate_features(self, current_states, actions):
    data_len = current_states.shape[0] #batch
    phi_s = self.Q(current_states,srl=True) #batch x f
    # augmantation
    f = phi_s.shape[1] #number of neurons in last hidden layer
    self.phi_s_a = torch.zeros(data_len, f * self.action_dim, device=self.device) #[data_len x f*|A|]
    #for each row
    rows = torch.arange(data_len).reshape(actions.shape).long().to(self.device) #should be vector
    #and each of these columns
    cols = (actions.long() * self.action_dim).long()
    #duplicate cols to be [data_len x self.action_dim]
    cols = cols.repeat(1,f)
    #add to arange so every cols[i,:] is equal to the indices of phi_s_a to change
    cols += torch.arange(f, device=self.device)
    cols = cols.long().to(self.device)
    self.phi_s_a[rows,cols] = phi_s

    # bias
    self.phi_s_a_bias = torch.zeros(data_len, self.action_dim,device=self.device)
    cols = actions.long()
    self.phi_s_a_bias[rows,cols] = 1


  def drl_train(self):
    for step in range(self.drl_steps):
      if self.done:
        if step > 0:
          self.writer.add_scalar("Reward",self.reward,self.ex_rp)
          string = "{}: episode {} done, drl step: {}, reward: {}".format(self.name,self.episode,step+1,self.reward)
          print(string)

        # new episode
        self.reward = 0
        self.episode += 1
        self.current_state = torch.Tensor([self.env.reset()])

      # choose action (eps greedy)
      action = self.epsilon_greedy(self.current_state, self.epsilon)

      # play action
      self.next_state, step_reward, self.done, info = self.env.step(action)
      self.next_state = torch.Tensor([self.next_state])
      #reward_logic works with lists
      step_reward = self.reward_logic(self.current_state.tolist()[0], action, self.reward, step_reward, self.done, self.next_state.tolist()[0])
      self.reward += step_reward

      # store action
      self.er_push(self.current_state, torch.Tensor([[action]]), torch.Tensor([[step_reward]]), torch.Tensor([[self.done]]), self.next_state)

      # experience replay
      #start when have more than batch samples
      if len(self.q) >= self.batch:
        #experience replay counter
        self.ex_rp += 1
        # sample
        current_states, actions, step_rewards, dones, next_states = self.er_glance(self.batch)
        # create "supervised" y from Q_hat
        target = self.target_calc(next_states,step_rewards,dones,self.discount)
        # create reward prediction from Q
        prediction = self.Q(current_states).gather(dim=1, index=actions.long())

        # train
        loss = self.loss_fn(prediction,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # plot
        if self.ex_rp % 100 == 0:
          self.writer.add_scalar("Loss", loss.item(),self.ex_rp)
        
        if self.ex_rp % self.C == 0:
          self.Q_hat.load_state_dict(self.Q.state_dict())

        if self.ex_rp < self.epsilon_steps:
          self.epsilon -= self.epsilon_delta

      self.current_state = self.next_state