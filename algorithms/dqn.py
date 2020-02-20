import torch, copy, random



class DQN():
  def __init__(self, state_dim, action_dim, nn_model, rb_model, device, name):
    self.device = device
    self.action_dim = action_dim
    self.state_dim = state_dim
    self.q = rb_model
    self.Q = nn_model.to(self.device)
    self.Q_hat = copy.deepcopy(self.Q)
    self.Q_hat.eval()
    self.name = name


  def __str__(self):
    name_colored = '\033[91m'+self.name+'\033[0m'
    return 'algorithm: {}, available actions: {}, state dimension: {}'.format(name_colored,self.action_dim,self.state_dim)


  def epsilon_greedy(self, current_state, epsilon):
    sample = torch.rand(1).item()
    if sample < epsilon:
      #exploration
      action = random.randrange(self.action_dim)
    else:
      #exploitation
      with torch.no_grad():
        action = self.Q(current_state).argmax().item()
    return action


  def target_calc(self, next_states, step_rewards, dones, discount):
    with torch.no_grad():
      mask = 1 - dones
      max_Q_hat, _ = self.Q_hat(next_states).max(dim=1, keepdim=True)
      target = step_rewards + discount * mask * max_Q_hat
    return target


  def loss(self, loss_fn, batch, discount):
    current_states, actions, step_rewards, dones, next_states = self.er_glance(batch)
    target = self.target_calc(next_states,step_rewards,dones,discount)
    prediction = self.Q(current_states)[range(batch), actions.long().view(-1)]
    return loss_fn(prediction.view(-1),target.view(-1))


  def er_glance(self, batch):
    current_states, actions, step_rewards, dones, next_states = self.q.glance(batch)
    current_states = torch.Tensor(current_states).to(self.device)
    actions = torch.Tensor(actions).to(self.device)
    step_rewards = torch.Tensor(step_rewards).to(self.device)
    dones = torch.Tensor(dones).to(self.device)
    next_states = torch.Tensor(next_states).to(self.device)
    return current_states, actions, step_rewards, dones, next_states


  def er_push(self, *args):
    current_state = args[0].numpy()
    action = args[1].numpy()
    step_reward = args[2].numpy()
    done = args[3].numpy()
    next_state = args[4].numpy()
    self.q.push(current_state,action,step_reward,done,next_state)


  def train(self, reward_logic, env, hypers, writer):
    # parsing
    optimizer = hypers['optimizer'](self.Q.parameters(), lr=hypers['lr'])
    loss_fn = hypers['loss_fn']
    epsilon_steps = hypers['epsilon_steps']
    epsilon_start,epsilon_end = hypers['epsilon_start'],hypers['epsilon_end']
    epsilon = epsilon_start
    epsilon_delta = (epsilon_start - epsilon_end) / epsilon_steps
    batch = hypers['batch']
    C = hypers['C']
    steps = hypers['steps']
    discount = hypers['discount']

    # start training
    episode = 0
    ex_rp = 0
    done = True
    for step in range(steps):
      if done:
        if step > 0:
          # plot
          writer.add_scalar("Reward", reward, ex_rp)

          episode += 1
          string = "{}: episode {} done, step: {}, reward: {}".format(self.name, episode, step+1,reward)
          print(string)

        # new episode
        reward = 0
        current_state = torch.Tensor([env.reset()]).view(1,-1)
     
      # choose action (eps greedy)
      action = self.epsilon_greedy(current_state.to(self.device), epsilon)

      # play action
      next_state, step_reward, done, info = env.step(action)
      step_reward = torch.Tensor([step_reward]).item()
      next_state = torch.Tensor([next_state]).view(1,-1)
      #reward_logic works with lists
      step_reward = reward_logic(current_state.tolist()[0], action, reward, step_reward, done, next_state.tolist()[0])
      reward += step_reward

      # store action, uses numpy
      self.er_push(current_state, torch.Tensor([[action]]), torch.Tensor([[step_reward]]), torch.Tensor([[done]]), next_state)

      # experience replay
      #start when have more than batch samples
      if len(self.q) >= batch:
        #experience replay counter
        ex_rp += 1

        # train
        loss = self.loss(loss_fn, batch, discount)
        # loss = loss_fn(y_prediction,y_target).to(self.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ex_rp % C == 0:
          self.Q_hat.load_state_dict(self.Q.state_dict())

        if ex_rp < epsilon_steps:
          epsilon -= epsilon_delta

        # plot
        if step % 100 == 0:
          writer.add_scalar("Loss", loss.item(),step)
        
      current_state = next_state

    env.close()


  def test(self, env):
    frames = []
    done = False
    current_state = env.reset()
    with torch.no_grad():
      while not done:
        frames += [env.render(mode="rgb_array")]
        current_state = torch.Tensor([current_state]).view(1,-1)
        #only greedy
        action = self.Q(current_state.to(self.device)).argmax().item()
        current_state, _, done, _ = env.step(action)
      env.close()
    return frames


  def save_nn_params(self, path):
    torch.save(self.Q.state_dict(), path+'_Q.pickle')


  def load_nn_params(self, path):
    self.Q.load_state_dict(torch.load(path+'_Q.pickle'))
