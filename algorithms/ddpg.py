import torch, copy



class DDPG():
  def __init__(self, state_dim, action_dim,actor_nn_model, critic_nn_model, rb_model, noise_model, name, device):
    self.action_dim = action_dim
    self.state_dim = state_dim
    self.name = name
    self.q = rb_model
    self.Q = critic_nn_model #last layers isnt relu
    self.Q_hat = copy.deepcopy(self.Q)
    self.U = actor_nn_model
    self.U_hat = copy.deepcopy(self.U)
    self.noise_prcs = noise_model
    self.device = device


  def __str__(self):
    name_colored = '\033[91m'+self.name+'\033[0m'
    return 'algorithm: {}, available actions: {}, state size: {}'.format(name_colored,self.action_dim,self.state_dim)


  def target_calc(self, next_states, step_rewards, dones, discount):
    with torch.no_grad():
      mask = 1 - dones
      actions = self.U_hat(next_states)
      Q_hats = self.Q_hat(next_states,actions)
      target = step_rewards + discount * mask * Q_hats
    return target


  def soft_update(self, source, target, tau):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    if source_dict.keys() != target_dict.keys():
      raise Exception('source and target arent the same model')
    for key in source_dict:
      target_dict[key] = tau * source_dict[key] + (1-tau) * target_dict[key]
    target.load_state_dict(target_dict)


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
    critic_optimizer = hypers['critic_optimizer'](self.Q.parameters(), lr=hypers['critic_lr'], weight_decay=hypers['critic_wd'])
    actor_optimizer = hypers['actor_optimizer'](self.U.parameters(), lr=hypers['actor_lr'])
    loss_fn = hypers['loss_fn']
    batch = hypers['batch']
    steps = hypers['steps']
    discount = hypers['discount']
    tau = hypers['tau']


    # start training
    episode = 0
    done = True
    for step in range(steps):
      if done:
        if step > 0:
          # plot
          writer.add_scalar("Reward", reward, step)

          episode += 1
          string = "{}: episode {} done, step: {}, reward: {}".format(self.name, episode,step+1,reward)
          print(string)

        # new episode
        noise = self.noise_prcs.reset()
        reward = 0
        current_state = torch.Tensor([env.reset()]).view(1,-1)
        
      with torch.no_grad():
        pure_action = self.U(current_state).cpu().numpy()
      action = pure_action + noise
      noise = self.noise_prcs.sample()

      # play action
      next_state, step_reward, done, info = env.step(action.reshape(-1))
      step_reward = torch.Tensor([step_reward]).item()
      next_state = torch.Tensor([next_state]).view(1,-1)
      #reward_logic works with lists and numbers
      step_reward = reward_logic(current_state.tolist()[0], action, reward, step_reward, done, next_state.tolist()[0])
      #now step_reward is a number
      reward += step_reward

      # store action, uses numpy
      self.er_push(current_state, torch.Tensor(action), torch.Tensor([[step_reward]]), torch.Tensor([[done]]), next_state)

      # experience replay
      #start when have more than batch samples
      if len(self.q) >= batch:
        # sample
        current_states, actions, steps_rewards, dones, next_states = self.er_glance(batch)
        # create "supervised" y from Q_hat
        target = self.target_calc(next_states,steps_rewards,dones,discount)
        # create reward prediction from Q
        prediction = self.Q(current_states,actions)

        # train critic
        critic_loss = loss_fn(prediction, target).to(self.device)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # train actor
        actor_actions = self.U(current_states)
        actor_loss_tmp = self.Q(current_states,actor_actions)
        actor_loss = -torch.mean(actor_loss_tmp).to(self.device) #minus because ascension
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # update hats
        self.soft_update(self.Q, self.Q_hat, tau)
        self.soft_update(self.U, self.U_hat, tau)

        # plot
        if step % 100 == 0:
          writer.add_scalar("Critic Loss", critic_loss.item(),step)
          writer.add_scalar("Actor Loss", actor_loss.item(),step)
        
      current_state = next_state

    env.close()


  def test(self, env, string=None):
    done = False
    current_state = env.reset()
    frames = []
    with torch.no_grad():
      while not done:
        frames += [env.render(mode="rgb_array")]
        current_state = torch.Tensor([current_state]).view(1,-1)
        action = self.U(current_state.to(self.device)).cpu().numpy()
        current_state, _, done, _ = env.step(action.reshape(-1))
      env.close()
    return frames


  def save_nn_params(self, path):
    torch.save(self.Q.state_dict(), path+'_Q.pickle')
    torch.save(self.U.state_dict(), path+'_U.pickle')


  def load_nn_params(self, path):
    self.Q.load_state_dict(torch.load(path+'_Q.pickle'))
    self.U.load_state_dict(torch.load(path+'_U.pickle'))