import numpy as np

class gaussian():
  def __init__(self, action_dim,mu,std):
    self.shape = (1,action_dim)
    self.mu = mu
    self.std = std


  def reset(self):
    self.walk = np.zeros(self.shape)
    return self.walk


  def sample(self):
    self.walk = np.random.normal(loc=self.mu,scale=self.std,size=self.shape)
    return self.walk


class gaussian_walk(gaussian):
  def __init__(self,action_dim,mu,std):
    super().__init__(action_dim,dim,mu,std)


  def sample(self):
    self.walk += np.random.normal(loc=self.mu,scale=self.std,size=self.shape)
    return self.walk


class uhlenbeck_ornstein():
  def __init__(self, action_dim, mu, std, theta=.15, dt=1e-2):
    self.shape = (action_dim,1)
    self.mu = mu
    self.theta = theta
    self.std = std
    self.dt = dt


  def reset(self):
    self.walk = np.zeros(self.shape)
    return self.walk


  def sample(self):
    self.walk += self.theta * (self.mu - self.walk) * self.dt + np.sqrt(self.dt) * np.random.normal(loc=0.0,scale=self.std,size=self.shape)
    return self.walk