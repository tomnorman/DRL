import random
import numpy as np
from multiprocessing import Lock
from multiprocessing.managers import BaseManager


# random.seed(2)

class MemQueue:
  def __init__(self, mem=int(1e6)):
    self.buffer_size = mem
    self.init = False
    self.samples = 0
    self.refills = -1


  def reset(self):
    __init__(self.buffer_size)


  def __len__(self):
    return min(self.buffer_size, self.samples)

  
  def push(self, current_state, action, step_reward, done, next_state):
    if not self.init:
      self.current_state = np.zeros((self.buffer_size, current_state.shape[1]))
      self.action = np.zeros((self.buffer_size, action.shape[1]))
      self.step_reward = np.zeros((self.buffer_size, step_reward.shape[1]))
      self.done = np.zeros((self.buffer_size, done.shape[1]))
      self.next_state = np.zeros((self.buffer_size, next_state.shape[1]))
      self.init = True

    self.position = self.samples % self.buffer_size #cyclic position
    if self.position == 0:
      self.refills += 1
    self.current_state[self.position,:] = current_state
    self.action[self.position,:] = action
    self.step_reward[self.position,:] = step_reward
    self.done[self.position,:] = done
    self.next_state[self.position,:] = next_state

    self.samples += 1

  
  def glance(self, batch):
    '''random.sample raises ValueError'''
    n = self.__len__()
    indices = random.sample(range(n), batch)
    return self.current_state[indices,:], self.action[indices,:], self.step_reward[indices,:], self.done[indices,:], self.next_state[indices,:]


class AgentMemQueue(MemQueue):
  def __init__(self, mem=int(1e6)):
    super().__init__(mem)
    self.lock = Lock()


  def acquire(self):
    self.lock.acquire()


  def release(self):
    self.lock.release()


  def can_sample(self, batch):
    # self.lock.acquire()
    can = batch <= self.__len__()
    # self.lock.release()
    return can


  def l(self):
    return self.__len__()



# class DistMemQueue():
#   def __init__(self, N, agents, mem=int(1e6)):
#     self.N = N
#     self.agents = agents
#     self.buffer_size = mem // self.agents 
#     self.init = False
#     self.samples = np.zeros(self.agents,dtype=int)
#     self.refills = np.zeros(self.agents,dtype=int) - 1
#     self.position = np.zeros(self.agents, dtype=int)
#     self.starts = np.array([i * self.buffer_size for i in range(self.agents)], dtype=int)
#     self.locks = np.array([Lock() for _ in range(self.agents)])
#     self.init_lock = Lock()


#   def __len__(self):
#     return min(self.buffer_size, self.samples)

  
#   def push(self, agent, current_state, action, step_reward, done, next_state):
#     self.init_lock.acquire()
#     if not self.init:
#       self.current_state = np.zeros((self.buffer_size * self.agents, current_state.shape[1]))
#       self.action = np.zeros((self.buffer_size * self.agents, action.shape[1]))
#       self.step_reward = np.zeros((self.buffer_size * self.agents, step_reward.shape[1]))
#       self.done = np.zeros((self.buffer_size * self.agents, done.shape[1]))
#       self.next_state = np.zeros((self.buffer_size * self.agents, next_state.shape[1]))
#       self.init = True
#     self.init_lock.release()

#     self.locks[agent].acquire()
    
#     self.position[agent] = self.starts[agent] + self.samples[agent] % self.buffer_size #cyclic position[agent]
#     if self.position[agent] == 0:
#       self.refills += 1
#     self.current_state[self.position[agent],:] = current_state
#     self.action[self.position[agent],:] = action
#     self.step_reward[self.position[agent],:] = step_reward
#     self.done[self.position[agent],:] = done
#     self.next_state[self.position[agent],:] = next_state

#     self.samples[agent] += 1

#     self.locks[agent].release()


#   def can_sample(self):
#     for i in range(self.agents):
#       self.locks[i].acquire()

#     can = (self.samples >= self.N).all()

#     for i in range(self.agents):
#       self.locks[-i-1].release()

#     return can

  
#   def glance(self, batch):
#     '''can only be called after can_sample is True'''
#     for i in range(self.agents):
#       self.locks[i].acquire()

#     self.get_indices(batch)
#     c_s = []
#     a = []
#     s_r = []
#     d = []
#     n_s = []

#     for i in range(self.N):
#       c_s.append(self.current_state[self.indices[i,:,0]])
#       a.append(self.action[self.indices[i,:,0]])
#       s_r.append(self.step_reward[self.indices[i,:,0]])
#       d.append(self.done[self.indices[i,:,0]])
#       n_s.append(self.next_state[self.indices[i,:,0]])

#     c_s = np.stack(c_s, axis=0)
#     a = np.stack(a, axis=0)
#     s_r = np.stack(s_r, axis=0)
#     d = np.stack(d, axis=0)
#     n_s = np.stack(n_s, axis=0)

#     for i in range(self.agents):
#       self.locks[-i-1].release()

#     return c_s, a, s_r, d, n_s


#   def get_indices(self, batch):
#     self.indices = np.zeros((self.N,batch,1))
#     agents = np.random.choice(range(self.agents), batch)
#     for i,agent in enumerate(agents): #len(agents) == batch
#       first_index = max(0,self.samples[agent] - self.buffer_size)
#       last_index = self.samples[agent] - 1
#       if last_index - first_index + 1 < self.N:
#         raise "should call can_sample() first"
#       index = random.randrange(first_index, last_index+1)
#       while index + 1 - self.N < first_index: #N samples from the current state's past already been replaced
#         index = random.randrange(first_index, last_index+1)
#       index_arr = (index + 1 - self.N) % self.buffer_size
#       agent_indices = [self.starts[agent] + ((index_arr + n) % self.buffer_size) for n in range(self.N)]
#       self.indices[:,i,0] = agent_indices #in [0,:,:] is the past, [-1,:,:] is present
#       self.indices = self.indices.astype(int)



# class DistManager(BaseManager):  
  # pass

class AgentManager(BaseManager):
  pass

# def Manager():
  # m = DistManager()
  # return m

# DistManager.register('DistMemQueue', DistMemQueue)

AgentManager.register("AgentMemQueue", AgentMemQueue)

# def DistMemQueueWrapper(N,agents,mem):
  # m = DistManager()
  # m.start()
  # return m.DistMemQueue(N,agents,mem)

def AgentMemQueueWrapper(mem):
  m = AgentManager()
  m.start()
  return m.AgentMemQueue(mem)



# agents =20
# N=13
# mem = 600
# pushes = 650


# class MyManager(BaseManager):  
#   pass  

# numManager.register('DistMemQueue', DistMemQueue)
# mymanager = numManager()  
# mymanager.start()
# a=mymanager.DistMemQueue(N,agents,mem)



# def check(i_d,arr,a):
#   for ar in arr:
#     a.push(i_d,ar,ar,ar,ar,ar)

# p=[1 for _ in range(agents)]
# for i in range(agents):
#   arr = [i+np.zeros((1,2))+0.01*j for j in range(1,pushes+1)]
#   p[i] = Process(target=check, args=(i,arr,a))
#   p[i].start()

# for i in range(agents):
#   p[i].join()

# # print(a.glance(54)[0])
# test =a.glance(65)[2]
# print(test)
# # for i in range(N-1):
#   # print(((test[i]-test[i+1] + 0.01 < 0.001)).all())
