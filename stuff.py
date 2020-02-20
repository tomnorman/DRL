import imageio

# create gif from list of PIL images
def create_gif(file_name, frames_list, duration):
  imageio.mimsave(file_name, frames_list, duration=duration)


import matplotlib.pyplot as plt

from random import random

def drunken_walk(N,n):
  uniform_arr = np.random.uniform(low=0.0, high = 1.0, size=(N,n))
  uniform_arr[uniform_arr < 0.5] = -1.0
  uniform_arr[uniform_arr >= 0.5] = 1.0
  walk = np.cumsum(uniform_arr, axis=1)
  return walk


# plot proccesses
def plot(arr,title,wrong=False):
  fig, ax = plt.subplots(1,1)
  if not wrong:
    ax.plot(arr.T) #plot columns
  else:
    ax.plot(arr)
  fig.suptitle(title)
  plt.show()


def create_walk(n, action_dim,mu,theta,std,dt):
  uhl= uhlenbeck_ornstein(action_dim,mu,theta,std,dt)
  uhl= gaussian(action_dim,mu,std)
  uhl.reset()
  walk = uhl.reset()
  for a in range(n):
    walk = np.hstack((walk,uhl.sample()))
  return walk

# cool0=create_walk(int(1e5),10,0.0,1.0,0.3,1)
# plot(cool0, 'gaussian walk')
# cool1 = drunken_walk(10,int(1e3))
# plot(cool1, 'drunk',True)
# cool2 = gaussian_walk(10,int(1e3),0,1)
# plot(cool2, 'gauss',True)