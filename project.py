#Title#
# Colab stuff

#Code#
# import sys
# IN_COLAB = 'google.colab' in sys.modules
# if IN_COLAB:
#   from google.colab import drive
#   drive.mount("/content/gdrive")
#   %cd gdrive/My Drive/DRL
#   !apt-get install -y xvfb python-opengl > /dev/null 2>&1
#   !pip install gym pyvirtualdisplay > /dev/null 2>&1
#   !pip install JSAnimation==0.1
#   !pip install pyglet==1.3.2
  
#   from pyvirtualdisplay import Display
  
#   # Start virtual display
#   dis = Display(visible=0, size=(400, 400))
#   dis.start()

#   %matplotlib inline
#   %load_ext tensorboard
#   !pip install box2d-py
#   !pip install gym[Box_2D]

        
#Title#
# Imports

#Code#
# import gym
from framework.framework import *
from models.replay_buffer import *
from models.processes import *
from models.nn_models import *
from algorithms.dqn import *
from algorithms.ddqn import *
from algorithms.cdqn import *
from algorithms.ls_ddqn import *
from algorithms.ddpg import *
from algorithms.dddpg import *
from algorithms.ls_dddpg import *

#Title#
# Framework

#Code#
DEBUG = 20 #DEBUG- each 20 episodes -> save video
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("pytorch device:", device)
PLAYER = Player(DEBUG)
f = lambda current_state, action, reward, step_reward, done, next_state: step_reward
REWARD_LOGIC = f
# print("available environments",gym.envs.registry.all())

#Title#
# Discrete Environment

#Title#
# DQN + DDQN

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"LunarLander-v2",
  "optimizer":torch.optim.Adam, "lr":0.001,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "batch":64, "C":50, "steps":100000,
  "epsilon_start":1, "epsilon_end":0.05, "epsilon_steps":15000,
  "discount":0.99,
  "h1":128, "h2":256,
  "rb_mem":20000
}

for batch in [16, 32]:
  for C in [50, 150]:
    for epsilon_steps in [40000, 60000]:
      for rb_mem in [20000, 40000]:
        HYPERS = MAIN_HYPERS.copy()
        HYPERS["batch"] = batch
        HYPERS["C"] = C
        HYPERS["epsilon_steps"] = epsilon_steps
        HYPERS["rb_mem"] = rb_mem
        state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
        layers = [state_dim, HYPERS["h1"], HYPERS["h2"], action_dim]
        nn_model = sequential_model(layers,device)
        rb_model = MemQueue(HYPERS["rb_mem"])
        dqn = DQN(state_dim,action_dim,nn_model,rb_model,device,"dqn")
        ddqn = DDQN(state_dim,action_dim,nn_model,rb_model,device,"ddqn")
        # AutoPlayer(dqn,HYPERS,REWARD_LOGIC)
        # AutoPlayer(ddqn,HYPERS,REWARD_LOGIC)

PLAYER.add(dqn,MAIN_HYPERS,REWARD_LOGIC)
PLAYER.add(ddqn,MAIN_HYPERS,REWARD_LOGIC)


#Title#
# LS-DDQN

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"LunarLander-v2",
  "optimizer":torch.optim.Adam, "lr":0.001,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "batch":32, "C":100,
  "epsilon_start":1, "epsilon_end":0.05, "epsilon_steps":20000,
  "discount":0.99,
  "h1":128, "h2":256,
  "rb_mem":40000,
  "drl_steps":2000, "srl_steps":20, "regu":1
}

for drl_steps in [5000, 7000]:
  for srl_steps in [20, 50]:
      for regu in [.1, 1, 10, 100]:
        HYPERS = MAIN_HYPERS.copy()
        HYPERS["drl_steps"] = drl_steps
        HYPERS["srl_steps"] = srl_steps
        HYPERS["regu"] = regu
        state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
        layers = [state_dim, HYPERS["h1"], HYPERS["h2"], action_dim]
        nn_model = sequential_model(layers,device)
        rb_model = MemQueue(HYPERS["rb_mem"])
        lsddqn = LS_DDQN(state_dim,action_dim,nn_model,rb_model,device,"ls-ddqn")
        # AutoPlayer(lsddqn,HYPERS,REWARD_LOGIC)

PLAYER.add(lsddqn,MAIN_HYPERS,REWARD_LOGIC)


#Title#
# CDQN

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"LunarLander-v2",
  "optimizer":torch.optim.Adam, "lr":0.001,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "batch":64, "C":50, "steps":70000,
  "epsilon_start":1, "epsilon_end":0.05, "epsilon_steps":15000,
  "discount":0.99,
  "h1":128, "h2":256,
  "rb_mem":20000,
  "atoms":51, "v":[-200.0,200.0]
}

for batch in [16, 32]:
  for C in [50, 150]:
    for epsilon_steps in [40000, 60000]:
      for rb_mem in [20000, 40000]:
        HYPERS = MAIN_HYPERS.copy()
        HYPERS["batch"] = batch
        HYPERS["C"] = C
        HYPERS["epsilon_steps"] = epsilon_steps
        HYPERS["rb_mem"] = rb_mem
        state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
        layers = [state_dim, HYPERS["h1"], HYPERS["h2"], action_dim]
        nn_model = probs_sequential_model(state_dim,HYPERS["h1"],HYPERS["h2"],action_dim,HYPERS["atoms"],HYPERS["v"],"cpu")
        rb_model = MemQueue(HYPERS["rb_mem"])
        cdqn = CDQN(state_dim,action_dim,HYPERS["atoms"],HYPERS["v"],nn_model,rb_model,device,"cdqn")
        # AutoPlayer(cdqn,HYPERS,REWARD_LOGIC)


PLAYER.add(cdqn,MAIN_HYPERS,REWARD_LOGIC)


#Title#
##################################################################################################

#Title#
# Continuous Environment

#Title#
# DDPG, gaussian noise

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"BipedalWalker-v2",
  "critic_optimizer":torch.optim.Adam, "actor_optimizer":torch.optim.Adam, "critic_lr":1e-3, "actor_lr":1e-4, "critic_wd":1e-4,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "tau":0.001, "batch":128, "steps":500000,
  "discount":0.99,
  "lim":3e-3,
  "actor_h1":256, "actor_h2":256,
  "critic_h1":256, "critic_h2":256,
  "noise_mu":.0, "noise_std":.05,
  "rb_mem":50000
}


for batch in [64, 128]:
  for noise_std in [.05, .1, .3]:
      for rb_mem in [50000, 100000]:
        HYPERS = MAIN_HYPERS.copy()
        HYPERS["batch"] = batch
        HYPERS["noise_std"] = noise_std
        HYPERS["rb_mem"] = rb_mem
        state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
        actor_nn_model = actor_model(state_dim, HYPERS["actor_h1"], HYPERS["actor_h2"], action_dim, HYPERS["lim"],device)
        critic_nn_model = critic_model(state_dim, action_dim, HYPERS["critic_h1"], HYPERS["critic_h2"], HYPERS["lim"],device)
        rb_model = MemQueue(HYPERS["rb_mem"])
        noise_model = gaussian(action_dim,HYPERS["noise_mu"],HYPERS["noise_std"])
        ddpg_gaussian = DDPG(state_dim,action_dim,actor_nn_model,critic_nn_model,rb_model,noise_model,"ddpg", device)
        # AutoPlayer(ddpg_gaussian,HYPERS,REWARD_LOGIC)

PLAYER.add(ddpg_gaussian,MAIN_HYPERS,REWARD_LOGIC)


#Title#
# DDDPG, gaussian noise

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"BipedalWalker-v2",
  "critic_optimizer":torch.optim.Adam, "actor_optimizer":torch.optim.Adam, "critic_lr":1e-3, "actor_lr":1e-4, "critic_wd":1e-4,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "tau":0.001, "batch":64, "steps":500000,
  "discount":0.99,
  "lim":3e-3,
  "actor_h1":256, "actor_h2":256,
  "critic_h1":256, "critic_h2":256,
  "noise_mu":.0, "noise_std":.05,
  "rb_mem":50000,
  "agents":4, "past":1, #distributed
}


for rb_mem in [50000, 100000]:
  for noise_std in [.05, .1, .3]:
    HYPERS = MAIN_HYPERS.copy()
    HYPERS["noise_std"] = noise_std
    HYPERS["rb_mem"] = rb_mem
    state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
    actor_nn_model = actor_model(state_dim, HYPERS["actor_h1"], HYPERS["actor_h2"], action_dim, HYPERS["lim"],"cpu")
    critic_nn_model = critic_model(state_dim, action_dim, HYPERS["critic_h1"], HYPERS["critic_h2"], HYPERS["lim"],device)
    rb_model = AgentMemQueueWrapper(HYPERS["rb_mem"])
    noise_model = [gaussian(action_dim,HYPERS["noise_mu"],HYPERS["noise_std"]) for _ in range(HYPERS["agents"])]
    dddpg_gaussian = DDDPG(state_dim,action_dim,actor_nn_model,critic_nn_model,rb_model,noise_model,HYPERS["agents"],HYPERS["past"],"dddpg", device)
    # AutoPlayer(dddpg_gaussian,HYPERS,REWARD_LOGIC)

PLAYER.add(dddpg_gaussian,MAIN_HYPERS,REWARD_LOGIC)



#Title#
# LS-DDDPG, gaussian noise

#Code#
### architecture ###
MAIN_HYPERS = {
  "env_name":"BipedalWalker-v2",
  "critic_optimizer":torch.optim.Adam, "actor_optimizer":torch.optim.Adam, "critic_lr":1e-3, "actor_lr":1e-4, "critic_wd":1e-4,
  "loss_fn":torch.nn.SmoothL1Loss(),
  "tau":0.001, "batch":64, "steps":500000,
  "discount":0.99,
  "lim":3e-3,
  "actor_h1":256, "actor_h2":256,
  "critic_h1":256, "critic_h2":256,
  "noise_mu":.0, "noise_std":.3,
  "rb_mem":50000,
  "drl_steps":10, "srl_steps":50, "regu":100, #srl
  "agents":4, "past":1, #distributed
}


for drl_steps in [35000, 50000]:
  for srl_steps in [10, 15]:
      for regu in [.1,1,10]:
        HYPERS = MAIN_HYPERS.copy()
        HYPERS["drl_steps"] = drl_steps
        HYPERS["srl_steps"] = srl_steps
        HYPERS["regu"] = regu
        state_dim, action_dim = PLAYER.env_dimensions(HYPERS["env_name"])
        actor_nn_model = actor_model(state_dim, HYPERS["actor_h1"], HYPERS["actor_h2"], action_dim, HYPERS["lim"],"cpu")
        critic_nn_model = critic_model(state_dim, action_dim, HYPERS["critic_h1"], HYPERS["critic_h2"], HYPERS["lim"],device)
        rb_model = AgentMemQueueWrapper(HYPERS["rb_mem"])
        noise_model = [gaussian(action_dim,HYPERS["noise_mu"],HYPERS["noise_std"]) for _ in range(HYPERS["agents"])]
        ls_dddpg_gaussian = LS_DDDPG(state_dim,action_dim,actor_nn_model,critic_nn_model,rb_model,noise_model,HYPERS["agents"],HYPERS["past"],"ls-dddpg", device)
        # AutoPlayer(ls_dddpg_gaussian,HYPERS,REWARD_LOGIC)

PLAYER.add(ls_dddpg_gaussian,MAIN_HYPERS,REWARD_LOGIC)


#Title#
# PLAY

#Code#
PLAYER.play()