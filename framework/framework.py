import os
import pickle
import gym
from gym import wrappers
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter


print(os.getcwd())


# gif maker
from PIL import Image
import imageio
def create_gif(file_name, frames_list, duration):
  imageio.mimsave(file_name, frames_list, duration=duration)


# seed
import numpy as np
import torch, random
seed = 77

def seed_all(seed):
  np.random.seed(seed)
  seed_torch(seed)
  random.seed(seed)


def seed_torch(seed):
  torch.manual_seed(seed)
  if torch.backends.cudnn.enabled:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def question(question_str, answers, more_info=''):
  '''get answer from questin, 1 char for answer'''
  question_str = '\033[4m'+question_str+'\033[0m'+' '
  question_str += '/'.join(answers)+'\n'
  if more_info:
    question_str += more_info+'\n'
  ans = '' #just initialization, no meaning
  while ans not in answers:
    ans = input(question_str)
  return ans



class Player():
  def __init__(self, debug=20):
    self.debug = debug
    self.algorithms = {}
    self.i = 0


  def add(self, algorithm, hypers, reward_logic):
    self.algorithms[algorithm.name] = (algorithm, hypers, reward_logic)


  def play(self):
    if AutoPlayer.auto_players:
      return

    algorithms = list(self.algorithms.keys())
    ans = question("which algorithm to run?", algorithms+["exit"])
    if ans == "exit":
      return

    algorithm, hypers, reward_logic = self.algorithms[ans]
    self.run_algorithm(algorithm, hypers, reward_logic)


  def run_algorithm(self, algorithm, hypers, reward_logic, auto=False):
    # seed
    seed_all(seed)

    time = datetime.now().strftime("%d_%m_%y__%H_%M")
    cwd = os.getcwd()
    save_base_path = os.path.join(cwd,"data",algorithm.name)
    save_base_name = "trained"
    save_path = os.path.join(save_base_path,time+"__{}".format(AutoPlayer.auto_players))
    # os.makedirs(save_path, exist_ok=True)
    files_path = os.path.join(save_path,save_base_name)
    hypers["files_path"] = files_path

    print(algorithm)

    # create environment
    env_name = hypers["env_name"]
    env_colored = "\033[94m"+env_name+"\033[0m"
    print("gym environment: {}".format(env_colored))
    try:
      #for distributed algorithm
      env = []
      for i in range(hypers["agents"]):
        tmp = gym.make(env_name)
        tmp.seed(seed*i)
        env.append(tmp)
    except Exception as e:
      env = gym.make(env_name)
      env.seed(seed)
      # wrap to record
      path = os.path.join(cwd,"videos",algorithm.name, time)
      os.makedirs(path, exist_ok=True)
      show_world = lambda episode_id: not (episode_id % self.debug)
      env = wrappers.Monitor(env, path, video_callable=show_world, force=True)

    # train/test
    answers = ["train", "test", "skip"]
    if not auto:
      ans = question("train/test/skip?",answers)
    if auto or ans == answers[0]:
      #first dump hypers
      if not os.path.exists(save_path):
          os.makedirs(save_path)
      with open(files_path+"_hypers.pickle","wb") as fh:
        pickle.dump(hypers,fh)

      # train nn
      writer = SummaryWriter(save_path)
      algorithm.train(reward_logic,env,hypers,writer)
      writer.close()

      # save nn
      algorithm.save_nn_params(files_path)

      # test
      self.i += 1
      frames = algorithm.test(gym.make(env_name))
      create_gif(files_path+"_test_"+str(self.i)+".gif", frames, 0.01)

    if not auto and ans == answers[1]:
      # load weights
      loaded = False
      while not loaded:
        try:
          folder_name = input("folder name: (i.e. 31_12_19__18_54__1)\n")
          if folder_name == '':
            return self.play()
          files_path = os.path.join(save_base_path,folder_name,save_base_name)
          algorithm.load_nn_params(files_path)
          with open(files_path+"_hypers.pickle","rb") as fh:
            hypers = pickle.load(fh)
          loaded = True
        except Exception as e:
          print(e)

      # test
      self.i += 1
      for key in hypers.keys():
        print(key,':',hypers[key])
      frames = algorithm.test(gym.make(env_name))
      create_gif(files_path+"_test_"+str(self.i)+".gif", frames, 0.01)

    if not auto:
      return self.play()


  def env_dimensions(self, env_name):
    env = gym.make(env_name)
    try: #continuous
      state_dim = env.observation_space.shape[0]
    except Exception as e: #discrete
      state_dim = env.observation_space.n
    try: #continuous
      action_dim = env.action_space.shape[0]
    except Exception as e: #discrete
      action_dim = env.action_space.n
    env.close()
    return state_dim, action_dim



class AutoPlayer(Player):
  '''over night player'''
  auto_players = 0
  def __init__(self, algorithm, hypers, reward_logic, debug=20):
    AutoPlayer.auto_players += 1
    super().__init__(debug)
    super().run_algorithm(algorithm, hypers, reward_logic, auto=True)