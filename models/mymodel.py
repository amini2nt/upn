
import gym
import pickle
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from tempfile import TemporaryFile
import numpy as np
import ipdb
import os
from skimage.transform import resize






#env = gym.make('Reacher-v2')
#env = DummyVecEnv([lambda: env])  # The algorithms require a vectorize$

#model = PPO2(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=10)
#model.save("Reacher-v2-model")
model = PPO2.load("Reacher-1e6-model.pkl")
env = gym.make('Reacher-v2')
env = DummyVecEnv([lambda: env])  
traj = 1
all_actions = []
all_imgs = []
all_states = []
while traj<=20000:
  print("trajectory"+str(traj))
  currentStepImages = []
  obs = env.reset()
  allStates = obs
  allActions = np.empty((0, 2))
  screen = env.render(mode='rgb_array') *255
  currentStepImages.append(resize(screen,(84,84)).astype(np.uint8).tolist())

  i = 1
#keep in mind that if we have n actions we have n+1 observations(states)
  while i: 
    action, _states = model.predict(obs)
    allActions = np.append(allActions,action,axis=0)
    obs, rewards, dones, info = env.step(action)
    allStates = np.append(allStates,obs,axis=0)
    screen = env.render(mode='rgb_array')*255
    currentStepImages.append(resize(screen,(84,84)).astype(np.uint8).tolist())


    if dones:
            break
    i +=1
  all_actions.append(allActions.tolist())
  all_states.append(allStates.tolist())
  all_imgs.append(currentStepImages)
  traj += 1
  
with open('all_imgs.pkl', 'wb') as f:
    pickle.dump(all_imgs, f)
with open('all_states.pkl', 'wb') as f:
    pickle.dump(all_states, f)
with open('all_actions.pkl', 'wb') as f:
    pickle.dump(all_states, f)




