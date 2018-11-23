import gym
import pickle
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np



env = gym.make('Reacher-v2')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorize$

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
model.save("Reacher-1000-model")
print("1000 done")
model.learn(total_timesteps=9000)
model.save("Reacher-10000-model")
print("10000 done")

model.learn(total_timesteps=90000)
model.save("Reacher-100000-model")
print("100000 done")


model.learn(total_timesteps=900000)
model.save("Reacher-1e6-model")
print("1000000 done")





