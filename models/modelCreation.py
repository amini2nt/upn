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

model.learn(total_timesteps=9000)
model.save("Reacher-10000-model")

model.learn(total_timesteps=90000)
model.save("Reacher-100000-model")


model.learn(total_timesteps=900000)
model.save("Reacher-1e6-model")



model.learn(total_timesteps=9000000)
model.save("Reacher-10e6-model")


model.learn(total_timesteps=1000)
model.save("Reacher-1000-model")
