""" Installing the packages
! pip install gym
! pip install tensorflow==1.15
! pip install stable-baselines
"""
#import all the necessary dependency libraries
import gym
import datetime as dt
import pandas as pd
import numpy as np

#the stable_baselines is used for the model training in RL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

#This is used for checking the environment to see whether we will get any errors once the environment is done
from stable_baselines.common.env_checker import check_env

#importing our just created environment and using the class that we have just created 
from env.gasprodenv import GasProdEnv


#preprocessing the data given
df = pd.read_csv('/home/ningi/Desktop/derrick/RL_for_matching_values/data/rl_test_data.csv')
df = df.drop([0, 1, 2, 3]).reset_index()
df = df.drop(columns=['index', 'Unnamed: 2', 'Unnamed: 9', 'Unnamed: 18', 
                      'Unnamed: 27', 'Unnamed: 36', 'Unnamed: 45','Unnamed: 54', 
                      'Unnamed: 63', 'Unnamed: 72', 'Unnamed: 81', 'Unnamed: 90', 
                      'Unnamed: 99', 'Unnamed: 108', 'Unnamed: 117', 'Unnamed: 126', 
                      'Unnamed: 135'])
#grab the first row for the header
new_header = df.iloc[0]

#take the data less the header row
df = df[1:]
#set the header row as the df header
df.columns = new_header
df.rename(columns=df.iloc[0])
df = df.reset_index()
df = df.drop(columns='index')
df = df[: -1426]
df = df.apply(pd.to_numeric)
df.isnull()
df.isnull().sum()#returns the column names with the null values plus the sum

"""
~~~~~~~~~~~~~~~~~Understanding the dataset given~~~~~~~~~

looking at the data we already know that the conductivity layers column is just 77 but how do we deal with this
From the four guidelines, guideline no. 2 stipulates that each layer in the dataset has a corresponding conductivity value which is 
ideally true and meaning there are as many conductivity values as there are conductivity layers which is 77 but on a lighter
note looking at our data:
a) The no of days still continue to increase past the corresponding 77th layer
b) The Injection rate is a constant past the 77th layer
c) The Injection pressure has a flunctuating trend past the 77th layer
d) The oil Rate has a decreasing trend past the 77th layer
e) The water Rate is a constant value as seen (0)
f)  The production Pressure is a constant value past the 77th layer(3500)
"""
"""
In the case where you want to randomize an agent to run in your custom env here is the code to do so

env = GasProdEnv()
obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
"""
 
#this allows for one to see all the descriptive data even in the case when the dataframe is too large
pd.set_option('max_columns', None)
print(df.head())

#The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: GasProdEnv(df)])
state = GasProdEnv(df).reset()
print(f'Required shape: {GasProdEnv(df).observation_space.shape}, state: {state.shape}')
print(f'low: {GasProdEnv(df).observation_space.low}, checking: {np.all(state >= GasProdEnv(df).observation_space.low)}')
print(f'high: {GasProdEnv(df).observation_space.high}, checking: {np.all(state <= GasProdEnv(df).observation_space.high)}')

check_env(GasProdEnv(df))

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=250)
obs = env.reset()
for i in range(250):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()