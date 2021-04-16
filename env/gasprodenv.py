''' 
We are going to be utilising OpenAI Gym for the creation of the environment
So our dataset has 8 columns as follows:
a)Conductivity Layers
b) Conductivity
c) Days
d) Injection Rate
e) Injection Pressure 
f) Oil Rate
g) Water Rate
h) Production Pressure

Key words and values that we have to define in the environment 
a) Action Space
b) Observation_Space
c) Reset

for 3D env rendering we may need PyOpenGL
Remember that each layer has a conductivity value but is not unique, for example, conductivity layer 1,6,8,9 share the conductivity 
value 25
By varying the pressure (Independent variable) the conductivity values and production values (first of all I find this very vague)
are affected
Formulae for this correlation

~pressure(what we are searching for) = ~conductivity values(we already have this values) and 
                                        ~ production values(we are also searching for this values)
'''


#import necessary dependencies
import gym
import random
import numpy as np
from numpy import *
import pandas as pd

from gym  import spaces

#Create the environment constants
MIN_production_pressure = 0
Max_production_Pressure = 36000
Production_Pressure_Capacity = 36000
MAX_Injection_pressure = 48276
MIN_Injection_pressure = 0
MAX_Injection_Rate = 100
MIN_Injection_Rate = 0
MAX_Conductivity_Layers = 77
MIN_Conductivity_Layers = 1
MAX_Conductivity_Value = 677
MIN_Conductivity_Value = 0
MAX_Number_of_Days = 30194
MIN_Number_of_Days = 1
MAX_Water_Rate= 81
MIN_Water_Rate = 1
MAX_Oil_Rate = 100
MIN_Oil_Rate = 1
MAX_STEPS = 20000

INITIAL_Pressure = 0

class GasProdEnv(gym.Env):
    # a custom gas production environment for OpenAI gymn environment
    metadata = {"render.modes": ["human"]}
    
    #create and define the observation and action space
    def __init__(self, df):
        super(GasProdEnv, self).__init__()
        
        
        #the reward for our case should be headed towards a constant pressure
        self.df = df
        
        self.Production_Pressure_Capacity = Production_Pressure_Capacity
        self.MIN_Conductivity_Layers = MIN_Conductivity_Layers
        self.MAX_Conductivity_Layers = MAX_Conductivity_Layers
        self.MIN_production_pressure = MIN_production_pressure
        self.Max_production_Pressure = Max_production_Pressure
        self.MIN_Conductivity_Value = MIN_Conductivity_Value
        self.MAX_Conductivity_Value = MAX_Conductivity_Value
        
        #The pressure range is between 0 and 35548.52734
        self.reward_range = range(-1, 1)
        
        #create the action_space which am assuming am going to be using Discrete values if I can
        self.action_space = spaces.Discrete(self.Production_Pressure_Capacity)

        self.observation_space = spaces.Box(low=self.MIN_Conductivity_Value, high=self.MAX_Conductivity_Value, shape=(10, ), dtype=int32)
        """
        In arrays we have what we call the dtypes and are immutable just like strings and numbers in Python
        We have diffrent types of tensors ex: a scalar or a rank-0 tensor which is a scalar that contains one value
        and has no axes

        In the observation space below we create a vector or a rank-1 tensor and since it is not a numpy array and more of a tensor use the dtype as int32
        self.InitialProductionPressure = self.observation_space.sample()
        .sample returns a tuple of values and the random.randit() returns a scalar value
        """
    
    
    def _next_observation(self):
        if self.CurrentConductivityValue.shape > (5,):
          frame = np.take(self.CurrentConductivityValue, [-5, -4, -3, -2, -1])
        else:
          frame = self.CurrentConductivityValue
        # Append additional data and scale each value to between 0-1 for the observation space which is filled with whatever values that you defined in the reset
        added = np.array([self.Production_Pressure_Capacity,self.MIN_Conductivity_Value,self.MAX_Conductivity_Value,
            self.MIN_production_pressure,self.Max_production_Pressure]) 
        obs = np.append(frame, added)
        
        return obs
    
    
    def _take_action(self, action):
        
        for l in range(0, 77):#l here represents the no. of layers
            for t in range(0, 16):#t here represents the no. of iterations
                i = random.randint(0, 76)
                k = random.randint(0, 15)
                if self.CurrentConductivityValue[-1] == self.df.loc[:, 'Conductivity'].values[i][k] and self.current_step == self.df.loc[:, 'Layer'].values[i][k]:
                    reward = 1
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
                elif self.CurrentConductivityValue[-1] - 2 <= self.df.loc[:, 'Conductivity'].values[i][k] <= self.CurrentConductivityValue[-1] + 2 and self.current_step == self.df.loc[:, 'Layer'].values[i][k]:
                    reward = 0.5
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    #self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
                elif self.CurrentConductivityValue[-1] - 4 <= self.df.loc[:, 'Conductivity'].values[i][k] <= self.CurrentConductivityValue[-1] + 4 and self.current_step - 1 <= self.df.loc[:, 'Layer'].values[i][k] <= self.current_step + 1:
                    reward = 0.3
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]               
                elif self.CurrentConductivityValue[-1] - 6 <= self.df.loc[:, 'Conductivity'].values[i][k] <= self.CurrentConductivityValue[-1] + 6 and self.current_step - 1 <= self.df.loc[:, 'Layer'].values[i][k] <= self.current_step + 1:
                    reward = 0.1
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
                elif self.CurrentConductivityValue[-1] - 8 <= self.df.loc[:, 'Conductivity'].values[i][k] <= self.CurrentConductivityValue[-1] + 8 and self.current_step - 2 <= self.df.loc[:, 'Layer'].values[i][k] <= self.current_step + 2:
                    reward = 0.05
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
                elif self.CurrentConductivityValue[-1] - 10 <= self.df.loc[:, 'Conductivity'].values[i][k] <= self.CurrentConductivityValue[-1] + 10 and self.current_step - 2 <= self.df.loc[:, 'Layer'].values[i][k] <= self.current_step + 2:
                    reward = 0.01
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
                else:
                    reward = -1
                    self.days = self.df.loc[:, 'Days'].values[i][k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[i][k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[i][k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[i][k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[i][k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[i][k]
        
        
        self.InitialProductionPressure = np.random.uniform(self.MIN_production_pressure, self.Max_production_Pressure, 1)
        #production pressure can either rise, drop or remain as a constant for our actions
        #Production_Pressure reduction
        ProductionPressureRemovalQty = np.random.uniform(self.MIN_production_pressure, Production_Pressure_Capacity, 1)
        # updated Production Pressure after reduction of Pressure
        UpdatedProductionPressure = (self.InitialProductionPressure - ProductionPressureRemovalQty)
        # add ProductionPressure - action taken
        UpdatedProductionPressure_ = UpdatedProductionPressure +  action
        
        if UpdatedProductionPressure_ <= self.MIN_production_pressure:
            reward = -1
            done = True
        elif UpdatedProductionPressure_> self.Production_Pressure_Capacity:
            reward = -1
            done = True
        else:
            reward = 0.5
            done = False
    
    
    def step(self, action):
        #Execute one time step within the environment as dictated by the _take_action
        self._take_action(action)
        
        self.CurrentConductivityValue += 1
        
        delay_modifier = (self.CurrentConductivityValue[-1] / MAX_STEPS)
        
        reward = int(self.Production_Pressure_Capacity * delay_modifier)
        done = self.Production_Pressure_Capacity <= 0
        obs = self._next_observation()
        return obs, reward, done, {}
    
    
    def reset(self):
        """
        Reset the Environment
        """
        self.Production_Pressure_Capacity = 0
        self.MIN_Conductivity_Layers = 0
        self.MAX_Conductivity_Layers = 0
        self.MIN_production_pressure = 0
        self.Max_production_Pressure = 0
        
        ## Set the CurrentConductivityValue to a random point within the data frame
        self.CurrentConductivityValue = self.observation_space.sample()
        self.current_step = random.randint(1, len(self.df.loc[:, 'Layer'].values))
        
        return self._next_observation()
    
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Layer: {self.current_step}')
        print(f'Current Conductivity Value: {self.CurrentConductivityValue[-1]}')
        print(f'Days: {self.days}')
        print(f'Injection rate: {self.injection_rate}')
        print(f'Injection Pressure: {self.injection_pressure}')
        print(f'Oil rate: {self.oil_rate}')
        print(f'Water rate: {self.water_rate}')
        print(f'Production Pressure: {self.production_pressure}')