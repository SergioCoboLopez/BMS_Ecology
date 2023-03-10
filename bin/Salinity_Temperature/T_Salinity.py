#import libraries
#--------------------------------------------------
import pandas as pd
import numpy as np
import sys
import warnings
import gc

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from copy import deepcopy,copy
from ipywidgets import IntProgress
from itertools import chain
from IPython.display import display
from datetime import datetime
import pickle
import os

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
path = './rguimera-machine-scientist/machinescientist.py'
ms = SourceFileLoader('ms', path).load_module()
#--------------------------------------------------


# Read data
#--------------------------------------------------
path_data='../data/microbial_growth/'
data=pd.read_csv(path_data + 'Benson_Krause_Salinity.csv')


data['T']=deepcopy(data['Temperature (K)'])
data['Sal']=deepcopy(data['Salinity'])

XLABS=['T','Sal']
x=data[XLABS]
y=np.log(data['Concentration (micromol/dm3)'])

print(data.head())
print(x.head())
print(y.head())
#--------------------------------------------------

# Machine Scientist
#--------------------------------------------------------------------------
#Parameters
mcmc_resets = 1
mcmc_steps = 5000
params = 3

res={}

best_model, state_ensemble = ms.machinescientist(x=x,y=y,
                                               XLABS=XLABS,
                                               n_params=params,
                                               resets=mcmc_resets,
                                               steps_prod=mcmc_steps,
                                               log_scale_prediction=False)
#--------------------------------------------------------------------------
