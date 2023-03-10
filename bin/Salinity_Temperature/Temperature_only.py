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

def fit_ben_krs_fun(var, c1,c2,c3,c4,c5):
    return c1 + c2/var + c3 /(var **2.) + c4 /(var **3.) + c5/(var **4.)


# Import Machine Scientist
from importlib.machinery import SourceFileLoader
path = './rguimera-machine-scientist/machinescientist.py'
ms = SourceFileLoader('ms', path).load_module()

# Read data
#--------------------------------------------------
path_data='../data/microbial_growth/'
data=pd.read_csv(path_data + 'Benson_Krause.csv')

x=data['Temperature (K)']
data['T']=deepcopy(data['Temperature (K)'])
y=np.log(data['Concentration (micromol/dm3)'])

print(data.head())
print(y.head())

#--------------------------------------------------


# Machine Scientist
#Parameters
mcmc_resets = 1
mcmc_steps = 2000

XLABS = ['T']
params = 3

res={}

best_model, state_ensemble = ms.machinescientist(x=data,y=y,
                                               XLABS=XLABS,n_params=params,
                                               resets=mcmc_resets,
                                               steps_prod=mcmc_steps,
                                               log_scale_prediction=False)


parameters_dict=state_ensemble.par_values['d0']
Equation=state_ensemble.__eq__.__self__
print(Equation)
print(parameters_dict)



