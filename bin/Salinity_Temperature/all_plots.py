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

#10. Relative errors of simplified dynamics
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Get_Relative_Error(True_Dynamics_Vec,Model_Dynamics_Vec):
    
    Bacteria_True=True_Dynamics_Vec[0];Bacteria_Model=Model_Dynamics_Vec[0]
    
    Rel_Err_B=[]
    
    for (BT, BM) in zip(Bacteria_True, Bacteria_Model):
        
        Relative_Error_B_i=(np.absolute(BT-BM))/BT
        
        Rel_Err_B.append(Relative_Error_B_i)
        
        
    Rel_Err_P=[]
    
    Phage_True=True_Dynamics_Vec[1];Phage_Model=Model_Dynamics_Vec[1]
    
    for (PT, PM) in zip(Phage_True, Phage_Model):
        
        Relative_Error_P_i=(np.absolute(PT-PM))/PT
        
        Rel_Err_P.append(Relative_Error_P_i)
        
        
    Rel_Error=[np.mean((bacteria,phage)) for (bacteria,phage) in zip(Rel_Err_B,Rel_Err_P)]
    
    #Mean values
    
    #---------------------------------------------------------------
    Mean_Rel_Error_Bacteria=np.mean(Rel_Err_B)
    
    Mean_Rel_Error_Phage=np.mean(Rel_Err_P)
    
    Mean_Rel_Error=np.mean([Mean_Rel_Error_Phage,Mean_Rel_Error_Bacteria])
    
    #---------------------------------------------------------------
    
    return Rel_Error, Rel_Err_B, Rel_Err_P
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Get_Relative_Error_Test(True_Dynamics_Vec,Model_Dynamics_Vec):
    
    Bacteria_True=True_Dynamics_Vec;Bacteria_Model=Model_Dynamics_Vec
    
    Rel_Err_B=[]
    for (BT, BM) in zip(Bacteria_True, Bacteria_Model):        
        Relative_Error_B_i=(np.absolute(BT-BM))/BT
        Rel_Err_B.append(Relative_Error_B_i)
        
        
#    Rel_Error=[np.mean((bacteria,phage)) for (bacteria,phage) in zip(Rel_Err_B,Rel_Err_P)]

    Rel_Error=[bacteria for bacteria in Rel_Err_B]
    
    #Mean values
    #---------------------------------------------------------------
    Mean_Rel_Err_Bacteria=np.mean(Rel_Err_B)
    #---------------------------------------------------------------
    
    return Rel_Error, Mean_Rel_Err_Bacteria
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




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

#Benson-Krause fit
fit_ben_krs=[fit_ben_krs_fun(T,-135.902,1.575701e5,-6.642308e7,1.2438e10,-8.621949e11)\
             for T in data['Temperature (K)']]

def machine_scientist_best_model(T, a1,a2,a3,a4,a5):
    return 

#Plot Data
#===================================================================

#Figure size and resolution, text sizes
#......................................................
cm = 1/2.54  # centimeters in inches
Width=8*cm;Height=4*cm #Width and height of plots
figure(figsize=(Width, Height), dpi=300) #Resolution
size_axis=7;size_ticks=5;size_title=5
#......................................................

#Axes, title and ticks
#......................................................
plt.ylabel('O2 concentration (micromol/dm3)',fontsize=size_axis)
plt.xlabel('Temperature (K)',fontsize=size_axis)
plt.xticks(fontsize=size_ticks);plt.yticks(fontsize=size_ticks)
sns.despine(top=True, right=True, left=False, bottom=False) 
#......................................................

plt.plot(data['Temperature (K)'],fit_ben_krs,color='blue',linewidth=1)
plt.scatter(data['Temperature (K)'],y,color='red',s=1.5)


def MS_best_model_3p(T, parameters_var):
    _a0_=parameters_var['_a0_']
    _a1_=parameters_var['_a1_']
    _a2_=parameters_var['_a2_']

    return ((_a1_ + (_a0_ * T)) / (_a2_ + (T ** 2)))

parameters_dict_3p={'_a0_':2512.4949432011367,'_a1_':-509254.6736332924,'_a2_': -45708.77376730233}

def MS_best_model_5p(T, parameters_var):
    _a0_=parameters_var['_a0_']
    _a1_=parameters_var['_a1_']
    _a2_=parameters_var['_a2_']
    _a3_=parameters_var['_a3_']
    _a4_=parameters_var['_a4_']

    return ((np.sinh((_a0_ + (_a4_ / T))) + (_a0_ * _a2_)) / (_a3_ + (T * (_a1_ + (_a4_ * T)))))

parameters_dict_5p={'_a0_': 17.13157657284335, '_a1_': -595995.4234955711, '_a2_': -4926847.202176255, '_a3_': 126219348.23597056, '_a4_': 707.632382485426}

Test_5p=[MS_best_model_5p(T, parameters_dict_5p) for T in data['Temperature (K)']]
Test_3p=[MS_best_model_3p(T, parameters_dict_3p) for T in data['Temperature (K)']]

print(len(Test_3p))
print(type((Test_3p)))
print(y)
test_data=y.tolist()
print(test_data)
print(len(y))
print(type((y)))

test_error_5p,test_error_mean_5p=Get_Relative_Error_Test(test_data,Test_5p)
test_error_3p,test_error_mean_3p=Get_Relative_Error_Test(test_data,Test_3p)
test_error_fit,test_error_mean_fit=Get_Relative_Error_Test(test_data,fit_ben_krs)

print(test_error_mean_3p)
print(test_error_mean_5p)
print(test_error_mean_fit)

plt.plot(data['Temperature (K)'],Test_3p,color='green',linewidth=1)
plt.show()


figure(figsize=(Width, Height), dpi=300) #Resolution

#Axes, title and ticks
#......................................................
plt.ylabel('O2 concentration (micromol/dm3)',fontsize=size_axis)
plt.xlabel('Temperature (K)',fontsize=size_axis)
plt.xticks(fontsize=size_ticks);plt.yticks(fontsize=size_ticks)
sns.despine(top=True, right=True, left=False, bottom=False) 
#......................................................

plt.plot(data['Temperature (K)'],test_error_3p,color='green',linewidth=1,label='3 parameters')
plt.plot(data['Temperature (K)'],test_error_5p,color='red',linewidth=1,label='5 parameters')
plt.plot(data['Temperature (K)'],test_error_fit,color='blue',linewidth=1,label='BK-Fit')
plt.legend(loc='best',fontsize=size_ticks)
plt.show()



#===================================================================



