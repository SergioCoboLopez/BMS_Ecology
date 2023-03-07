#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import pandas as pd
import numpy as np
import sys
import warnings
import gc
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from copy import deepcopy,copy
from ipywidgets import IntProgress
from itertools import chain
from IPython.display import display
from datetime import datetime
import pickle
import os
from sympy import sympify,latex,Float
import random
from math import ceil,sqrt
import seaborn as sbrn
from scipy.optimize import curve_fit
import pynumdiff
# Catch stout
from io import StringIO 
import sys
from contextlib import redirect_stdout

# Since the 'user' column do not have relevant information will not be read

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
path = f'{os.getcwd()}/rguimera-machine-scientist/machinescientist.py'
ms = SourceFileLoader('ms', path).load_module()
#
folder_name='res_bact_growth'+datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
#folder_name='test'
#folder_name='test'
dir=os.path.join(os.getcwd(),folder_name)
if not os.path.exists(dir):
    os.mkdir(dir)
# Read data

#deriv=pd.read_csv('microbial_growth_derivatives.csv')
data=pd.read_csv('microbial_growth_full.csv')


data['Time(min)'] = data['Time(min)'].shift(1)
data=data.drop([0])

#display(derivates)
display(data)




# ## BMS C: One model, multiDataFrame

# In[ ]:



#"""

cols=['A3','A6','A7','A8','A9',
      'B4','B5','B6','B8','B9','B11','B12',
      'C1','C2','C3','C5','C7','C11','C12',
      'D2','D5','D6','D7','D8','D9','D12',
      'E3','E5','E12',
      'F2','F3','F7','F10',
      'G3','G6','G7','G8','G9',
      'H1','H2','H3','H4','H5','H6','H8','H10','H12']
"""      
cols=['A3','A6','A9',
      'B5','B6','B9','B11','B12',
      'C2','C5','C7','C11',
      'D5','D6','D12',
      'E3','E5','E12',
      'F3','F7','F10',
      'G7','G8','G9',
      'H12']

cols=['A3','A9',
      'B5',
      'C2',
      'E3',
      'G7','G8','G9',
      'H12']
"""
print(len(cols))

sqr_size=int(ceil(sqrt(len(cols))))
print(sqr_size)
n_cols=7
n_rows=7
# Plot data

x={}
y={}
for col in cols:
	#Savitzky-Golay
	#par = [6,9,9]
	#x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(x[col].B.to_numpy(),10, par)
	y[col]=deepcopy(data[col])
	x[col]=deepcopy(pd.DataFrame(data={'t':data['Time(min)']}))
	#print(type(y[col]),y[col])
	#print(x[col])


#print(x)
#print(y)

mcmc_resets = 3
mcmc_steps = 3000
XLABS = ['t']
params = 4

"""
# Logistic:
logistic_growth= '(_a0_ / (_a1_ + (_a2_ * (exp((_a3_ * t))))))'
logistic_growth= '(_a0_ + ((_a1_ - _a0_) / (_a2_ + (exp((((_a3_ * (_a4_ - t)) + _a5_))))))'

logistic_model=ms.from_string_model(x,y,logistic_growth,1,6,XLABS,silence=False)
#logistic_predict=logistic_model.predict(x)
q
# Gompertz:
gompertz_growth= '(_a0_ * exp((_a1_ * (exp((_a3_ * t)) - _a2_))))'

gompertz_model=ms.from_string_model(x,y,gompertz_growth,1,4,XLABS,silence=False)
#gompertz_predict=gompertz_model.predict(x)

print(logistic_model.par_values)
print(gompertz_model.par_values)
"""

with open(f'./{folder_name}/stdout.txt', 'a') as f:
    with redirect_stdout(f):
        best_model, mdl, fig_dl = ms.machinescientist(x,
                                      y,
                                      XLABS=XLABS,n_params=params,
                                      resets=mcmc_resets,
                                      steps_prod=mcmc_steps,
                                      log_scale_prediction=False
                                      )
fig_dl.savefig(f'./{folder_name}/1description_length_B.pdf',format='pdf')
file1 = open(f"./{folder_name}/res.txt", "a")  # append mode
file1.write(f"Best model: {best_model}\n")
file1.write(f"DL: {mdl}\n")
file1.write(f"Latex: {best_model.latex()}\n")
file1.write(f"Parameters: {best_model.par_values}\n")
file1.write("###################### \n")
ms_predict=best_model.predict(x)
                                  
#fig, axs = plt.subplots(sqr_size,sqr_size)#,figsize=(25,17.5))#,sharey='col',sharex='col')
#fig.supxlabel('Real',va='baseline')

SMALL_SIZE = 26
MEDIUM_SIZE = 26
BIGGER_SIZE = 26
#sbrn.set(style='ticks', font_scale=2)
fig = plt.figure()
fig1=plt.figure()
fig2=plt.figure()
g = gs.GridSpec(n_rows,n_cols, wspace=0.1, hspace=0.1)
#g.update(wspace=2.0, hspace=0.3)

# B(t)
offset=0
for i in range(0,n_rows):
	for j in range(0,n_cols):
		#print(i,j)
		col=cols[offset]
		ax = fig.add_subplot(g[offset])
		ax1 = fig1.add_subplot(g[offset])
		ax2 = fig2.add_subplot(g[offset])
		#ax.title(col)
		ax2.plot(x[col].t.to_numpy(),y[col].to_numpy(),color='tab:blue',label='data')
		# Median smoothing Smooth Params
		par = [20,4]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.meandiff(data[col].to_numpy(), 10, par, options={'iterate': True})
		ax2.plot(x[col].t.to_numpy(),x_hat,label='Mean')
		# Gausian smoothing Smooth Params
		par = [20]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(data[col].to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(x[col].t.to_numpy(),x_hat,label='Gauss')
		# Friedrichs smoothing Smooth Params
		par = [20]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.friedrichsdiff(data[col].to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(x[col].t.to_numpy(),x_hat,label='Fried')
		# Butterworth smoothing Smooth Params
		par = [3,0.09]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(data[col].to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(x[col].t.to_numpy(),x_hat,label='Butter')
		#Savitzky-Golay
		par = [6,9,9]
		x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(data[col].to_numpy(), 10, par)
		ax2.plot(x[col].t.to_numpy(),x_hat,label='Savgol')
		# Data
		ax.plot(x[col].t.to_numpy(),y[col].to_numpy(),color='tab:blue',label='data')
		# Logistic
		# Scipy optimize
		def logistic(x, a0,a1,a2,a3,a4,a5):
		    #return a0/(1.+a1*np.exp(-a2*x))
		    return a0 + ((a1 - a0) / (a2 + np.exp((a3 * (a4 - x)) + a5)))
		
		lpopt, lpcov = curve_fit(logistic, x[col].t.to_numpy() , y[col].to_numpy())
		file1.write(f"Logistic {col} params: {lpopt}\n")
		if 'in' in lpopt:
			lpopt, lpcov = curve_fit(logistic, x[col].t.to_numpy() , y[col].to_numpy(),p0=[2.91709554e-01,1.48474918e+00,-7.60461841e+02])
		#print(lpopt,lpcov)

		ax.plot(x[col].to_numpy(),logistic(x[col].t.to_numpy(),*lpopt),color='tab:orange',label='Logistic')
		ax1.plot(y[col].to_numpy(),logistic(x[col].t.to_numpy(),*lpopt),color='tab:orange',label='Logistic')
		# Gompertz
		# Scipy optimize
		"""
		def gompertz(x, a0,a1,a2):
		    #return a0*np.exp(-a1*np.exp(-a2*x))
		    #return a0*np.exp(a1*(1.-np.exp(-a2*x)))
		    return a0+a1*np.exp(-np.exp(a1*(a2-x)))
		
		gpopt, gpcov = curve_fit(gompertz, x[col].t.to_numpy() , y[col].to_numpy())
		file1.write(f"Gompertz {col} params: {gpopt}\n")
		if 'in' in gpopt:
			gpopt, gpcov = curve_fit(gompertz, x[col].t.to_numpy() , y[col].to_numpy(),p0=[0.1,2.67791891e-01,1.39481299e-04])
		#print(gpopt,gpcov)
		ax.plot(x[col].to_numpy(),gompertz(x[col].t.to_numpy(),*gpopt),color='tab:green',label='Gompertz')
		ax1.plot(y[col].to_numpy(),gompertz(x[col].t.to_numpy(),*gpopt),color='tab:green',label='Gompertz')
		"""
		# MS
		ax.plot(x[col].t.to_numpy(),ms_predict[col].to_numpy(),color='tab:red',label='BMS')
		ax1.plot(y[col].to_numpy(),ms_predict[col].to_numpy(),color='tab:red',label='BMS')
		#axs[i,j].set_yscale('log')
		#axs[i,j].set(adjustable='box', aspect='equal')
		if offset==0:
			ax.legend(loc='best',fontsize='x-small')
			ax1.legend(loc='best',fontsize='x-small')
			ax2.legend(loc='best',fontsize='x-small')
		if j!=0:
			ax.set_yticks([])
			ax1.set_yticks([])
			ax2.set_yticks([])
		if i!=n_cols-1:
			ax.set_xticks([])
			ax1.set_xticks([])
			ax2.set_xticks([])
		ax.text(0.8, 0.1, col, transform=ax.transAxes)
		ax1.text(0.8, 0.1, col, transform=ax1.transAxes)
		ax2.text(0.8, 0.1, col, transform=ax2.transAxes)
		offset+=1
		if offset==len(cols): break
fig.supxlabel('Time(min)')
fig.supylabel('B')
fig1.supxlabel('Real')
fig1.supylabel('Predicted')
fig2.supxlabel('Time(min)')
fig2.supylabel('B')

fig.tight_layout()
fig.savefig(f'./{folder_name}/2B-t.pdf',format='pdf')
fig1.tight_layout()
fig1.savefig(f'./{folder_name}/3B_real_pred.pdf',format='pdf')
fig2.tight_layout()
fig2.savefig(f'./{folder_name}/B-t_pynumdiff.pdf',format='pdf')


# Diferential Equation

x={}
y={}
for col in cols:
	x[col]=deepcopy(data).rename(columns={col:'B','Time(min)':'t'})
	B=x[col].B.values
	# 3point
	y[col]=pd.Series([(B[1]-B[0])/10.]+[(B[i+1]-B[i-1])/20. for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/10.])
	# 5point
	"""
	y[col]=pd.Series([(B[1]-B[0])/10.]+
			  [(B[2]-B[0])/20.]+
			  [(B[i-2]-8*B[i-1]+8*B[i+1]-B[i+2])/(12*10) for i in range(2,len(B)-2)]+
			  [(B[len(B)-1]-B[len(B)-3])/20.]+
			  [(B[len(B)-1]-B[len(B)-2])/10.])"""
	#print('Len deriv:',len(y[col]))
	#par = [20]
	#x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(x[col].B.to_numpy(), 10, par, options={'iterate': False})
	#Savitzky-Golay
	par = [2,21,21]
	x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(x[col].B.to_numpy(), 10, par)
	y[col]=pd.Series(dxdt_hat)
	#print(x[col],y[col])

mcmc_resets = 5
mcmc_steps = 7000
XLABS = ['B']
params = 4

print()
with open(f'./{folder_name}/stdout.txt', 'a') as f:
    with redirect_stdout(f):
        best_model, mdl, fig_dl = ms.machinescientist(x,
                                      y,
                                      XLABS=XLABS,n_params=params,
                                      resets=mcmc_resets,
                                      steps_prod=mcmc_steps,
                                      log_scale_prediction=False
                                      )
fig_dl.savefig(f'./{folder_name}/4description_length_dBdt.pdf',format='pdf')
file1.write("###################### \n")
file1.write(f"Best model: {best_model}\n")
file1.write(f"DL: {mdl}\n")
file1.write(f"Latex: {best_model.latex()}\n")
file1.write(f"Parameters: {best_model.par_values}\n")
file1.write("###################### \n")
ms_predict=best_model.predict(x)

with open(f'./{folder_name}/stdout.txt', 'a') as f:
    with redirect_stdout(f):
        # Logistic:
        logistic_growth= '(_a0_ * (t * (_a1_ - (t * _a2_))))'

        logistic_model=ms.from_string_model(x,y,logistic_growth,1,3,XLABS,silence=True)
        #logistic_predict=logistic_model.predict(x)

        # Gompertz:
        gompertz_growth= '(_a0_ * (t * log((_a2_ / t))))'

        gompertz_model=ms.from_string_model(x,y,gompertz_growth,1,2,XLABS,silence=True)
        #gompertz_predict=gompertz_model.predict(x)

        print('Logistic',logistic_model.E,logistic_model.par_values)
        print('Gompertz',gompertz_model.E, gompertz_model.par_values)


def euler_BMS(model,y0,h,steps,col):
	res=[]
	res.append(y0)
	for i in range(steps-1):
		f=model.predict({col:pd.DataFrame(data={'B':[res[-1]]})})[col].to_numpy()[0]
		res.append(res[-1]+h*f)
	return res
def euler_curvefit(model,y0,h,steps,params):
	res=[]
	res.append(y0)
	for i in range(steps-1):
		res.append(res[-1]+h*model(res[-1],*params))
	return res
def resize(t1,t2,B):
	res=[]
	for i,j in zip(t2,B):
		if int(i) in t1:
			res.append(j)
	return res

fig = plt.figure()
fig1=plt.figure()
fig2=plt.figure()
fig3=plt.figure()
fig4=plt.figure()
g = gs.GridSpec(n_rows,n_cols, wspace=0.1, hspace=0.1)
#g.update(wspace=2.0, hspace=0.3)

# B(t)
offset=0
for i in range(0,n_rows):
	for j in range(0,n_cols):
		#print(i,j)
		col=cols[offset]
		ax = fig.add_subplot(g[offset])
		ax1 = fig1.add_subplot(g[offset])
		ax2 = fig2.add_subplot(g[offset])
		ax3 = fig3.add_subplot(g[offset])
		ax4 = fig4.add_subplot(g[offset])
		B=x[col].B.values
		h=1
		times=np.arange(0,max(x[col].t.values)+h,h)
		# 3point
		dev3=pd.Series([(B[1]-B[0])/10.]+[(B[i+1]-B[i-1])/20. for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/10.])
		ax2.plot(dev3,label='3point')
		# First Order Smooth Params
		x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(x[col].B.to_numpy(), 10)
		ax2.plot(dxdt_hat,label='Finite1')
		# First Order Smooth Params
		par = [50]
		x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(x[col].B.to_numpy(), 10, par, options={'iterate': True})
		ax2.plot(dxdt_hat,label='Finite2')
		# Median smoothing Smooth Params
		par = [20,4]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.meandiff(x[col].B.to_numpy(), 10, par, options={'iterate': True})
		ax2.plot(dxdt_hat,label='Mean')
		# Gausian smoothing Smooth Params
		par = [20]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(x[col].B.to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(dxdt_hat,label='Gauss')
		# Friedrichs smoothing Smooth Params
		par = [20]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.friedrichsdiff(x[col].B.to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(dxdt_hat,label='Fried')
		# Butterworth smoothing Smooth Params
		par = [3,0.09]
		x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(x[col].B.to_numpy(), 10, par, options={'iterate': False})
		ax2.plot(dxdt_hat,label='Butter')
		#Savitzky-Golay
		par = [6,9,9]
		x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(x[col].B.to_numpy(), 10, par)
		ax2.plot(dxdt_hat,label='Savgol')
		
		# Data
		ax.plot(x[col].B.to_numpy(),y[col].to_numpy(),color='tab:blue',label='data')
		# Logistic
		# Scipy optimize
		def logistic(x, a0,a1,a2):
		    return a0*x*(a1-x*a2)
		
		lpopt, lpcov = curve_fit(logistic, x[col].B.to_numpy() , y[col].to_numpy())
		file1.write(f"Logistic-diff {col} params: {lpopt}\n")
		#if 'in' in lpopt:
		#	lpopt, lpcov = curve_fit(logistic, x[col].B.to_numpy() , y[col].to_numpy(),p0=[2.91709554e-01,1.48474918e+00,-7.60461841e+02])
		#print(lpopt,lpcov)

		ax.plot(x[col].B.to_numpy(),logistic(x[col].B.to_numpy(),*lpopt),color='tab:orange',label='Logistic')
		ax1.plot(y[col].to_numpy(),logistic(x[col].B.to_numpy(),*lpopt),color='tab:orange',label='Logistic')
		log_num=euler_curvefit(logistic,B[0],h,len(times),lpopt)
		ax3.plot(times,log_num,color='tab:orange',label='Logistic')
		ax4.plot(data[col].to_numpy(),resize(data['Time(min)'].to_numpy(),times,log_num),color='tab:orange',label='Logistic')
		# Gompertz
		# Scipy optimize
		def gompertz(x, a0,a1):
		    #return -a0*np.log(x/a1)*x
		    return a0*np.log(a1/x)*x
		
		gpopt, gpcov = curve_fit(gompertz, x[col].B.to_numpy() , y[col].to_numpy())
		file1.write(f"Gompertz-diff {col} params: {gpopt}\n")
		#if 'in' in gpopt:
		#	gpopt, gpcov = curve_fit(gompertz, x[col].B.to_numpy() , y[col].to_numpy(),p0=[0.1,2.67791891e-01,1.39481299e-04])
		#print(gpopt,gpcov)
		ax.plot(x[col].B.to_numpy(),gompertz(x[col].B.to_numpy(),*gpopt),color='tab:green',label='Gompertz')
		ax1.plot(y[col].to_numpy(),gompertz(x[col].B.to_numpy(),*gpopt),color='tab:green',label='Gompertz')
		gom_num=euler_curvefit(gompertz,B[0],h,len(times),gpopt)
		ax3.plot(times,gom_num,color='tab:green',label='Gompertz')
		ax4.plot(data[col].to_numpy(),resize(data['Time(min)'].to_numpy(),times,gom_num),color='tab:green',label='Gompertz')
		# MS
		ax.plot(x[col].B.to_numpy(),ms_predict[col].to_numpy(),color='tab:red',label='BMS')
		ax1.plot(y[col].to_numpy(),ms_predict[col].to_numpy(),color='tab:red',label='BMS')
		bms_num=euler_BMS(best_model,B[0],h,len(times),col)
		ax3.plot(times,bms_num,color='tab:red',label='BMS')
		ax4.plot(data[col].to_numpy(),resize(data['Time(min)'].to_numpy(),times,bms_num),color='tab:red',label='BMS')
		
		# Real data for numerical integration
		ax3.plot(data['Time(min)'].to_numpy(),data[col].to_numpy(),color='tab:blue',label='Real')
		#axs[i,j].set_yscale('log')
		#axs[i,j].set(adjustable='box', aspect='equal')
		if offset==0:
			ax.legend(loc='best',fontsize='x-small')
			ax1.legend(loc='best',fontsize='x-small')
			ax2.legend(loc='center right',fontsize='x-small')
			ax3.legend(loc='best',fontsize='x-small')
			ax4.legend(loc='best',fontsize='x-small')
		if j!=0:
			ax.set_yticks([])
			ax1.set_yticks([])
			ax2.set_yticks([])
			ax3.set_yticks([])
			ax4.set_yticks([])
		if i!=n_cols-1:
			ax.set_xticks([])
			ax1.set_xticks([])
			ax2.set_xticks([])
			ax3.set_xticks([])
			ax4.set_xticks([])
		ax.text(0.5, 0.1, col, transform=ax.transAxes)
		ax1.text(0.1, 0.8, col, transform=ax1.transAxes)
		ax2.text(0.8, 0.8, col, transform=ax2.transAxes)
		ax3.text(0.8, 0.1, col, transform=ax3.transAxes)
		ax4.text(0.8, 0.1, col, transform=ax4.transAxes)
		offset+=1
		if offset==len(cols): break
file1.close()

fig.supxlabel('B')
fig.supylabel('$dB/dt$')
fig.tight_layout()
fig.savefig(f'./{folder_name}/5dbBdt-B.pdf',format='pdf')

fig1.supxlabel('Real')
fig1.supylabel('Predicted')
fig1.tight_layout()
fig1.savefig(f'./{folder_name}/6dBdT_real_pred.pdf',format='pdf')

fig2.supxlabel('Time(min)')
fig2.supylabel('$dB/dt$')
fig2.tight_layout()
fig2.savefig(f'./{folder_name}/7dBdT_time.pdf',format='pdf')

fig3.supxlabel('Time(min)')
fig3.supylabel('$B(t)$')
fig3.tight_layout()
fig3.savefig(f'./{folder_name}/8numerical_Euler.pdf',format='pdf')

fig4.supxlabel('B (Real)')
fig4.supylabel('$B (Euler)$')
fig4.tight_layout()
fig4.savefig(f'./{folder_name}/9numerical_Euler_real_pred.pdf',format='pdf')


