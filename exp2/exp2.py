# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 23:11:39 2022

@author: dorsh
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

from ValueError import *

# part 1
def prop(a,x):
    return a*x

def liner_curve(a,b,x):
    return a*x+b

def r_squre(data,fit):
    RSS  = np.sum((data-fit)**2)
    TSS = np.sum(data**2)
    return 1 - RSS/TSS

theta_min_value = np.array([-48.42,-31.86,-15.75,0,15.84,31.86,48.64]) * np.pi/180
theta_min_err = np.array([0.1,0.75,0.05,0.05,0.05,0.75,0.1])  * np.pi/180
theta_min = Valerr(theta_min_value,theta_min_err)

f = lambda x: np.sin(x/2)
sin_alpha = Valerr.general_funcion(f,theta_min) #notice degree or radian!!



n = np.array(range(-3,4))
lambda_Hg = 546.074e-9 #m

fit1 = cfit(prop,sin_alpha.val,n)
fig1 = plt.figure("fig1",dpi=200)
plt.errorbar(sin_alpha.val,n,xerr=sin_alpha.err,fmt=".",label="Data")
plt.plot(sin_alpha.val,prop(fit1[0],sin_alpha.val),"-.",label="Regression")
plt.grid()
plt.legend()
plt.ylabel("n - order",fontsize=10)
plt.xlabel(r"$sin(\frac{\theta_{min}}{2})$",fontsize=10)
plt.show()

m_val = fit1[0][0]
m_err =  np.sqrt(fit1[1][0][0]) * 2
m = Valerr(m_val,m_err)
d = m * lambda_Hg / 2

print("d=",d) # need to calculte error

R_squre = r_squre(n,prop(fit1[0],sin_alpha.val))
print("R^2 =",R_squre)


# part 2
theta_min_value = np.array([20.5,19.35,17.01,14.6,14.49,14.2,13.63,12.96,12.69,11.25]) * np.pi/180
#theta_min_value = np.append(theta_min_value[:2],theta_min_value[7:-1])
theta_min_err = np.array([0.1,0.05,0.05,0.1,0.05,0.1,0.05,0.05,0.1,0.1]) * np.pi/180
#theta_min_err = np.append(theta_min_err[:2],theta_min_err[7:-1])
theta_min = Valerr(theta_min_value,theta_min_err)

n = -1
f = lambda d,x: (2 * d * np.sin(x/2) / n) 
d = Valerr(d.val*np.ones(len(theta_min.val)),d.err*np.ones(len(theta_min.val)))
wave_len = Valerr.general_funcion(f,d,theta_min)
print("wave_len:",wave_len)


fig, ax = plt.subplots(dpi=300)

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

# wave_for_table = -wave_len.val * 1e11
# wave_for_table = np.round(wave_for_table) /1e2
f = lambda w: np.round(-w*1e11) /1e2
wave_for_table = Valerr.general_funcion(f, wave_len)

col1 = list(range(1,10)) + [13]
col1 = [str(x) for x in col1]
col2 = [str(r"$%d\pm%d$"%(wave_for_table.val[i],wave_for_table.err[i])) for i in range(len(wave_for_table.val))]
col3 = np.round(np.array([706.52,667.82,587.56,504.77,501.57,492.19,471.31,447.15,438.79,396.47]))
col3 = [str(int(x)) for x in col3]
data = {"n" :col1, r"$\lambda_{Messured}$ $[nm]$":col2,r"$\lambda_{Theory}$ $[nm]$":col3}
df = pd.DataFrame(data)


ax.table(cellText=df.values, colLabels=df.columns,loc='center')

fig.tight_layout()

plt.show()




# part 3
wave_len = Valerr(wave_len.val[:-3],wave_len.err[:-3])
n = -1
delta_min_value = -np.array([47.92,48.55,49.68,49.86,50.26,50.85,51.07,65.25,66.5][:-2])  * np.pi/180
#delta_min_value = np.append(delta_min_value[:2],delta_min_value[7:])
delta_min_err = np.array([0.05,0.05,0.05,0.05,0.1,0.05,0.1,0.1,0.2][:-2]) * np.pi/180
#delta_min_err = np.append(delta_min_err[:2],delta_min_err[7:]) 
delta_min = Valerr(delta_min_value,delta_min_err)

x = wave_len**(-2)

fit2 = linregress(x.val,delta_min.val)
fig2 = plt.figure("fig2",dpi=200)
plt.errorbar(x.val,delta_min.val,yerr=delta_min.err,xerr=x.err,fmt=".",label="Data")
plt.plot(x.val,liner_curve(fit2.slope,fit2.intercept,x.val),"-.",label="Regression")
plt.grid()
plt.legend()
plt.xlabel(r"$\frac{1}{\lambda^2}[m^{-2}]$",fontsize=10)
plt.ylabel(r"$\delta_{min}[Rad]$",fontsize=10)
plt.show()

B = fit2.intercept
C = fit2.slope

# part 4
c = scipy.constants.c # in m/s
h = scipy.constants.physical_constants["Planck constant in eV/Hz"][0] # in eV/Hz

delta_min_value = -np.array([48.06,49.95,51.21,51.525])* np.pi/180
delta_min_err = np.array([0.1,0.05,0.1,0.1])* np.pi/180
delta_min = Valerr(delta_min_value,delta_min_err)

f = lambda delta: (C/(delta-B))**0.5
wave_len = Valerr.general_funcion(f,delta_min)
# wave_len = Valerr(np.array([656.279,486.135,434.0472,410.1734])*1e-9)  #True values
# f = lambda w: B+C/w**2
# delta_min = Valerr.general_funcion(f,wave_len)

f = Valerr(c) / wave_len
E = f*h

n = np.array(range(3,(len(E.val)+3)))
R = E / (1/2**2 - 1/n**2)
R_mean = np.mean(R.val)
R_err = Valerr._dist(*R.err) /4
print("R=",R_mean,"eV")

R_theory = scipy.constants.physical_constants["Rydberg constant times hc in eV"][0] # in eV
R_rel_err = abs(R_mean-R_theory) / R_theory *100
print("in presicion of:",R_rel_err,"%")

wave_len = wave_len *1e9
wave_len_theory = [656.279,486.135,434.0472,410.1734]
E_theory = [1.89,2.55,2.86,3.03]
data = { "Transition of n" : ["Wavelength [nm]          ","Energy difference [eV]  ","Rydberg constant[eV]       ", "WAVELENGTH THEORY [nm]","ENERGY DIFFERENCE [eV]"],
        r"$3\rightarrow2$": [str(r"$%d\pm%d$"%(wave_len.val[0],wave_len.err[0])),str(r"$%.2f\pm%.2f$"%(E.val[0],E.err[0])),str(r"$%.1f\pm%.1f$"%(R.val[0],0.3)),wave_len_theory[0],E_theory[0]],
        r"$4\rightarrow2$": [str(r"$%d\pm%d$"%(wave_len.val[1],wave_len.err[1])),str(r"$%.2f\pm%.2f$"%(E.val[1],E.err[1])),str(r"$%.2f\pm%.2f$"%(R.val[1],R.err[1])),wave_len_theory[1],E_theory[1]],
        r"$5\rightarrow2$": [str(r"$%d\pm%d$"%(wave_len.val[2],wave_len.err[2])),str(r"$%.2f\pm%.2f$"%(E.val[2],E.err[2])),str(r"$%.1f\pm%.1f$"%(R.val[2],R.err[2])),wave_len_theory[2],E_theory[2]],
        r"$6\rightarrow2$": [str(r"$%d\pm%d$"%(wave_len.val[3],wave_len.err[3])),str(r"$%.2f\pm%.2f$"%(E.val[3],E.err[3])),str(r"$%.1f\pm%.1f$"%(R.val[3],R.err[3])),wave_len_theory[3],E_theory[3]] 
        }
        
fig, ax = plt.subplots(dpi=400)

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
df = pd.DataFrame(data)


ax.table(cellText=df.values, colLabels=df.columns,loc='center',fontsize=30,colWidths=[0.5]*len(df.columns))

fig.tight_layout()

plt.show()
