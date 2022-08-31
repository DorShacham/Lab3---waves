import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit




# helper function for plotting data and regression
def one4all(xdata,ydata,yerr=0,xerr=0,mode="general function",f=None,xlabel="x",ylabel="y"):
   # print(xdata,ydata,yerr,xerr,mode,f,xlabel,ylabel)
    fig = plt.figure(dpi=300)
    plt.errorbar(xdata,ydata,yerr,xerr,"o",label="Data")

    
    if mode == "none":
        fit= []
        
        
    
    elif mode == "linear":
        fit = linregress(xdata,ydata)
        f = lambda x,a,b: a*x+b
        #fit_label = "Regression: y=" + str(fit.slope) + str("x+") + str(fit.intercept)
        plt.plot(xdata,f(xdata,fit.slope,fit.intercept),"-.",label="Regression")
    
       
        
    elif mode == "0 intercept":
        f = lambda x,a: a*x
        fit = cfit(f,xdata,ydata)
        plt.plot(xdata,f(xdata,*fit[0]),"-.",label="Regression")
        
    elif mode == "general function":
        if f==None:
            raise TypeError

        fit = cfit(f,xdata,ydata)
        plt.plot(xdata,f(xdata,*fit[0]),"-.",label="Regression")
    
    
    else:
        print("mode='",mode,"' is not definted!")
        raise TypeError


    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid()
    if mode != "none":
        plt.legend()

    plt.show()
    
    return (fig,fit)



#%% 3 michelson

print("Michelson")

x_node = [9.5,10.9,12.4,13.7,15.1,16.5,18,19.4,20.8,22.1,23.6,25,26.4,27.8] 
x_max = [9,10.3,11.7,13.2,14.5,16,17.4,18.8,20.2,21.6,23,24.4,25.7,27.1,28.5]
x_err = 0.1

V_node = [0.037,0.029,0.032,0.033,0.027,0.03,0.032,0.035,0.035,0.035,0.037,0.035,0.036,0.042]
V_max = [0.247,0.241,0.225,0.227,0.215,0.21,0.212,0.21,0.207,0.205,0.212,0.2105,0.21,0.207,0.208]
V_err = 0.002


x_node = np.array(x_node) 
x_max = np.array(x_max)
V_node = np.array(V_node)
V_max = np.array(V_max)

wavelen_node = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen node:',wavelen_node,"+-",wavelen_node_err,"cm")

wavelen_max = 2 * (np.abs(np.diff(x_max))).mean()
wavelen_max_err = 2* np.sqrt(2) * x_err / np.sqrt(len(x_max) - 1)
print('wavelen max:',wavelen_max,"+-",wavelen_max_err,"cm")


wavelen = np.array([wavelen_node,wavelen_max]).mean()
wavelen_err = np.sqrt(wavelen_node_err**2+wavelen_max_err**2) / 2
print('wavelen:',wavelen,"+-",wavelen_err,"cm")

x = np.append(x_node,x_max)
V = np.append(V_node,V_max)
fig = plt.figure()
f = lambda x,a,b,phi,c: a*np.sin(b*x+phi) + c
fig,fit = one4all(x,V,V_err,x_err,"none",xlabel="x [cm]",ylabel="V [V]")
#fig,fit = one4all(np.append(V_node[:-1],V_max[:-1]),np.append(np.diff(x_node),np.diff(x_max)),np.sqrt(2) * x_err / np.sqrt(len(x) - 1),V_err,"linear",xlabel="V [V]",ylabel=r"$\Delta x [cm]$")
n = np.array(range(0,len(x_node)))
fig,fit = one4all(n,x_node,0,x_err,"linear",xlabel="n",ylabel="x [cm]")
wavelen_node = 2*fit.slope
wavelen_node_err = fit.stderr*2
print('wavelen node:',2*fit.slope,"+-",fit.stderr*2,"cm")

n = np.array(range(0,len(x_max)))
fig,fit = one4all(n,x_max,0,x_err,"linear",xlabel="n",ylabel="x [cm]")
wavelen_max = 2*fit.slope
wavelen_max_err = fit.stderr*2
print('wavelen max:',2*fit.slope,"+-",fit.stderr*2,"cm")

wavelen = np.array([wavelen_node,wavelen_max]).mean()
wavelen_err = np.sqrt(wavelen_node_err**2+wavelen_max_err**2) / 2
print('wavelen:',wavelen,"+-",wavelen_err,"cm")

#%% 4 ferbri febro 

print("ferbri febro")
x1 = 62 #cm

d1 = x1-58.1


x_node = x1 - np.array([57.6,56.1,54.7,53.4,52,50.7,49.3,47.9,46.5,45.1])  #cm
x_err = 0.1 * np.sqrt(2) #cm

d2 = x_node[-1]
number_of_nodes = len(x_node)

x_node = np.array(x_node)
V_node = np.array(V_node)

wavelen_node = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen 1:',wavelen_node,"+-",wavelen_node_err,"cm")

n = np.array(range(0,len(x_node)))
fig,fit = one4all(n,x_node,0,x_err,"linear",xlabel="n",ylabel="x [cm]")
wavelen_node = 2*fit.slope
wavelen_node_err = fit.stderr*2
print('wavelen 1:',wavelen_node,"+-",wavelen_node_err,"cm")


#### part 2 
x1 = 62 #cm

d1 = x1-56.7


x_node = x1 - np.array([56.1,54.7,53.3,52,50.6,49.3,47.8,46.5,45,43.6])  #cm
x_err = 0.1 * np.sqrt(2) #cm

d2 = x_node[-1]
number_of_nodes = len(x_node)

x_node = np.array(x_node)
V_node = np.array(V_node)

wavelen_node2 = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err2 = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen 2:',wavelen_node2,"+-",wavelen_node_err2,"cm")

n = np.array(range(0,len(x_node)))
fig,fit = one4all(n,x_node,0,x_err,"linear",xlabel="n",ylabel="x [cm]")
wavelen_node2 = 2*fit.slope
wavelen_node_err2 = fit.stderr*2
print('wavelen 2:',wavelen_node2,"+-",wavelen_node_err2,"cm")

wavelen = np.array([wavelen_node,wavelen_node2]).mean()
wavelen_err = np.sqrt(wavelen_node_err**2+wavelen_node_err2**2) / 2
print('wavelen:',wavelen,"+-",wavelen_err,"cm")

#%% 5 Loyd
print("Loyed")


d1= 40-10 #cm
h_fix = 9
h = np.array([7.7,7.9,8.1,8.3,8.5,8.7,8.9,9.1,9.3,9.5,9.7,9.9,10.1,10.3,10.5,10.7,10.9,11.1,11.3]) + h_fix #cm
h_err = np.sqrt(0.1**2+0.5**2) #cm
v = np.array([0.336,0.335,0.333,0.337,0.341,0.344,0.345,0.344,0.345,0.341,0.338,0.340,0.332,0.339,0.335,0.331,0.335,0.335,0.342]) 
v_err = 0.002
h1 = h[2]
h2 = h[-4]
# notice V and I !!!!!
fig,fit = one4all(h,v,0,0,"none",xlabel="x [cm]",ylabel="V [V]")
wavelen = 2 * (np.sqrt(h2**2 + d1**2) - np.sqrt(h1**2 + d1**2))
wavelen_err = 2 * np.sqrt((h2*h_err/np.sqrt(h2**2+d1**2))**2+(h1*h_err/np.sqrt(h2**2+d1**2))**2+(d1*h_err/np.sqrt(h2**2+d1**2))**2)
print('wavelen:',wavelen,"+-",wavelen_err,"cm")


phi = (2*np.sqrt(h**2+d1**2)-2*d1) * 2 * np.pi / wavelen + np.pi
cos_phi = np.cos(phi)
#(fig,fit)=one4all(cos_phi,v,mode="linear",xlabel=r"$cos(\theta)$",ylabel="$V [V]$")



 ### part 2
print("Loyed")


d1= 40-10+1 #cm
h_fix = 9
h = np.array([6.6,6.8,7,7.2,7.4,7.6,7.8,8,8.25,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,10,10.2,10.4,10.6,10.8,11,11.2,11.4]) + h_fix #cm
h_err = np.sqrt(0.1**2+0.5**2) #cm
v = np.array([0.366,0.359,0.355,0.352,0.351,0.348,0.349,0.354,0.356,0.357,0.36,0.36,0.363,0.36,0.36,0.355,0.353,0.35,0.349,0.349,0.35,0.353,0.358,0.358,0.361]) 
v_err = 0.002
h1 = h[5]
h2 = h[-6]
# notice V and I !!!!!
fig,fit = one4all(h,v,0,0,"none",xlabel="x [cm]",ylabel="V [V]")
wavelen = 2 * (np.sqrt(h2**2 + d1**2) - np.sqrt(h1**2 + d1**2))
wavelen_err = 2 * np.sqrt((h2*h_err/np.sqrt(h2**2+d1**2))**2+(h1*h_err/np.sqrt(h2**2+d1**2))**2+(d1*h_err/np.sqrt(h2**2+d1**2))**2)
print('wavelen:',wavelen,"+-",wavelen_err,"cm")


phi = (2*np.sqrt(h**2+d1**2)-2*d1) * 2 * np.pi / wavelen + np.pi
cos_phi = np.cos(phi)
(fig,fit)=one4all(1+cos_phi,v,mode="linear",xlabel=r"$cos(\theta)$",ylabel="$V [V]$")
