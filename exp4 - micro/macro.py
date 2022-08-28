import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

# helper function for plotting data and regression
def one4all(xdata,ydata,yerr=0,xerr=0,mode="general function",f=None,xlabel="x",ylabel="y"):
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


    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()
    
    return (fig,fit)

#prep question 7
theta=np.array(range(0,9,1))*np.pi/4
#E1=E2
E2=1
E1=1
size_of_E = np.sqrt((E1*np.sin(theta))**2 +(E2*np.sin(theta+np.pi/2))**2)
dir_of_E= np.arccos(E1*np.sin(theta)/size_of_E)
fig0, ax0 = plt.subplots(subplot_kw={'projection': 'polar'})
ax0.plot(dir_of_E,size_of_E)
ax0.set_rmax(2)
ax0.set_rlabel_position(-22.5)
ax0.grid(True)
ax0.set_title("axial plot of E for multiple values of wt")
plt.show()

# part 1 - lattice as a polarizor
# wo lattice
L = 55 #cm 
zeroth_intensity = 0.014 #intensity when the power is off in V
zeroth_intensity_err=0.001
theta = [0,5,10,15,20,25,30,35,40,50,60,70,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dg ee  
theta_err= [5] * len(theta) 
intensity = np.array([5.15,5.10,5.10,5.02,4.9,4.75,4.62,4.44,4.17,3.90,3.3,1.9,0.43,1.53,3,3.97,4.65,5.02,5.14,5.02,4.55,3.72,2.63,1.24,0.55,1.9,3,4.01,4.66,4.95,5.15])/10 #V
intensity_err= [0.05/10] * len(theta)

theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
cos_theta_squared = np.cos(theta)**2
cos_theta_squared_err=theta_err*np.sin(2*theta)


(fig1,fit1)=one4all(cos_theta_squared[-5:],intensity[-5:],intensity_err[-5:],cos_theta_squared_err[-5:],"linear",None,r"$cos(\theta)^2$","intensity [W/m^2]")
print(fit1)
(fig12,fit12)=one4all(theta,intensity,intensity_err,theta_err,"none",None,r"$\theta$","intensity [W/m^2]")


# with lattice
theta = [0,10,20,30,40,50,60,70,80,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dgree
theta_err= [5] * len(theta) 
intensity = [0.007,0.077,0.082,0.105,0.156,0.253,0.337,0.424,0.495,0.522,0.497,0.405,0.277,0.15,0.066,0.05,0.058,0.085,0.147,0.317,0.43,0.44,0.464,0.361,0.27,0.099,0.052,0.06]
intensity_err= [0.05/10] * len(theta)

theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
cos_theta_pow_4= np.array(np.cos(theta)**4)
cos_theta_squared_err=theta_err*2*np.sin(2*theta)*np.cos(theta)**2

(fig2,fit2)=one4all(cos_theta_pow_4[9:16],intensity[9:16],intensity_err[9:16],cos_theta_pow_4[9:16],"linear",None,r"$cos(\theta)^4$","intensity [W/m^2]")
print(fit2)
(fig22,fit22)=one4all(theta,intensity,intensity_err,theta_err,"none",None,r"$\theta$","intensity [W/m^2]")

#part 2 waveguide properties

# #3.
d = np.array([1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])*1e-2 #meter
d_err = 1e-3 #meter
intensity = [0.319,0.319,0.321,0.346,0.407,0.45,0.507,0.548,0.608,0.678,0.714] # V
intensity_err = 0.005
fig3= plt.figure(dpi=300)
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")
plt.xlabel("d [m]", fontsize=12)
plt.ylabel("intensity [W/m^2]", fontsize=12)
plt.grid()
plt.legend()
plt.show()

#%%
# #4.
#polar graph to show the polarization in the exit of waveguide is linear
d = 4e-2 #meter
fig4=plt.figure()
ax4=plt.axes(polar=True)
theta=[0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,300,330,360]
theta_err = 5
intensity= [0.45,0.443,0.432,0.433,0.393,0.292,0.09,0.245,0.343,0.403,0.438,0.44,0.454,0.447,0.438,0.428,0.383,0.295,0.112,0.339,0.448,0.442]
intensity_err =0.005

theta = np.array(theta)*np.pi/180
theta_err = np.array(theta_err)*np.pi/180
intensity= np.array(intensity)
intensity_err= np.array(intensity_err)

ax4.plot(theta,intensity,"ro")
ax4.errorbar(theta,intensity,intensity_err,theta_err)
plt.grid(True)
plt.show()

#%%
# #6
d = np.array([1,1.3,1.4,1.5,1.7,2,2.5,3,3.5,4,4.5,5,5.5])*1e-2 #meter
d_err = 1e-3 #meter
intensity = [0.012,0.015,0.504,0.441,0.45,0.5,0.52,0.527,0.567,0.577,0.592,0.58,0.575,]# V
intensity_err = 0.005
fig3= plt.figure(dpi=300)
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")
plt.xlabel("d [m]", fontsize=12)
plt.ylabel("intensity [W/m^2]", fontsize=12)
plt.grid()
plt.legend()
plt.show()

#%%


#7 
# find the minimum width d_min for which the intensity drops drasticly at
d_min = 1.4e-2 #fill currect d_min
lambda_for_d = 2*d_min

#10
d = 2e-2 # meter
x_err = 2e-3
x = np.array([12,10.2,8.5,6.6,4.6]) *1e-2 #meter
distance_between_nodes = np.abs(np.diff(x))
distance_between_nodes_err=np.array([])
I = np.array([0.383,0.39,0.395,0.395,0.395])
I_err =0.005

#one4all(distance_between_nodes, I,I_err,distance_between_nodes_err,mode="none",None,"X","I")

distance_between_nodes= distance_between_nodes.mean()
lambda_g_from_distance_between_nodes=distance_between_nodes*2
print(lambda_g_from_distance_between_nodes)


d = 2e-2 # meter
x_err = 2e-3
x = np.array([14.8,13,11.1,9.3,7.4,5.6]) *1e-2 #meter
distance_between_max = np.abs(np.diff(x))
distance_between_nodes_max=np.array([])
I = np.array([0.528,0.522,0.512,0.503,0.504,0.51])
I_err =0.005

#one4all(distance_between_nodes, I,I_err,distance_between_nodes_err,mode="none",None,"X","I")

distance_between_max= distance_between_max.mean()
lambda_g_from_distance_between_max=distance_between_max*2
print(lambda_g_from_distance_between_max)

#%%


#13
d=np.array([1.6,2.3,2.5,2.7]) * 1e-2 #meter
d_err=1e-3 #meter
distance_between_nodes = np.array([
    np.abs(np.diff(np.array([13,10.3,7.5]))).mean(),
    np.abs(np.diff(np.array([13.5,12.6,10,8.3]))).mean(),
    np.abs(np.diff(np.array([12.5,11.7,9,7.5]))).mean(),
    np.abs(np.diff(np.array([12.3,10.7,9,7.4]))).mean()
   ]
    ) * 1e-2
distance_between_nodes_err = np.array([2e-3,3e-3,3e-3,3e-3])
lambda_g= distance_between_nodes*2
y = 1/lambda_g**2
y_err= 2*1/lambda_g**3
x=1/(2*d)**2
x_err=4/(2*d)**3

(fig13,fit13)= one4all(x,y,0,0,"linear",None,"1/lambda_g**2","1/(2*d)**2")
lambda_found = 1/np.sqrt(fit13.intercept)
print("lambda found=",lambda_found)
#check if slope makes sense - needs to be -1
#%%

# #16 ????????
# #cyclic polarization
# L = 15e-2
# L_err = 0
# wavelen = 2.8e-2
# wavelen_err = 0

# d= 1/ (2*np.sqrt(1/(2*L*wavelen)-1/(4*L)**2))

# fig16a=plt.figure()
# ax16a=plt.axes(polar=True)
# theta = np.array(theta)*np.pi/180
# theta_err = np.array(theta_err)*np.pi/180
# intensity= np.array(intensity)
# intensity_err= np.array(intensity_err)

# ax16a.plot(theta,intensity,"ro")
# ax16a.errorbar(theta,intensity,intensity_err,theta_err)
# plt.grid()
# plt.show()

# #linear polarization
# L = 15e-2
# L_err = 0
# wavelen = 2.8e-2
# wavelen_err = 0

# d= 1/ (2*np.sqrt(2/(L*wavelen)-1/L**2))


# fig16b=plt.figure()
# ax16b=plt.axes(polar=True)
# theta = np.array(theta)*np.pi/180
# theta_err = np.array(theta_err)*np.pi/180
# intensity= np.array(intensity)
# intensity_err= np.array(intensity_err)

# ax16b.plot(theta,intensity,"ro")
# ax16b.errorbar(theta,intensity,intensity_err,theta_err)
# plt.grid()
# plt.show()

#18 - make impormvements and repeat
