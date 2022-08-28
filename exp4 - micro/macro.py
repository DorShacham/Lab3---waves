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
theta = []
theta_err=[]
intensity = []
intensity_err=[]

theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
cos_theta_squared = np.array(np.cos(theta)**2)
cos_theta_squared_err=theta_err*np.sin(2*theta)

(fig1,fit1)=one4all(cos_theta_squared,intensity,intensity_err,cos_theta_squared_err,"linear",None,r"$cos(\theta)^2$","intensity [W/m^2]")
print(fit1)

# with lattice
theta = []
theta_err=[]
intensity = []
intensity_err=[]

theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
cos_theta_pow_4= np.array(np.cos(theta)**4)
cos_theta_squared_err=theta_err*2*np.sin(2*theta)**3

(fig2,fit2)=one4all(cos_theta_pow_4,intensity,intensity_err,cos_theta_pow_4,"linear",None,r"$cos(\theta)^4$","intensity [W/m^2]")
print(fit2)

#part 2 waveguide properties

#3.
d = []
d_err = []
intensity = []
intensity_err = []
fig3= plt.figure(dpi=300)
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")
plt.xlabel("d [m]", fontsize=12)
plt.ylabel("intensity [W/m^2]", fontsize=12)
plt.grid()
plt.legend()
plt.show()

#4.
#polar graph to show the polarization in the exit of waveguide is linear
fig4=plt.figure()
ax4=plt.axes(polar=True)
theta=[]
theta_err = []
intensity= []
intensity_err =[]

theta = np.array(theta)*np.pi/180
theta_err = np.array(theta_err)*np.pi/180
intensity= np.array(intensity)
intensity_err= np.array(intensity_err)

ax4.plot(theta,intensity,"ro")
ax4.errorbar(theta,intensity,intensity_err,theta_err)
plt.grid()
plt.show()

#6
d = []
d_err = []
intensity = []
intensity_err = []
fig6= plt.figure()
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")
plt.xlabel("d [m]", fontsize=12)
plt.ylabel("intensity [W/m^2]", fontsize=12)
plt.grid()
plt.legend()
plt.show()

#7 
# find the minimum width d_min for which the intensity drops drasticly at
d_min = 1 #fill currect d_min
lambda_for_d = 2*d_min

#10
distance_between_nodes = np.array([])
distance_between_nodes_err=np.array([])
I = np.array([])
I_err = np.array([])

one4all(distance_between_nodes, I,I_err,distance_between_nodes_err,mode="none",None,"X","I")

distance_between_nodes= distance_between_nodes.mean()
lambda_g_from_distance_between_nodes=distance_between_nodes*2


#12
distance_between_peaks = np.array([])
distance_between_peaks_err=np.array([])
distance_between_peaks = distance_between_peaks.mean()
lambda_g_from_distance_between_peaks= distance_between_peaks*2

#13
d=np.array([])
d_err=np.array([])
distance_between_nodes = np.array(
    np.array([]).mean(),
    np.array([]).mean(),
    np.array([]).mean(),
    np.array([]).mean(),
    np.array([]).mean()
    )
distance_between_nodes_err = np.array([])
lambda_g= distance_between_nodes/2
y = 1/lambda_g**2
y_err= 2*1/lambda_g**3
x=1/(2*d)**2
x_err=4/(2*d)**3

(fig13,fit13)= one4all(x,y,y_err,x_err,"linear",None,"1/lambda_g**2","1/(2*d)**2")
lambda_found = 1/np.sqrt(fit13.intercept)
#check if slope makes sense - needs to be -1

#16 ????????
#cyclic polarization
d= 1

fig16a=plt.figure()
ax16a=plt.axes(polar=True)
theta = np.array(theta)*np.pi/180
theta_err = np.array(theta_err)*np.pi/180
intensity= np.array(intensity)
intensity_err= np.array(intensity_err)

ax16a.plot(theta,intensity,"ro")
ax16a.errorbar(theta,intensity,intensity_err,theta_err)
plt.grid()
plt.show()

#linear polarization
d= 1
fig16b=plt.figure()
ax16b=plt.axes(polar=True)
theta = np.array(theta)*np.pi/180
theta_err = np.array(theta_err)*np.pi/180
intensity= np.array(intensity)
intensity_err= np.array(intensity_err)

ax16b.plot(theta,intensity,"ro")
ax16b.errorbar(theta,intensity,intensity_err,theta_err)
plt.grid()
plt.show()

#18 - make impormvements and repeat










