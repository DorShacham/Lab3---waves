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
'''
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

'''


# part 1 - lattice as a polarizor
# wo lattice
print("----Part 1 - without latice----")
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

print("Printing the regression fits for measurments:")
(fig11,fit11)=one4all(cos_theta_squared[:13],intensity[:13],intensity_err[:13],cos_theta_squared_err[:13],"linear",None,r"$cos(\theta)^2$","intensity [volt]")
print("The fit for measurements 0-90 degrees is:")
print(fit11)

(fig12,fit12)=one4all(cos_theta_squared[12:19],intensity[12:19],intensity_err[12:19],cos_theta_squared_err[12:19],"linear",None,r"$cos(\theta)^2$","intensity [volt]")
print("The fit for measurements 90-180 degrees is:")
print(fit12)

(fig13,fit13)=one4all(cos_theta_squared[18:25],intensity[18:25],intensity_err[18:25],cos_theta_squared_err[18:25],"linear",None,r"$cos(\theta)^2$","intensity [volt]")
print("The fit for measurements 180-270 degrees is:")
print(fit13)

(fig14,fit14)=one4all(cos_theta_squared[24:],intensity[24:],intensity_err[24:],cos_theta_squared_err[24:],"linear",None,r"$cos(\theta)^2$","intensity [volt]")
print("The fit for measurements 270-360 degrees is:")
print(fit14)

(fig12,fit12)=one4all(theta,intensity,intensity_err,theta_err,"none",None,r"$\theta[rad]$","intensity [volt]")
print("Conclusion : there is no linear dependancy between cos^2(theta) and the Intensity. seems more like cos(theta) linear dependancy.")
print("-----------------")

# with lattice
print("----Part 2 - with Latice----")
theta = [0,10,20,30,40,50,60,70,80,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dgree
theta_err= [5] * len(theta) 
intensity = [0.007,0.077,0.082,0.105,0.156,0.253,0.337,0.424,0.495,0.522,0.497,0.405,0.277,0.15,0.066,0.05,0.058,0.085,0.147,0.317,0.43,0.44,0.464,0.361,0.27,0.099,0.052,0.06]
intensity_err= [0.05/10] * len(theta)

theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
cos_theta_pow_4= np.array(np.cos(theta)**4)
cos_theta_squared_err=theta_err*2*np.sin(2*theta)*np.cos(theta)**2


print("Printing the regression fits for measurments:")
(fig21,fit21)=one4all(cos_theta_pow_4[:10],intensity[:10],intensity_err[:10],cos_theta_squared_err[:10],"linear",None,r"$cos(\theta)^4$","intensity [volt]")
print("The fit for measurements 0-90 degrees is:")
print(fit21)

(fig22,fit22)=one4all(cos_theta_pow_4[9:16],intensity[9:16],intensity_err[9:16],cos_theta_squared_err[9:16],"linear",None,r"$cos(\theta)^4$","intensity [volt]")
print("The fit for measurements 90-180 degrees is:")
print(fit22)

(fig23,fit23)=one4all(cos_theta_pow_4[15:22],intensity[15:22],intensity_err[15:22],cos_theta_squared_err[15:22],"linear",None,r"$cos(\theta)^4$","intensity [volt]")
print("The fit for measurements 180-270 degrees is:")
print(fit23)

(fig24,fit24)=one4all(cos_theta_pow_4[21:],intensity[21:],intensity_err[21:],cos_theta_squared_err[21:],"linear",None,r"$cos(\theta)^4$","intensity [volt]")
print("The fit for measurements 270-360 degrees is:")
print(fit24)

(fig25,fit25)=one4all(theta,intensity,intensity_err,theta_err,"none",None,r"$\theta[rad]$","intensity [volt]")

print("Conclusion : there is no linear dependancy between cos^4(theta) and the Intensity. seems more like cos^2(theta) linear dependancy.")
print("-----------------")

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
print ("---------------")
#%%
print("----- Part 2 - Waveguide ----- ")
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
print("Lambda_g from distance between nodes measurements is: ",lambda_g_from_distance_between_nodes)


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
print("Lambda_g from distance between extermums is: ",lambda_g_from_distance_between_max)

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
print("lambda found from measurment in multiple d's: ",lambda_found)

#%%
# #16 

L = 15 #cm
L_err = 0.2 #cm
wavelen = 2.8 #cm
wavelen_err = 0
d_of_phi = lambda phi : 1/(2*np.sqrt(phi/(np.pi*L*wavelen)-(phi/(2*np.pi*L))**2))

d_linear = np.array([d_of_phi(2*np.pi),d_of_phi(2*2*np.pi),d_of_phi(3*2*np.pi)])

#linear polarization 

# d1 = d_linear[2]
# theta1 = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300]
# theta_err1 = 5
# V1 = [0.297,0.207,0.158,0.214,0.275,0.312,0.328,0.259,0.244,0.192,0.255,0.303,0.348,0.226,0.153,0.218,0.269,0.292,0.133,0.123,0.272]
# V_err1 = 0.001


# fig1=plt.figure()
# ax1=plt.axes(polar=True)
# theta1 = np.array(theta1)*np.pi/180
# theta_err1 = np.array(theta_err1)*np.pi/180
# V1 = np.array(V1)
# V_err1 = np.array(V_err1)

# ax1.errorbar(theta1,V1,V_err1,theta_err1,"-o")
# plt.grid(True)
# plt.show()

#########
d2  = 3.32 
theta2 = [0,30,60,90,120,150,180,210,240,270,300,330,360,45,165,225,345]
theta_err2 = 5
V2 = [0.503,0.53,0.408,0.335,0.225,0.128,0.484,0.540,0.378,0.333,0.23,0.1,0.503,0.459,0.29,0.447,0.3]
V_err2 = 0.005


fig2=plt.figure()
ax2=plt.axes(polar=True)
theta2 = np.array(theta2)*np.pi/180
theta_err2 = np.array(theta_err2)*np.pi/180
V2 = np.array(V2)
V_err2 = np.array(V_err2)

ax2.errorbar(theta2,V2,V_err2,theta_err2,"o")
plt.grid(True)
#plt.show()

#%%
#cyclic polarization
L = 15
L_err = 0.8
wavelen = 2.8
wavelen_err = 0
d_of_phi = lambda phi : 1/(2*np.sqrt(phi/(np.pi*L*wavelen)-(phi/(2*np.pi*L))**2))
d_circler = np.array([d_of_phi(np.pi/2),d_of_phi(3*np.pi/2),d_of_phi(5*np.pi/2)])



d1 = 2.18 #cm
theta1 = [0,30,45,60,90,120,150,180,210,240,270,300,330,360]
theta_err1 = 5
V1 = [0.37,0.345,0.286,0.3,0.317,0.287,0.223,0.328,0.335,0.346,0.32,0.255,0.144,0.26]
V_err1 = 0.005


fig1=plt.figure()
ax1=plt.axes(polar=True)
theta1 = np.array(theta1)*np.pi/180
theta_err1 = np.array(theta_err1)*np.pi/180
V1 = np.array(V1)
V_err1 = np.array(V_err1)

ax1.errorbar(theta1,V1,V_err1,theta_err1,"o")
plt.grid(True)
plt.show()