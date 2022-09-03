import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties.unumpy import nominal_values as uval
from uncertainties.unumpy import std_devs as uerr


# helper function for plotting data and regression
def one4all(xdata,ydata,yerr=0,xerr=0,mode="general function",f=None,xlabel="x",ylabel="y",show=True):
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
        
    if show:
        plt.show()
    
    return (fig,fit)

def Rsqrue(x,y):
    RSS = np.sum((y-x)**2)
    TSS = np.sum(y**2)
    return 1 - RSS/TSS

def Reg_print(fit):
    m = ufloat(fit.slope,fit.stderr*2)
    b = ufloat(fit.intercept,fit.intercept_stderr*2)
    print("==> y =(",m,")x + (",b,") , R^2=",fit.rvalue**2)
#%%
# #prep question 7
# theta=np.array(range(0,9,1))*np.pi/4
# #E1=E2
# E2=1
# E1=1
# size_of_E = np.sqrt((E1*np.sin(theta))**2 +(E2*np.sin(theta+np.pi/2))**2)
# dir_of_E= np.arccos(E1*np.sin(theta)/size_of_E)
# fig0, ax0 = plt.subplots(subplot_kw={'projection': 'polar'})
# ax0.plot(dir_of_E,size_of_E)
# ax0.set_rmax(2)
# ax0.set_rlabel_position(-22.5)
# ax0.grid(True)
# ax0.set_title("axial plot of E for multiple values of wt")
# plt.show()

# part 1 - lattice as a polarizor
# wo lattice
#%%
L = 55 #cm 
zeroth_intensity = 0.014 #intensity when the power is off in V
zeroth_intensity_err=0.001
theta = [0,5,10,15,20,25,30,35,40,50,60,70,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dg ee  
theta_err= [5] * len(theta) 
intensity = (np.array([5.15,5.10,5.10,5.02,4.9,4.75,4.62,4.44,4.17,3.90,3.3,1.9,0.43,1.53,3,3.97,4.65,5.02,5.14,5.02,4.55,3.72,2.63,1.24,0.55,1.9,3,4.01,4.66,4.95,5.15])/10) #V
intensity_err= [0.05/10] * len(theta)
intensity = unumpy.uarray(intensity,intensity_err)
theta = unumpy.uarray(theta,theta_err) 

theta_array = [theta[uval(theta)<=90],theta[(uval(theta)>=90) * (uval(theta) <=180)],theta[(uval(theta)>=180) * (uval(theta) <=270)],theta[(uval(theta)>=270) * (uval(theta) <=360)]]
intensity_array = [intensity[uval(theta)<=90],intensity[(uval(theta)>=90) * (uval(theta) <=180)],intensity[(uval(theta)>=180) * (uval(theta) <=270)],intensity[(uval(theta)>=270) * (uval(theta) <=360)]]



fig_array=[]
fit_array=[]
for (theta_i,intensity_i) in zip(theta_array,intensity_array):
    cos_theta_squared = (unumpy.cos(theta_i*np.pi/180))**2
    print("\n-----\n")
    (fig,fit)=one4all(uval(cos_theta_squared),uval(intensity_i),uerr(intensity_i),uerr(cos_theta_squared),"linear",None,r"$cos(\theta)^2$","$V [V]$")
    fig_array.append(fig)
    fit_array.append(fit)
    slope = ufloat(fit.slope,2*fit.stderr)
    intercept = ufloat(fit.intercept,2*fit.intercept_stderr)
    print(fit_array.index(fit)+1,"==> ","y=(",slope,")x","+(",intercept,")"," R^2=",fit.rvalue**2)

for fig in fig_array:
    fname =str( "fig/plot" + str(fig_array.index(fig) + 1))
    fig.savefig(fname)

f1 = lambda x,a: a*np.abs(np.cos(x))
f2 = lambda x,a: a*np.cos(x)**2
fit_f1 = cfit(f1,uval(theta*np.pi/180),uval(intensity))
fit_f2 = cfit(f2,uval(theta*np.pi/180),uval(intensity))
Rsqrue1 = Rsqrue(f1(uval(theta*np.pi/180),*fit_f1[0]),uval(intensity))
Rsqrue2 = Rsqrue(f2(uval(theta*np.pi/180),*fit_f2[0]),uval(intensity))

a = ufloat(fit_f1[0][0],2*np.sqrt(np.diag(fit_f1[1])[0]))
print("f1==> y = (",a,")|cos(x)|, R^2=",Rsqrue1)

a = ufloat(fit_f2[0][0],2*np.sqrt(np.diag(fit_f2[1])[0]))
print("f2==> y = (",a,")cos^2(x), R^2=",Rsqrue2)



#cos_theta_squared_err=theta_err*np.sin(2*theta)
fig4=plt.figure()
ax4=plt.axes(polar=True)
#ax4.plot(uval(theta*np.pi/180),uval(intensity),"ro")
ax4.errorbar(uval(theta*np.pi/180),uval(intensity),uerr(intensity),uerr(theta*np.pi/180),"o")
ax4.plot(uval(theta*np.pi/180),f1(uval(theta*np.pi/180),*fit_f1[0]),"-.",label=r"a$|\cos(\theta)|$")
ax4.plot(uval(theta*np.pi/180),f2(uval(theta*np.pi/180),*fit_f2[0]),"-.",color="black",label=r"$a\cos^2(\theta)$")

plt.grid(True)
ax4.legend()
plt.show()
fig4.savefig("fig/plot6")

(fig12,fit12)=one4all(uval(theta*np.pi/180),uval(intensity),uerr(intensity),uerr(theta*np.pi/180),"none",None,r"$\theta [rad]$","$V [V]$",show=False)
plt.plot(uval(theta*np.pi/180),f1(uval(theta*np.pi/180),*fit_f1[0]),"-.",label=r"$a|\cos(\theta)|$")
plt.plot(uval(theta*np.pi/180),f2(uval(theta*np.pi/180),*fit_f2[0]),"-d",color="black",label=r"$a\cos^2(\theta)$")
plt.legend()
plt.show()
fig12.savefig("fig/plot5")
#%%

# with lattice
theta = [0,10,20,30,40,50,60,70,80,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dgree
theta_err= [5] * len(theta) 
V = (np.array([0.07,0.077,0.082,0.105,0.156,0.253,0.337,0.424,0.495,0.522,0.497,0.405,0.277,0.15,0.066,0.05,0.058,0.085,0.147,0.317,0.43,0.44,0.464,0.361,0.27,0.099,0.052,0.06])-0.014)
V_err= [0.05/10] * len(theta)
intensity = unumpy.uarray(V,V_err)
theta = unumpy.uarray(theta,theta_err)

theta_array = [theta[uval(theta)<=90],theta[(uval(theta)>=90) * (uval(theta) <=180)],theta[(uval(theta)>=180) * (uval(theta) <=270)],theta[(uval(theta)>=270) * (uval(theta) <=360)]]
intensity_array = [intensity[uval(theta)<=90],intensity[(uval(theta)>=90) * (uval(theta) <=180)],intensity[(uval(theta)>=180) * (uval(theta) <=270)],intensity[(uval(theta)>=270) * (uval(theta) <=360)]]

fig_array=[]
fit_array=[]
for (theta_i,intensity_i) in zip(theta_array,intensity_array):
   cos_theta_power4 =(unumpy.cos((theta_i+90)*np.pi/180))**4
   print("\n-----\n")
   (fig,fit)=one4all(uval(cos_theta_power4),uval(intensity_i),uerr(intensity_i),uerr(cos_theta_power4),"linear",None,r"$cos(\theta+\frac{\pi}{2})^4$","$V [V]$")
   fig_array.append(fig)
   fit_array.append(fit)
   slope = ufloat(fit.slope,2*fit.stderr)
   intercept = ufloat(fit.intercept,2*fit.intercept_stderr)
   print(fit_array.index(fit)+1,"==> ","y=(",slope,")x","+(",intercept,")"," R^2=",fit.rvalue**2)
plt.show()

for fig in fig_array:
    fname =str( "fig/plot" + str(fig_array.index(fig) + 7))
    fig.savefig(fname,bbox_inches='tight')




fig4=plt.figure()
ax4=plt.axes(polar=True)
#ax4.plot(uval(theta*np.pi/180),uval(intensity),"ro")
ax4.errorbar(uval(theta*np.pi/180),uval(intensity),uerr(intensity),uerr(theta*np.pi/180),"o")
plt.grid(True)
fig4.savefig("fig/plot12")


(fig12,fit12)=one4all(uval(theta*np.pi/180),uval(intensity),uerr(intensity),uerr(theta*np.pi/180),"none",None,r"$\theta [rad]$","$V [V]$")
fig12.savefig("fig/plot11")

'''
theta = [0,10,20,30,40,50,60,70,80,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360] #dgree
theta_err= [5] * len(theta) 
intensity = [0.007,0.077,0.082,0.105,0.156,0.253,0.337,0.424,0.495,0.522,0.497,0.405,0.277,0.15,0.066,0.05,0.058,0.085,0.147,0.317,0.43,0.44,0.464,0.361,0.27,0.099,0.052,0.06]
intensity_err= [0.05/10] * len(theta)
intensity = unumpy.uarray(intensity,intensity_err)
theta= np.array(theta)*np.pi/180
theta_err= np.array(theta_err)*np.pi/180
theta = unumpy.uarray(theta,theta_err) 
cos_theta_pow_4= unumpy.cos(theta)**4
#cos_theta_squared_err=theta_err*2*np.size_of_En(2*theta)*np.cos(theta)**2
(fig2,fit2)=one4all(uval(cos_theta_pow_4),uval(intensity),uerr(intensity),uerr(cos_theta_pow_4),"linear",None,r"$cos(\theta)^4$","$V [V]$")
print(fit2)
(fig22,fit22)=one4all(uval(theta),uval(intensity),uerr(intensity),uerr(theta),"none",None,r"$\theta[rad]$","$V [V]$")
'''
#%%
#part 2 waveguide properties

# #3.
d = np.array([1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])*1e-2 #meter
d_err = 1e-3 #meter
V = np.array([0.319,0.319,0.321,0.346,0.407,0.45,0.507,0.548,0.608,0.678,0.714]) # V
intensity = V
intensity_err = 0.005
fig3= plt.figure(dpi=300)
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")


plt.xlabel("d [m]", fontsize=14)
plt.ylabel("$V [V]$", fontsize=14)
plt.grid()
plt.show()
fig3.savefig("fig/plot13")


# (fig,fit) = one4all(d[3:],intensity[3:],0,0,"linear")
# slope = ufloat(fit.slope,2*fit.stderr)
# intercept = ufloat(fit.intercept,2*fit.intercept_stderr)
# print("==> ","y=(",slope,")x","+(",intercept,")"," R^2=",fit.rvalue**2)


#%%
# #4.
#polar graph to show the polarization in the exit of waveguide is linear
d = 4e-2 #meter
fig4=plt.figure()
ax4=plt.axes(polar=True)
theta=[0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,300,330,360]
theta_err = 5
V= (np.array([0.45,0.443,0.432,0.433,0.393,0.292,0.09,0.245,0.343,0.403,0.438,0.44,0.454,0.447,0.438,0.428,0.383,0.295,0.112,0.339,0.448,0.442])-0.09)
intensity = V
intensity_err =0.005

theta = np.array(theta)*np.pi/180
theta_err = np.array(theta_err)*np.pi/180
intensity= np.array(intensity)
intensity_err= np.array(intensity_err)

ax4.errorbar(theta,intensity,intensity_err,theta_err,"o")
plt.grid(True)
plt.show()
fig4.savefig("fig/plot14")

#%%
# #6
d = np.array([1,1.3,1.4,1.5,1.7,2,2.5,3,3.5,4,4.5,5,5.5])*1e-2 #meter
d_err = 1e-3 #meter
V = np.array([0.012,0.015,0.504,0.441,0.45,0.5,0.52,0.527,0.567,0.577,0.592,0.58,0.575])-0.012# V
intensity = V
intensity_err = 0.005
fig3= plt.figure(dpi=300)
plt.errorbar(d,intensity,intensity_err,d_err,fmt="o",label="Data")
plt.xlabel("d [m]", fontsize=14)
plt.ylabel("$V [V]$", fontsize=14)
plt.grid()
plt.show()
fig3.savefig("fig/plot15")


#%%


#7 
# find the minimum width d_min for which the intensity drops drasticly at
d_min = ufloat(1.4e-2,1e-3) #fill currect d_min
lambda_for_d = 2*d_min
print("d min is", d_min,"m")
print("The wavelen is",lambda_for_d,"m")

#10
d = 2e-2 # meter
x_err = 2e-3
x = np.array([12,10.2,8.5,6.6,4.6][::-1]) *1e-2 #meter
distance_between_nodes = np.abs(np.diff(x))
distance_between_nodes_err= sqrt(2) * x_err
I = np.array([0.383,0.39,0.395,0.395,0.395][::-1])
I_err =0.005

n = np.arange(0,len(x))
fig,fit = one4all(n,x,x_err,0,"linear",xlabel="n",ylabel="x [m]")
fig.savefig("fig/plot16")
Reg_print(fit)
m = ufloat(fit.slope,2*fit.stderr)
lambda_g_from_distance_between_nodes=2*m


# distance_between_nodes= distance_between_nodes.mean()
# distance_between_nodes_err = distance_between_nodes_err/np.sqrt(np.size(distance_between_nodes))
# distance_between_nodes = ufloat(distance_between_nodes,distance_between_nodes_err)
# lambda_g_from_distance_between_nodes=distance_between_nodes*2
print("Lambda g according to the diff in nodes is",lambda_g_from_distance_between_nodes,"m")


d = 2e-2 # meter
x_err = 2e-3
x = np.array([14.8,13,11.1,9.3,7.4,5.6][::-1]) *1e-2 #meter
distance_between_max = np.abs(np.diff(x))
distance_between_max_err= sqrt(2) * x_err

distance_between_nodes_max=np.array([])
I = np.array([0.528,0.522,0.512,0.503,0.504,0.51][::-1])
I_err =0.005

n = np.arange(0,len(x))
fig,fit = one4all(n,x,x_err,0,"linear",xlabel="n",ylabel="x [m]")
fig.savefig("fig/plot17")
Reg_print(fit)
m = ufloat(fit.slope,2*fit.stderr)
lambda_g_from_distance_between_max=2*m


#one4all(distance_between_nodes, I,I_err,distance_between_nodes_err,mode="none",None,"X","I")

# distance_between_max= distance_between_max.mean()
# distance_between_max_err = distance_between_max_err/np.sqrt(np.size(distance_between_max))
# distance_between_max = ufloat(distance_between_max,distance_between_max_err)




print("Lambda g according to the diff in picks is",lambda_g_from_distance_between_max,"m")

#%%


#13
d=np.array([1.6,2,2.3,2.5,2.7]) * 1e-2 #meter
d_err=1e-3 #meter

# d = unumpy.uarray(d,d_err)
# f = lambda x: ufloat(linregress(range(0,len(x)),x[::-1]).slope,2*linregress(range(0,len(x)),x[::-1]).stderr)
# distance_between_nodes = np.array([
#    f([13,10.3,7.5]),f([12,10.2,8.5,6.6,4.6]),f([13.5,12.6,10,8.3]),f([12.5,11.7,9,7.5]),f([12.3,10.7,9,7.4])
#    ]
#     ) * 1e-2

d = unumpy.uarray(d,d_err)
distance_between_nodes = np.array([
    np.abs(np.diff(np.array([13,10.3,7.5]))).mean(),
    np.abs(np.diff(np.array([12,10.2,8.5,6.6,4.6]))).mean(),
    np.abs(np.diff(np.array([13.5,12.6,10,8.3]))).mean(),
    np.abs(np.diff(np.array([12.5,11.7,9,7.5]))).mean(),
    np.abs(np.diff(np.array([12.3,10.7,9,7.4]))).mean()
    ]
    ) * 1e-2
distance_between_nodes_err = np.array([2e-3,2e-3,3e-3,3e-3,3e-3]) * sqrt(2) / np.sqrt(np.array([3,5,4,4,4])-1) # the sqrt(2) is from the diff and the other is from the mean
distance_between_nodes = unumpy.uarray(distance_between_nodes,distance_between_nodes_err)
lambda_g= distance_between_nodes*2
y = 1/lambda_g**2
#y_err= 2*uerr(lambda_g)/uval(lambda_g)**3

x=1/(2*d)**2
#x_err=4*d_err/(2*d)**3

(fig13,fit13)= one4all(uval(x),uval(y),uerr(y),uerr(x),"linear",None,r"$\frac{1}{(2d)^2}[\frac{1}{m^2}]$",r"$\frac{1}{\lambda_g^2}[\frac{1}{m^2}]$")
slope = ufloat(fit13.slope,2*fit13.stderr)
intercept = ufloat(fit13.intercept,2*fit13.intercept_stderr)
fig13.savefig("fig/plot18",bbox_inches='tight')
print("==> ","y=(",slope,")x","+(",intercept,")"," R^2=",fit13.rvalue**2)

lambda_found = 1/sqrt(intercept)
print("lambda found=",lambda_found,"m")
#check if slope makes sense - needs to be -1
#%%

# #16 

L = 15 #cm
L_err = 0.2 #cm
wavelen = 2.8 #cm
wavelen_err = 0
d_of_phi = lambda phi : 1/(2*np.sqrt(phi/(np.pi*L*wavelen)-(phi/(2*np.pi*L))**2))

d_linear = np.array([d_of_phi(np.pi*n) for n in range(1,5)])

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

theta2, V2 = zip(*sorted(zip(theta2,V2)))


fig2=plt.figure()
ax2=plt.axes(polar=True)
theta2 = np.array(theta2)*np.pi/180
theta_err2 = np.array(theta_err2)*np.pi/180
V2 = np.array(V2)
V_err2 = np.array(V_err2)

f = lambda x,a,phi: a*np.cos(x+phi)**2
fit_f = cfit(f,theta2,V2)
Rsqrue_f = Rsqrue(f(theta2,*fit_f[0]),V2)
a = ufloat(fit_f[0][0],2*np.sqrt(np.diag(fit_f[1])[0]))
phi = ufloat(fit_f[0][1],2*np.sqrt(np.diag(fit_f[1])[1]))
print("==> y = (",a,")cos^2(x +(",phi,")), R^2=",Rsqrue_f)

ax2.errorbar(theta2,V2,V_err2,theta_err2,"o",label="Data")
ax2.plot(theta2,f(theta2,*fit_f[0]),"-.",label=r"$a\cos^2(\theta)$")

plt.grid(True)
plt.legend(loc="upper left")
plt.show()
fig2.savefig("fig/plot19")
#%%
#cyclic polarization
L = 15
L_err = 0.8
wavelen = 2.8
wavelen_err = 0
d_of_phi = lambda phi : 1/(2*np.sqrt(phi/(np.pi*L*wavelen)-(phi/(2*np.pi*L))**2))
d_circler = np.array([d_of_phi(np.pi/2),d_of_phi(3*np.pi/2),d_of_phi(5*np.pi/2)])
d_circler = np.array([d_of_phi(np.pi/2+n*np.pi) for n in range(0,5)])


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

f = lambda x,a: a*np.ones(len(x))
fit_f = cfit(f,theta1,V1)
Rsqrue_f = Rsqrue(f(theta1,*fit_f[0]),V1)
a = ufloat(fit_f[0][0],2*np.sqrt(np.diag(fit_f[1])[0]))
print("==> y = (",a,"), R^2=",Rsqrue_f)


ax1.errorbar(theta1,V1,V_err1,theta_err1,"o", label="Data")
ax1.plot(theta1,f(theta1,*fit_f[0]),"-.",label=r"$y=a$")

plt.grid(True)
plt.legend(loc="upper left",fontsize=8)
plt.show()
fig1.savefig("fig/plot20")
#########
# d2 = d_circler[0]
# theta2 = []
# theta_err2 = []
# V2 = []
# V_err2 = [] 


# fig2=plt.figure()
# ax2=plt.axes(polar=True)
# theta2 = np.array(theta2)*np.pi/180
# theta_err2 = np.array(theta_err2)*np.pi/180
# V2 = np.array(V2)
# V_err2 = np.array(V_err2)

# ax2.errorbar(theta,intensity,intensity_err,theta_err,"o")
# plt.grid(True)
# plt.show()
