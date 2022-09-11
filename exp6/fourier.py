
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit


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
    
    
    
    
#%% prep
# L = 62 #mm
# x = np.linspace(0, 62,1000)
# wavelen0 = 2*L / 32
# wavelen = [n*wavelen0 for n in [1,2,4,8,16,32]]
# wave = [np.sin(2*np.pi/l*x) for l in wavelen]
# plt.figure()
# tot_wave = 0
# for w in wave:
#  #   plt.plot(x,w,"-.")
#     tot_wave +=w
# plt.plot(x,tot_wave,label="superposition")
# plt.grid()
# plt.legend()
# plt.show()

#%% ft for jpg
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist



def fft(file):
    img = cv2.imread(path_name,0)
    img = img[:1024,:1024]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = ((np.abs(fshift)))**0.5
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image:' + file), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
def ifft(file):
    img = cv2.imread(path_name,0)
    img = img[:1024,:1024]
    img_shift = np.fft.ifftshift(img)
    new_img = np.fft.ifft2(img)
    magnitude_spectrum = 20*np.log(np.abs(new_img))
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image:' + file), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def fourier_transform_image(image_location):
    im= imread(image_location)[:,:,:3]
    im = im[:1024, :1024]
    im= rgb2gray(im)
    plt.figure(dpi=300)
    

    im_fourier = np.fft.fftshift(np.fft.fft2(im))

    im_fourier = (100*np.log(abs(im_fourier))**2)

    plt.subplot(121), plt.imshow(im, cmap="gray")
    plt.title('Input Image:' + image_location), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(im_fourier, cmap = 'gray')
    plt.title('fourier transform image '), plt.xticks([]), plt.yticks([])
    plt.show()
    

def ifourier_transform_image(image_location):
    print("check if got here")
    im= imread(image_location)[:,:,:3]
    im = im[:1024, :1024]
    im= rgb2gray(im)
    plt.figure(dpi=300)

    im_fourier = np.fft.ifftshift(np.fft.ifft2(im))

    im_fourier = (100*np.log(abs(im_fourier)**0.001))


    plt.subplot(121), plt.imshow(im, cmap="gray")
    plt.title('Input Image:' + image_location), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(im_fourier, cmap = 'gray')
    plt.title('Inverse fourier transform image '), plt.xticks([]), plt.yticks([])
    plt.show()

files = os.listdir("fig")
for file in files:
    if (file.find("NF") != -1):
        path_name = str("fig/" + file)
        fourier_transform_image(path_name)
        
        
    if (file.find("FF") != -1):
        path_name = str("fig/" + file)
        ifourier_transform_image(path_name)
        
