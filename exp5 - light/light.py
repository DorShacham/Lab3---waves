import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

#%% scatering of polarized light

#5
# summary in writing: qualative comparison of light intensity : 
'''
up (perpendicular to the table):

side (parallel to the optic table):

'''
#6
'''
The direction of polraization is 
Because

'''
#7
#5 and 6 again but after rotation of 90 degrees
'''
up (perpendicular to the table):

side (parallel to the optic table):

The direction of polraization is 
Because
'''

#8
#Total summary
'''
what happend was
which means the direction of the polarizors is
because
which means
the direction of dipoles is
because
'''
#%% reflection of polorized light
#2
'''
explanation of where we placed the polarizor and why:

'''
#3
brooster_deg = 0 #in degrees
booster_deg_err =0
#4
polarization_direction = 0 #in degrees
'''
explanation of how we found it or why that is the polarization:
'''

#5
n = np.tan(brooster_deg * np.pi / 180)

#6
'''
outside source for the refractive index:
statistical comparison:
'''

#%% interference and diffraction
#6
sign_of_screw_while_closed = 0 
#8
d = 0 #width of slit, for the future

#10
x_of_nodes = np.array([])
x_of_nodes_err =np.array([])
#lets do it tomorrow...

#11
