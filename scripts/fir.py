#!/usr/bin/env python3

"""
This script contains functions relating to Finite Impulse Response Filter
"""

# Script details
__author__ = "Sudarshan Parvatikar"
__copyright__ = "Copyright 2018-2019, Kyonen-no-Project"
__credits__ = ["Sudarshan Parvatikar"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Sudarshan Parvatikar"
__email__ = "sudarshan.parvatikar@null.net"
__status__ = "Production"


##############################################################################

import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from .utilities import *
from random import uniform
from operator import add

order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz
T = 5.0             # seconds
n = int(T * fs)     # total number of samples
t = np.linspace(0, T, n, endpoint=False)

data = np.sin(uniform(1, 10)*2*np.pi*t) + uniform(1, 10)*np.cos(9*2*np.pi*t) \
	+ uniform(1, 10) * np.sin(uniform(1, 10)*2*np.pi*t) #+ np.random.uniform(low=0.1, high=0.9, size=(len(t),))

plt.subplot(1, 1, 1)
plt.plot(t, data, 'b-', label='data')
#plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

