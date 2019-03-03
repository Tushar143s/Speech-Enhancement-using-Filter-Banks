#!/usr/bin/env python3

"""
This script contains common functions and variables that can be use my all/many filter scripts.
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


#################################################################################################

from random import uniform
from operator import add
import numpy as np



order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz
T = 5.0             # seconds
n = int(T * fs)     # total number of samples
t = np.linspace(0, T, n, endpoint=False)

def generateSignal():
	""" Fn to generate a random input signal """

	# generate random sine + cosine + sine function
	data = np.sin(uniform(1, 10)*2*np.pi*t) + uniform(1, 10)*np.cos(9*2*np.pi*t) \
	+ uniform(1, 10) * np.sin(uniform(1, 10)*2*np.pi*t)

def generateNoise():
	""" Fn to generate a random noise signal """
	pass


def generateNoisySignal():
	""" Fn to generate a random noisy signal 
	params: t - time period """
	# noisySignal = []
	# for i, j in zip(generateSignal(), generateNoisySignal()):
	# 	noisySignal.append(i+j)
	# return noisySignal
	# return list(map(add, generateSignal(), generateNoisySignal()))
	pass