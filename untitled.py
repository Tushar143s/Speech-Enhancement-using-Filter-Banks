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

from numpy import cos, sin, pi, absolute, arange, asarray, array, log10
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter, freqs, butter, lfilter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend, suptitle
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.random import randn

#------------------------------------------------------------------------------------------

# Global variables

# Create a signal for demonstration.
# sample_rate = 44000
# nsamples = 44000 * 4
sample_rate, x = wavfile.read('./out.wav')
# sample_rate = 44000
nsamples = 4 * int(sample_rate)
t = arange(nsamples) / int(sample_rate)
# x = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
#         0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
#             0.1*sin(2*pi*23.45*t+.8)
print("<<<", len(x))
print(">>",len(t))

# if len(x) != len(t):
	# append zeroes to x


original_signal = x[:len(t)]
x = x[:len(t)] + randn(len(t)) * 0.08
x = x[:len(t)]
# print(len(t), len(x))
# print(sample_rate)



# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

#------------------------------------------------------------------------------------------

# Create FIR Filter and apply to x

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
# cutoff_hz = 10.0
low = 20.0
high = 20000.0

# Use firwin with a Kaiser window to create a bandpass FIR filter.
taps = firwin(N, [low/nyq_rate, high/nyq_rate], window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)

figure(1)

delay = 0.5 * (N-1) / sample_rate

# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')
# Plot Noisy signal
plot(t, x, label='noisy signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'green', label='filtered signal of fir')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Fir, IIR output and Original Signal")

show()