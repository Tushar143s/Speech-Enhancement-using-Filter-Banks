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
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter, freqs, butter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend
from scipy.io import wavfile
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------

# Global variables

# Create a signal for demonstration.
# sample_rate = 44000
# nsamples = 44000 * 4
# sample_rate, x = wavfile.read('./audio.wav')
sample_rate = 44000
nsamples = 4 * int(sample_rate)
t = arange(nsamples) / int(sample_rate)
x = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
        0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
            0.1*sin(2*pi*23.45*t+.8)
x = x[:len(t)]
print(len(t), len(x))
print(sample_rate)



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
taps = firwin(N, [low/nyq_rate, high/nyq_rate] , window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)

# #------------------------------------------------
# # Plot the FIR filter coefficients.
# #------------------------------------------------

# figure(1)
# plot(taps, 'bo-', linewidth=2)
# title('Filter Coefficients (%d taps)' % N)
# grid(True)

# #------------------------------------------------
# # Plot the magnitude response of the filter.
# #------------------------------------------------

# figure(2)
# clf()
# w, h = freqz(taps, worN=8000)
# plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
# xlabel('Frequency (Hz)')
# ylabel('Gain')
# title('Frequency Response')
# ylim(-0.05, 1.05)
# grid(True)

# # Upper inset plot.
# ax1 = axes([0.42, 0.6, .45, .25])
# plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
# xlim(0,8.0)
# ylim(0.9985, 1.001)
# grid(True)

# # Lower inset plot
# ax2 = axes([0.42, 0.25, .45, .25])
# plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
# xlim(12.0, 20.0)
# ylim(0.0, 0.0025)
# grid(True)

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.

delay = 0.5 * (N-1) / sample_rate

figure(1)
# Plot the original signal.
plot(t, x, label='original signal')
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-', label='shifted signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4, label='filtered signal')
legend()
xlabel('t')
grid(True)

# show()

#-------------------------------------------------------------------------

# IIR Filter

# reuse the above coefficients

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


out = butter_bandpass_filter(x, low, high, sample_rate)
print(out)

figure(2)
# Plot the original signal.
plot(t, x, label='original signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'green', linewidth=4, label='filtered signal of fir')
plot(t, out, 'yellow', linewidth=4, label='filtered signal of iir')
legend()
xlabel('t')
grid(True)

show()

#############################################################################################

# Multirate

from sp import multirate