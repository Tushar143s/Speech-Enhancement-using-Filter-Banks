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

from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from scipy.io import wavfile

##############################################################################

# read sampling rate and data values from wavfile
# fs is sample rate and data is the amplitude values
fs, data = wavfile.read('./audio.wav')
nsamples = 1000
# t is the time axis
t = arange(nsamples) / fs

print(fs, data, t)

# Create an FIR filter

# The Nyquist rate of the signal.
nyq_rate = fs / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)



# The cutoff frequency of the filter.
cutoff_hz1 = 20.0
cutoff_hz2 = 20000.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, [cutoff_hz1/nyq_rate, cutoff_hz2/nyq_rate], window=('kaiser', beta))
print("here")
# Use lfilter to filter data with the FIR filter.
filtered_x = lfilter(taps, 1.0, data[0:1000])
print("there")
#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
grid(True)

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(2)
clf()
w, h = freqz(taps, worN=8000)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
ylim(-0.05, 1.05)
grid(True)

# Upper inset plot.
ax1 = axes([0.42, 0.6, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlim(0,8.0)
ylim(0.9985, 1.001)
grid(True)

# Lower inset plot
ax2 = axes([0.42, 0.25, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlim(12.0, 20.0)
ylim(0.0, 0.0025)
grid(True)

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / fs

figure(3)
# Plot the original signal.
plot(t, data[0:1000])
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

xlabel('t')
grid(True)

show()



