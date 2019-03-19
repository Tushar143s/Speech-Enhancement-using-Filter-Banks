#

from sp import multirate

import sys
import fractions
import numpy
from scipy import signal
from numpy import cos, sin, pi, absolute, arange, asarray, array, log10
# from sp import multirate4
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter, freqs, butter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend
from itertools import zip_longest
from random import randint as ri


sample_rate = 44000
nsamples = 4 * int(sample_rate)
t = arange(nsamples) / int(sample_rate)
original = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
        0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
            0.1*sin(2*pi*23.45*t+.8)


# noise = numpy.random.uniform(low=0, high=10, size=(int(len(t)/ri(1, 3000)),))
# corrupted_signal = [x + y for x, y in zip_longest(list(original), list(noise), fillvalue=0)]
# print(">>>>>>>>>>>>>>", len(original), len(corrupted_signal))
x = original[:len(t)]
print(x)



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


#------------------------------------------------------------------------------------------

# Multirate block

# -----------------------------------------------------

from sp import multirate

### Low Pass Filter Block


def lowPassFilter(input_signal, order, nyq_rate):
	# Create a low pass filter with cut off frequency 20k Hz -> pass all freq below 20k Hz
	# The cutoff frequency of the filter.
	cutoff_hz = 20000.0

	# Use firwin with a Kaiser window to create a lowpass FIR filter.
	taps = firwin(order, cutoff_hz/nyq_rate, window=('kaiser', beta))

	# Use lfilter to filter x with the FIR filter.
	filtered_x = lfilter(taps, 1.0, input_signal)

	return filtered_x

#### /Low Pass Filter Block

### High Pass Filter Block

def highPassFilter(input_signal, order, nyq_rate):
	# Create a High pass filter, with cut off freq 20 Hz -> pass all freq above 20k Hz
	# The cutoff frequency of the filter.
	cutoff_hz = 20.0

	# Use firwin with a Kaiser window to create a lowpass FIR filter.
	taps = firwin(order, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)

	# Use lfilter to filter x with the FIR filter.
	filtered_x = lfilter(taps, 1.0, input_signal)

	return filtered_x

### /High Pass Filter Block
# ------------------------------------------------------

# 1. Pass Signal via the low pass filter and get output

filtered_signal = lowPassFilter(x, N, nyq_rate)

print(filtered_signal)

# 2. Down Sample the flitered signal

# 3. Upsample the filtered signal

# 4. Pass signal via high pass filter and get output

filtered_signal = lowPassFilter(x, N, nyq_rate)
print(filtered_signal)

# 5. Add the signals to get output signals

delay = 0.5 * (N-1) / sample_rate

figure(1)
# Plot the original signal.
plot(t, original, label='original signal')
# Plot the filtered signal, shifted to compensate for the phase delay.
# plot(t-delay, filtered_x, 'r-', label='shifted signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
# plot(t[N-1:]-delay, filtered_signal[N-1:], 'g', linewidth=4, label='filtered signal')
legend()
xlabel('t')
grid(True)
show()