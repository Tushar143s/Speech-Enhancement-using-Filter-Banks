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
taps = firwin(N, [low/nyq_rate, high/nyq_rate] , window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)


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


iir_output = butter_bandpass_filter(x, low, high, sample_rate)

# figure(2)
# # Plot the original signal.
# plot(t, x, label='original signal')
# # Plot just the "good" part of the filtered signal.  The first N-1
# # samples are "corrupted" by the initial conditions.
# plot(t[N-1:]-delay, filtered_x[N-1:], 'green', linewidth=4, label='filtered signal of fir')
# plot(t, iir_output, 'yellow', linewidth=4, label='filtered signal of iir')
# legend()
# xlabel('t')
# grid(True)

# show()

#############################################################################################

# Multirate

import sys
import fractions
import numpy
from scipy import signal
from numpy import cos, sin, pi, absolute, arange, asarray, array, log10
# from sp import multirate4
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter, freqs, butter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend
from random import randint as ri


# # sample_rate = 44000 # set sample rate between 9 kHz to 11 kHz (original: 44 kHz)
# # nsamples = 4 * int(sample_rate)
# # t = arange(nsamples) / int(sample_rate)
# # original = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
# #         0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
# #             0.1*sin(2*pi*23.45*t+.8)


# # noise = numpy.random.uniform(low=0, high=10, size=(int(len(t)/ri(1, 3000)),))
# # corrupted_signal = [x + y for x, y in zip_longest(list(original), list(noise), fillvalue=0)]
# # print(">>>>>>>>>>>>>>", len(original), len(corrupted_signal))
# # x = original[:len(t)]



# # The Nyquist rate of the signal.
# nyq_rate = sample_rate / 2.0

# # The desired width of the transition from pass to stop,
# # relative to the Nyquist rate.  We'll design the filter
# # with a 5 Hz transition width.
# width = 5.0/nyq_rate

# # The desired attenuation in the stop band, in dB.
# ripple_db = 60.0

# #------------------------------------------------------------------------------------------

# # Create FIR Filter and apply to x

# # Compute the order and Kaiser parameter for the FIR filter.
# N, beta = kaiserord(ripple_db, width)

# # The cutoff frequency of the filter.
# # cutoff_hz = 10.0
# low = 20.0
# high = 20000.0


#------------------------------------------------------------------------------------------

# Multirate block

# -----------------------------------------------------

#####################################################

def downsample(s, n, phase=0):
    """Decrease sampling rate by integer factor n with included offset phase.
    """
    n = int(n)
    return s[phase::n]



def upsample(s, n, phase=0):
    """Increase sampling rate by integer factor n  with included offset phase.
    """
    return numpy.roll(numpy.kron(s, numpy.r_[1, numpy.zeros(int(n)-1)]), phase)



def decimate(s, r, n=None, fir=False):
    """Decimation - decrease sampling rate by r. The decimation process filters 
    the input data s with an order n lowpass filter and then resamples the 
    resulting smoothed signal at a lower rate. By default, decimate employs an 
    eighth-order lowpass Chebyshev Type I filter with a cutoff frequency of 
    0.8/r. It filters the input sequence in both the forward and reverse 
    directions to remove all phase distortion, effectively doubling the filter 
    order. If 'fir' is set to True decimate uses an order 30 FIR filter (by 
    default otherwise n), instead of the Chebyshev IIR filter. Here decimate 
    filters the input sequence in only one direction. This technique conserves 
    memory and is useful for working with long sequences.
    """
    if fir:
        if n is None:
            n = 30
        b = signal.firwin(n, 1.0/r)
        a = 1
        f = signal.lfilter(b, a, s)
    else: #iir
        if n is None:
            n = 8
        b, a = signal.cheby1(n, 0.05, 0.8/r)
        f = signal.lfilter(b, a, s)
    return downsample(f, r)



def interp(s, r, l=11, alpha=0.5):
    """Interpolation - increase sampling rate by integer factor r. Interpolation 
    increases the original sampling rate for a sequence to a higher rate. interp
    performs lowpass interpolation by inserting zeros into the original sequence
    and then applying a special lowpass filter. l specifies the filter length 
    and alpha the cut-off frequency. The length of the FIR lowpass interpolating
    filter is 2*l*r+1. The number of original sample values used for 
    interpolation is 2*l. Ordinarily, l should be less than or equal to 10. The 
    original signal is assumed to be band limited with normalized cutoff 
    frequency 0=alpha=1, where 1 is half the original sampling frequency (the 
    Nyquist frequency). The default value for l is 4 and the default value for 
    alpha is 0.5.
    """
    b = signal.firwin(2*l*r+1, alpha/r);
    a = 1
    return r*signal.lfilter(b, a, upsample(s, r))[r*l+1:-1]


def resample(s, p, q, h=None):
    """Change sampling rate by rational factor. This implementation is based on
    the Octave implementation of the resample function. It designs the 
    anti-aliasing filter using the window approach applying a Kaiser window with
    the beta term calculated as specified by [2].
    
    Ref [1] J. G. Proakis and D. G. Manolakis,
    Digital Signal Processing: Principles, Algorithms, and Applications,
    4th ed., Prentice Hall, 2007. Chap. 6

    Ref [2] A. V. Oppenheim, R. W. Schafer and J. R. Buck, 
    Discrete-time signal processing, Signal processing series,
    Prentice-Hall, 1999
    """
    gcd = fractions.gcd(p,q)
    if gcd>1:
        p=p/gcd
        q=q/gcd
    
    if h is None: #design filter
        #properties of the antialiasing filter
        log10_rejection = -3.0
        stopband_cutoff_f = 1.0/(2.0 * max(p,q))
        roll_off_width = stopband_cutoff_f / 10.0
    
        #determine filter length
        #use empirical formula from [2] Chap 7, Eq. (7.63) p 476
        rejection_db = -20.0*log10_rejection;
        l = numpy.ceil((rejection_db-8.0) / (28.714 * roll_off_width))
  
        #ideal sinc filter
        t = numpy.arange(-l, l + 1)
        ideal_filter=2*p*stopband_cutoff_f*numpy.sinc(2*stopband_cutoff_f*t)  
  
        #determine parameter of Kaiser window
        #use empirical formula from [2] Chap 7, Eq. (7.62) p 474
        beta = signal.kaiser_beta(rejection_db)
          
        #apodize ideal filter response
        h = numpy.kaiser(2*l+1, beta)*ideal_filter

    ls = len(s)
    lh = len(h)

    l = (lh - 1)/2.0
    ly = numpy.ceil(ls*p/float(q))

    #pre and postpad filter response
    nz_pre = numpy.floor(q - numpy.mod(l,q))
    nz_pre = int(nz_pre)
    hpad = h[-lh+nz_pre:]

    offset = numpy.floor((l+nz_pre)/q)
    nz_post = 0;
    while numpy.ceil(((ls-1)*p + nz_pre + lh + nz_post )/q ) - offset < ly:
        nz_post += 1
    hpad = hpad[:lh + nz_pre + nz_post]

    #filtering
    xfilt = upfirdn(s, hpad, p, q)

    return xfilt[int(offset)-1:int(offset)-1+int(ly)]


def upfirdn(s, h, p, q):
    """Upsample signal s by p, apply FIR filter as specified by h, and 
    downsample by q. Using fftconvolve as opposed to lfilter as it does not seem
    to do a full convolution operation (and its much faster than convolve).
    """
    return downsample(signal.fftconvolve(h, upsample(s, p)), q)

##################################################################################



### Low Pass Filter Block


def lowPassFilter(input_signal, nyq_rate, order=11, cutoff_hz=20000.0):
	# Create a low pass filter with cut off frequency 20k Hz -> pass all freq below 20k Hz
	# The cutoff frequency of the filter.
	# cutoff_hz = 20000.0

	# Use firwin with a Kaiser window to create a lowpass FIR filter.
	taps = firwin(order, cutoff_hz/nyq_rate, window=('kaiser', beta))

	# Use lfilter to filter x with the FIR filter.
	filtered_x = lfilter(taps, 1.0, input_signal)

	return filtered_x

#### /Low Pass Filter Block

### High Pass Filter Block

def highPassFilter(input_signal, nyq_rate, order=11, cutoff_hz=20.0):
	# Create a High pass filter, with cut off freq 20 Hz -> pass all freq above 20k Hz
	# The cutoff frequency of the filter.
	# cutoff_hz = 20.0

	# Use firwin with a Kaiser window to create a lowpass FIR filter.
	taps = firwin(order, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)

	# Use lfilter to filter x with the FIR filter.
	filtered_x = lfilter(taps, 1.0, input_signal)

	return filtered_x

### /High Pass Filter Block

# ------------------------------------------------------

def cascadedMultiRate(x, nyq_rate, N=11):
	# Cascaded signal

	# Params: x=signal, N=order, nyq_rate = nyquist rate

	# Break into cascades into 6 stages, of ranges -> low, high

	# Imp: Set rate of downsampling and upsampling to 11
	downsample_rate = 11
	upsample_rate = 11
	filter_length = 9 

	### Stage 1: low = 20.0 Hz, high = 100 Hz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass1 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=100.0) # x: signal, N=order, cutoff_hz=cutoff of LPF(pass signals lower this)


	# 2. Down Sample the flitered signal

	deciminated_signal1 = decimate(filtered_signal_lowpass1, downsample_rate, fir=False) 


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal1 = interp(deciminated_signal1, upsample_rate, l=filter_length, alpha=0.5) 

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass1 = highPassFilter(intropolated_signal1, nyq_rate, order=N, cutoff_hz=20.0)
	filtered_signal_highpass1 = resample(filtered_signal_highpass1, upsample_rate, downsample_rate)

	### /Stage 1: low = 20.0 Hz, high = 100 Hz

	### Stage 2: low = 95.0 Hz, high = 1 kHz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass2 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=1000.0)


	# 2. Down Sample the flitered signal

	deciminated_signal2 = decimate(filtered_signal_lowpass2, downsample_rate, fir=False)


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal2 = interp(deciminated_signal2, upsample_rate, l=filter_length, alpha=0.5) 

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass2 = highPassFilter(intropolated_signal2, nyq_rate, order=N, cutoff_hz=95.0)
	filtered_signal_highpass2 = resample(filtered_signal_highpass2, upsample_rate, downsample_rate)

	### /Stage 2: low = 95.0 Hz, high = 1 kHz

	### Stage 3: low = 0.9 kHz, high = 5 kHz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass3 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=900.0)

	# 2. Down Sample the flitered signal

	deciminated_signal3 = decimate(filtered_signal_lowpass3, downsample_rate, fir=False)


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal3 = interp(deciminated_signal3, upsample_rate, l=filter_length, alpha=0.5)

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass3 = highPassFilter(intropolated_signal3, nyq_rate, order=N, cutoff_hz=5000.0)
	filtered_signal_highpass3 = resample(filtered_signal_highpass3, upsample_rate, downsample_rate)

	### /Stage 3: low = 0.9 kHz, high = 5 kHz

	### Stage 4: low = 4.9 kHz, high = 10 kHz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass4 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=4900.0)


	# 2. Down Sample the flitered signal

	deciminated_signal4 = decimate(filtered_signal_lowpass4, downsample_rate, fir=False)


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal4 = interp(deciminated_signal4, upsample_rate, l=filter_length, alpha=0.5)

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass4 = highPassFilter(intropolated_signal4, nyq_rate, order=N, cutoff_hz=10000.0)
	filtered_signal_highpass4 = resample(filtered_signal_highpass4, upsample_rate, downsample_rate)

	### /Stage 4: low = 4.9 kHz, high = 10 kHz

	### Stage 5: low = 9.9 kHz, high = 15 kHz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass5 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=9900.0)


	# 2. Down Sample the flitered signal

	deciminated_signal5 = decimate(filtered_signal_lowpass5, downsample_rate, fir=False)


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal5 = interp(deciminated_signal5, upsample_rate, l=filter_length, alpha=0.5)

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass5 = highPassFilter(intropolated_signal5, nyq_rate, order=N, cutoff_hz=15000.0)
	filtered_signal_highpass5 = resample(filtered_signal_highpass5, upsample_rate, downsample_rate)

	### /Stage 5: low = 9.9 kHz, 15 kHz

	### Stage 6: low = 14.9 kHz, 20 kHz

	# 1. Pass Signal via the low pass filter and get output

	filtered_signal_lowpass6 = lowPassFilter(x, nyq_rate, order=N, cutoff_hz=14900.0)


	# 2. Down Sample the flitered signal

	deciminated_signal6 = decimate(filtered_signal_lowpass6, downsample_rate, fir=False)


	# 3. Upsample/intropolate the filtered signal

	intropolated_signal6 = interp(deciminated_signal6, upsample_rate, l=filter_length, alpha=0.5)

	# 4. Pass signal via high pass filter and get output

	filtered_signal_highpass6 = highPassFilter(intropolated_signal6, nyq_rate, order=N, cutoff_hz=20000.0)
	filtered_signal_highpass6 = resample(filtered_signal_highpass6, upsample_rate, downsample_rate)

	### /Stage 6: low = 14.9 kHz, high = 20 kHz


	


	# 5. Add the signals to get output signals

	# output of cascaded mn stages, where the overlapping samples are added
	cascaded12 = addCascadedOutputs(filtered_signal_highpass1, filtered_signal_highpass2, 5)
	cascaded123 = addCascadedOutputs(cascaded12, filtered_signal_highpass3, 100)
	cascaded1234 = addCascadedOutputs(cascaded12, filtered_signal_highpass4, 100)
	cascaded12345 = addCascadedOutputs(cascaded12, filtered_signal_highpass5, 100)
	cascaded123456 = addCascadedOutputs(cascaded12, filtered_signal_highpass6, 100)

	#----------------------------------------------------------------------------

	# return final output
	print(len(cascaded123456))
	
	return cascaded123456




def addCascadedOutputs(list1, list2, m, n=0):
	# Adds 2 lists as:
	# newList = list1[:-m] + sum of (list1[-m:], list2[:m]) elements + list2[m:]

	# First convert the numpy arrays to lists, add them, return them
	newList = list(list1[:-m]) + add_elements(list(list1[-m:]), list(list2[:m])) + list(list1[m:])
	# print(len(list1[:-m]), len(list1[-m:]),len(list2[:m]), len(list1[m:]))

	newList = asarray(newList)

	return newList


def add_elements(list1, list2):
	# adds elements of list1 and 2 parallely

	return [a + b for a, b in zip(list1, list2)]


multirate_output = cascadedMultiRate(x, nyq_rate, N=11)


###########################################################################################################
# Plotting Graph Area


# 1. Fir vs IIR vs Original Signal output

figure(1)

delay = 0.5 * (N-1) / sample_rate

# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')
# Plot Noisy signal
plot(t, x, label='noisy signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'green', label='filtered signal of fir')
plot(t, iir_output, 'yellow', label='filtered signal of iir')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Fir, IIR output and Original Signal")

# 2. Fir vs Original Signal 

figure(2)
# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')
# Plot Noisy signal
plot(t, x, label='noisy signal')
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-', label='shifted signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', label='filtered signal')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Fir output and Original Signal")

# 3. Fir vs Multirate vs Original Signal

figure(3)
# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')
# Plot Noisy signal
plot(t, x, label='noisy signal')
# Plot Multirate output
plot(t, multirate_output[:len(t)], 'black', label='multirate output signal')
# Plot Fir Output
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', label='filtered signal')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Fir, Multirate output and Original Signal")


# 4. Multirate vs Original Signal

figure(4)
# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')
# Plot Noisy signal
plot(t, x, label='noisy signal')
# Plot Multirate output
plot(t, multirate_output[:len(t)], 'black', label='multirate output signal')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Multirate output, Noisy and Original Signal")

# 5. original Signal vs Multirate output 

figure(5)
# Plot the original signal.
plot(t, original_signal, 'm', label='original signal')

# Plot Multirate output
plot(t, multirate_output[:len(t)], 'black', label='multirate output signal')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Multirate output and Original Signal")

show()