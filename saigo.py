#!/usr/bin/env python3

"""
Final Year project on Speech enhancement using filter banks
"""

# Script details
__author__ = ["Lukman Shaikh", "Keerti Karlingannavar", "Tushar Khot", "Sudarshan Parvatikar"]
__copyright__ = "Copyright 2018-2019, Kyonen-no-Project"
__credits__ = ["Lukman Shaikh", "Keerti Karlingannavar", "Tushar Khot", "Sudarshan Parvatikar"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = ["Lukman Shaikh", "Keerti Karlingannavar", "Tushar Khot", "Sudarshan Parvatikar"]
__email__ = None
__status__ = "Production"

###################################################################################################

import fractions
import warnings
from numpy import cos, sin, pi, absolute, pad, arange, asarray, array, log10, array_equal, \
     roll, kron, zeros, ceil, arange, sinc, kaiser, floor, mod, r_, int16
from numpy.random import randn
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter, freqs, butter, cheby1, \
     kaiser_beta, fftconvolve
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, \
     grid, axes, show, legend, suptitle
from scipy.io import wavfile
from random import randint as ri

warnings.filterwarnings("ignore")
#---------------------------------------------------------------------------------------------------

input_file = input("Enter the name of the file: ").strip()
if input_file == "":
    # if no input file is given, assume the default
    input_file = "./wav/yahama.wav"
else:
    # append wav extension to the given filename
    input_file = "./wav/" + input_file + ".wav"

#---------------------------------------------------------------------------------------------------

### Global variables

# Create a signal for demonstration.
sample_rate, x = wavfile.read(input_file)
nsamples = 4 * int(sample_rate)
t = arange(nsamples) / int(sample_rate)

if len(x) != len(t) and len(x) < len(t):
    # then pad x
    pad(x, (0, len(t) - len(x)), 'constant')
    t = t[:len(x)]
elif len(t) != len(x) and len(x) > len(t):
    # then trim x
    x = x[:len(t)]

original_signal = x[:len(t)]
x = x[:len(t)] + randn(len(t)) * ri(1,10)
x = x[:len(t)]

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
low = 20.0
high = 20000.0

# For the sake of FIR filter, take order as 11 to get better gain.
N1 = 11

# Use firwin with a Kaiser window to create a bandpass FIR filter.
taps = firwin(N1, [low/nyq_rate, high/nyq_rate], window=('kaiser', beta))

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

#####################################################
# Multirate block
#####################################################

def downsample(s, n, phase=0):
    """Decrease sampling rate by integer factor n with included offset phase.
    """
    n = int(n)
    return s[phase::n]



def upsample(s, n, phase=0):
    """Increase sampling rate by integer factor n  with included offset phase.
    """
    return roll(kron(s, r_[1, zeros(int(n)-1)]), phase)



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
        b = firwin(n, 1.0/r)
        a = 1
        f = lfilter(b, a, s)
    else: #iir
        if n is None:
            n = 8
        b, a = cheby1(n, 0.05, 0.8/r)
        f = lfilter(b, a, s)
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
    b = firwin(2*l*r+1, alpha/r);
    a = 1
    return r*lfilter(b, a, upsample(s, r))[r*l+1:-1]


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
        l = ceil((rejection_db-8.0) / (28.714 * roll_off_width))
  
        #ideal sinc filter
        t = arange(-l, l + 1)
        ideal_filter=2*p*stopband_cutoff_f*sinc(2*stopband_cutoff_f*t)  
  
        #determine parameter of Kaiser window
        #use empirical formula from [2] Chap 7, Eq. (7.62) p 474
        beta = kaiser_beta(rejection_db)
          
        #apodize ideal filter response
        h = kaiser(2*l+1, beta)*ideal_filter

    ls = len(s)
    lh = len(h)

    l = (lh - 1)/2.0
    ly = ceil(ls*p/float(q))

    #pre and postpad filter response
    nz_pre = floor(q - mod(l,q))
    nz_pre = int(nz_pre)
    hpad = h[-lh+nz_pre:]

    offset = floor((l+nz_pre)/q)
    nz_post = 0;
    while ceil(((ls-1)*p + nz_pre + lh + nz_post )/q ) - offset < ly:
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
    return downsample(fftconvolve(h, upsample(s, p)), q)

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
    cascaded1234 = addCascadedOutputs(cascaded123, filtered_signal_highpass4, 100)
    cascaded12345 = addCascadedOutputs(cascaded1234, filtered_signal_highpass5, 100)
    cascaded123456 = addCascadedOutputs(cascaded12345, filtered_signal_highpass6, 100)

    #----------------------------------------------------------------------------

    return cascaded123456

def addCascadedOutputs(list1, list2, m, n=0):
    # Adds 2 lists as:
    # newList = list1[:-m] + sum of (list1[-m:], list2[:m]) elements + list2[m:]
    # First convert the numpy arrays to lists, add them, return them
    newList = list(list1[:-m]) + add_elements(list(list1[-m:]), list(list2[:m])) + list(list1[m:])
    newList = asarray(newList)
    return newList

def add_elements(list1, list2):
    # adds elements of list1 and 2 parallely

    return [a + b for a, b in zip(list1, list2)]

multirate_output = cascadedMultiRate(x, nyq_rate, N=11)

###########################################################################################################

# write to wavfile
data2 = asarray(x, dtype=int16)
wavfile.write("./output/1_noisy_signal_output.wav", 44100, data2)
data2 = asarray(filtered_x, dtype=int16)
wavfile.write("./output/2_fir_output.wav", 44100, data2)
data2 = asarray(iir_output, dtype=int16)
wavfile.write("./output/3_iir_output.wav", 44100, data2)
data2 = asarray(multirate_output[:len(x)], dtype=int16)
wavfile.write("./output/4_multirate_output.wav", 44100, data2)

# Graph Block

# 1. Original vs Noisy
figure(1)
# Plot the original signal.
plot(t, original_signal, 'b', label='original signal')
# Plot Noisy signal
plot(t, x, 'c', label='noisy signal')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Original Signal vs Noisy Signal")

# 2. FIR vs IIR
figure(2)
# delay for FIRs
# delay = 0.5 * (N-1) / sample_rate

# Plot the original signal.
plot(t, original_signal, 'b', label='original signal')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
# plot(t[N-1:]-delay, filtered_x[N-1:], 'r', label='Filtered Signal of Bandpass FIR filter')
plot(t[N-1:], filtered_x[N-1:], 'r', label='Filtered Signal of Bandpass FIR filter')
plot(t, iir_output, 'y', label='Filtered Signal of Bandpass IIR filter')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Original Signal, FIR filter and IIR filter outputs")

# 3. Original vs IIR
figure(3)
# Plot the original signal.
plot(t, original_signal, 'b', label='original signal')
# Plot Multirate output
plot(t, iir_output, 'y', label='Filtered Signal of Bandpass IIR filter')
# Plot Fir Output
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Original Signal and IIR filter output")


# 4. Original vs IIR vs Multirate
figure(4)
# Plot the original signal.
plot(t, original_signal, 'b', label='original signal')
# Plot Multirate output
plot(t, multirate_output[:len(t)], 'g', label='Filtered Signal of Cascaded Multirate filter')
plot(t, iir_output, 'y', label='Filtered Signal of Bandpass IIR filter')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Original Signal, IIR filter and Multirate filter outputs")

# 5. Noisy vs Multirate
figure(5)
# Plot Noisy signal
plot(t, x, 'c', label='noisy signal')
# Plot Multirate output
plot(t, multirate_output[:len(t)], 'g', label='Filtered Signal of Cascaded Multirate filter')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Noisy Signal and Multirate filter output")

# 6. Original vs Multirate
figure(6)
# Plot the original signal.
plot(t, original_signal, 'b', label='original signal')
# Plot Multirate output
plot(t, multirate_output[:len(t)], 'g', label='Filtered Signal of Cascaded Multirate filter')
legend()
xlabel('time')
ylabel('amplitude')
grid(True)
suptitle("Original Signal and Multirate filter output")

# Display the graphs
show()

# End of code
