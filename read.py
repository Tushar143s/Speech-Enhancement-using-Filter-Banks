"""The WAV audio format was developed by Microsoft and has become one of the primary formats of uncompressed audio.
It stores audio at about 10 MB per minute at a 44.1 kHz sample rate using stereo 16-bit samples.
The WAV format is by definition, the highest quality 16-bit audio format. """

from scipy.io.wavfile import read
import numpy
from operator import add


a = read("speech.wav")
wave_to_numpy_array=numpy.array(a[1],dtype=float)
print(wave_to_numpy_array)


#determinr lenth of wave_to_numpy_array
numpy_array_length=len(wave_to_numpy_array)
print('this is the wave_to_numpy_array')
print(numpy_array_length)


#determine length of wave_to_numpy_array
maximum=max(wave_to_numpy_array)
minimum=min(wave_to_numpy_array)
print('this is the miximum value in wave_to_numpy_array')
print(maximum)
print('this is the minimum value in wave_to_numpy_array')
print(minimum)


#generate an random numpy array
random_araay=numpy.random.randint(minimum,maximum,size=numpy_array_length)
print('this is random array (noise)')
print(random_araay)
random_araay_length=len(random_araay)
print('this is random array legth (noise)')
print(random_araay_length)


""" to combine the wave file numpy array
with random generated numpy array
so it can act as noise (element wise addition)"""

noisy_numpy_array=list( map(add, wave_to_numpy_array, random_araay) )
print('this is the noisey numpy array(element wise addition)')
print(noisy_numpy_array)



