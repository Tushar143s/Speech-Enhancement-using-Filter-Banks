# This file contains some faqs

1. Why use kaiser window?
A. Please refer `https://www.quora.com/Why-is-kaiser-window-superior-to-other-window-functions` for details. From the above link: `In  Kaiser window, the passband and stopband ripple size are not affected much by the change in the window length(when compared to other windows). It is affected by how the coefficients roll off(the shape factor).The window length mainly affects the transition band.
So keeping the window length constant, we can adjust the shape factor, to design for the passband and stopband ripples. This is a huge advantage over other windows, where the window length, ripple size and transition bandwidth have a three-way tradeoff.
The only disadvantage with the Kaiser window is that we design for the same ripple size in both passband and stopband. So there would be some overdesign in either of the bands.`

2. Why is the band of the filter(s) from 20 to 20kHz?
A. Because that's the Hearing range of humans. Refer `https://en.wikipedia.org/wiki/Hearing_range` for more.

3. Why use wav files?
A. Beacause they are simple to use and hence it is easy to read from and write to a wav file. This is because, unlike other formats, wav files don't contain a lot of metadata. Refer `https://en.wikipedia.org/wiki/WAV` for more.

4. What's the sampling rate?
A. It's 44.1 kHz. Its so, because 44.1 Khz wav files are more abundant, easy to manipulate beacuse of less number of values. Refer `https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy.io.wavfile.read` for more.

5. What is Nyquist rate?
A.

6. What is the order of the filter?
A. The order is automatically calculated by the inbuilt function(s), depending upon the input signal. However for FIR filter, in order to maximize gain, we have taken a smaller order of 11.

7. What windows did you use?
A. For FIR: kaiser, IIR: butterworth, Multirate: kaiser. Refer `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html` and `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html`.

8. What are the packages/libraries used?
A. numpy, scipy, pylab, matplotlib, random, sys.

9. Why do you add noise even when noise is present in the wav files?
A. The noise in the wav files are the noise occuring during the recording/converting process. The noise we add is to simulate the channel noise that is present, whilst digital/analog signals are sent from sender to reciever over a channel.

10. In conclusion, which filter is better?
A. FIR filters are more powerful than IIR filters, but also require more processing power and more work to set up the filters. They are also less easy to change "on the fly" as you can by tweaking (say) the frequency setting of a parametric (IIR) filter. However, their greater power means more flexibility and ability to finely adjust the response of your active loudspeaker. However, since the nature and type of noises, which occur during recording/converting process and the channel noise is unpredictable, it is seen that the more general purpose IIR filters perform better. Refer `https://www.minidsp.com/applications/dsp-basics/fir-vs-iir-filtering`. However, the performance of Cascaded Multirate Filter Bank, which has been the focus of this project, is more better than the aforementioned filters. Hence in conclusion, we can say that Cascaded Multirate Filter Bank is better than FIR or IIR filters.