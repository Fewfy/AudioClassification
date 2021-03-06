import logging
import numpy
import decimal
import math
from scipy.fftpack import dct


class Extractor(object):
    def __init__(self):
        pass

    def mfcc(self, signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22,
             appendEnergy=True, winfunc=lambda x:numpy.ones((x, ))):
        """Compute the MFCC features from audio signal
        :param signal: the audio signal, 1-d array
        :param samplerate: the samplerate of the signal, default to 16000
        :param winlen: the length of the window
        :param winstep: the length of each step of the move of window
        :param numcep: the number cepstrum to return
        :param nfilt: the number of filters
        :param nfft: the FFT size
        :param lowfreq: the lower band edge of mel filters
        :param highfreq: the higher band edge of mel filters
        :param preemph: apply preemphasis filter with preemph as coefficent
        :param cepfilter: apply a lifter to final cepstral coefficients.
        :param appendEnergy: if true, the zeroth cepstral coefficient is replaced with the log of the total frame energy
        :param winfunc: the analysis windwo to apply to each frame
        :returns: a numpy array of size()
        """
        signal = self.preemphasis(signal, preemph)
        frames = self.framelize(signal, frame_len=winlen * samplerate, frame_step=winstep * samplerate, winfunc=winfunc)
        powspec = self.pow_spec(frames=frames, NFFT=nfft)
        energy = numpy.sum(powspec, 1)
        energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)
        fb = self.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
        feat = numpy.dot(powspec, fb.T)
        feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)
        
        feat = numpy.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
        feat = self.lifter(feat, ceplifter)
        return feat
        
        
    def preemphasis(self, signal, coeff=0.97):
        """
        preemphasis the signal with the preemph(defalt to 0.97)
        to make highlight the high frequency, make the spectrum more smoothy
        :param signal: The signal to filter
        :param coeff: the preemphasis coefficient
        :return: the filtered signal
        """
        return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

    def framelize(self, signal, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), strid_trick=True):
        """
        Make the signal to frames
        :param signal: numpy array
        :param frame_len: the length of each frame
        :param frame_step: the step of frame
        :param winfunc: the analysis window to apply to each frame
        :param strid_trick: use stride trick to compute the rolling window and window multiplication faster
        :return: an array of frames
        """
        signal_length = len(signal)
        frame_len = int(self.round_half_up(frame_len))
        frame_step = int(self.round_half_up(frame_step))
        if signal_length < frame_len:
            num_frames = 1
        else:
            num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_len) / frame_step))
        pad_len = (num_frames - 1) * frame_step + frame_len
        zeros = numpy.zeros((pad_len - signal_length))
        pad_signal = numpy.concatenate(signal, zeros)
        if strid_trick:
            win = winfunc(frame_len)
            frames = self.rolling_window(pad_signal, window=frame_len, step=frame_len)
        else:
            indices = numpy.tile(numpy.arange(0, frame_len), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
            indices = numpy.array(indices, dtype=numpy.float32)
            frames = pad_signal[indices]
            win = numpy.tile(winfunc(frame_len), (num_frames, 1))
    
    def rolling_window(self, a, window, step=1):
        """
        
        :param a: an numpy array
        :param window:
        :param step:
        :return:
        """
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

    def round_half_up(self, number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

    def pow_spec(self, frames, NFFT):
        """
        compute the power spectrum of each frame
        :param frames: the array of frames, each row is a frame
        :param NFFT: the FFT length to use.
        :return: if the frames is an NxD matrix, output will be Nx(NFFT/2+1)
        """
        return 1.0 / NFFT * numpy.square(self.magspec(frames, NFFT))

    def magspec(self, frames, NFFT):
        """
        compute the magnitude spectrum of each frame in frames
        :param frames: the array of frames, each row is a frame
        :param NFFT: the FFT length
        :return:
        """
        if numpy.shape(frames)[1] > NFFT:
            logging.warn('frame length is greater than FFT size, frame will be truncated')
        complex_spec = numpy.fft.rfft(frames, NFFT)
        return numpy.absolute(complex_spec)
        
    def get_filterbanks(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        highfreq = highfreq or samplerate / 2
        assert highfreq <= samplerate / 2
        
        lowmel = self.hztomel(lowfreq)
        highmel = self.hztomel(highfreq)
        melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
        bin = numpy.floor((nfft + 1) * self.meltohz(melpoints) / samplerate)
        
        fbank = numpy.zeros([nfilt, nfft // 2 + 1])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j, i] = (i - bin[j]) / (bin[j+1] - bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j, i] = (bin[j+2] - i) / (bin[j+2] - bin[j+1])
        return fbank

    def hztomel(self, hz):
        return 2595 * numpy.log(1 + hz / 700.)

    def meltohz(self, mel):
        return 700 * (10 **(mel / 2595.0) - 1)

    def lifter(self, cepstra, L=22):
        if L > 0:
            nframes, ncoeff = numpy.shape(cepstra)
            n = numpy.arange(ncoeff)
            lift = 1 + (L/2.)*numpy.sin(numpy.pi * n / L)
            return lift*cepstra
        else:
            return cepstra
        

