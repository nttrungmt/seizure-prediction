import numpy as np
from scipy import signal
from scipy.signal import resample, hann
from sklearn import preprocessing

# optional modules for trying out different transforms
try:
    import pywt
except ImportError, e:
    pass

try:
    from scikits.talkbox.features import mfcc
except ImportError, e:
    pass


# NOTE(mike): All transforms take in data of the shape (NUM_CHANNELS, NUM_FEATURES)
# Although some have been written work on the last axis and may work on any-dimension data.


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """
    def __init__(self, window=None):
        self.window = window

    def get_name(self):
        if self.window is not None:
            return "fft-%s"%self.window
        else:
            return "fft"

    def apply(self, data):
        if self.window is None:
            axis = data.ndim - 1
            return np.fft.rfft(data, axis=axis)
        nsamples = data.shape[-1]
        if self.window.endswith('0'):
            n = nsamples
        elif self.window.endswith('P2'):
            n = 512
            while n/2 < nsamples:
                n *= 2
        else:
            n = 2*nsamples
        if self.window.startswith('hamming'):
            res = np.fft.rfft(data*np.hamming(nsamples), n=n) # add zero padding
        else:
            res = np.fft.rfft(data, n=n) # add zero padding

        # return to original frequencies
        if n == nsamples:
            return res

        nf = res.shape[1]
        nf_down = (nsamples//2)+1
        res_down = np.empty((data.shape[0],nf_down), dtype=complex)
        for i in range(nf_down):
            j = (i*(nf-1))//(nf_down-1)
            res_down[:,i] = res[:,j]
        return res_down




class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d-%d" % (self.start, self.end)

    def apply(self, data):
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]

class Band:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, bands, Fs=399.61):
        self.bands = bands
        self.Fs = Fs

    def get_name(self):
        return 'band-' + '-b'.join(map(str,self.bands))

    def apply(self, data):
        res = np.empty((data.shape[0],len(self.bands)-1))
        for i,(s,e) in enumerate(zip(self.bands[:-1],self.bands[1:])):
            nyq = (self.Fs/2.)/(data.shape[1]-1)
            istart = int(s/nyq + 0.5)
            iend = int(e/nyq + 0.5)
            res[:,i] = data[:,istart:iend].sum(axis=-1)
        return res

class LPF:
    """
    Low-pass filter using FIR window
    """
    def __init__(self, f):
        self.f = f

    def get_name(self):
        return 'lpf%d' % self.f

    def apply(self, data):
        nyq = self.f / 2.0
        cutoff = min(self.f, nyq-1)
        h = signal.firwin(numtaps=101, cutoff=cutoff, nyq=nyq)

        # data[i][ch][dim0]
        for i in range(len(data)):
            data_point = data[i]
            for j in range(len(data_point)):
                data_point[j] = signal.lfilter(h, 1.0, data_point[j])

        return data


class MFCC:
    """
    Mel-frequency cepstrum coefficients
    """
    def get_name(self):
        return "mfcc"

    def apply(self, data):
        all_ceps = []
        for ch in data:
            ceps, mspec, spec = mfcc(ch)
            all_ceps.append(ceps.ravel())

        return np.array(all_ceps)


class Magnitude:
    """
    Take magnitudes of Complex data
    """
    def get_name(self):
        return "mag"

    def apply(self, data):
        return np.absolute(data)


class MagnitudeAndPhase:
    """
    Take the magnitudes and phases of complex data and append them together.
    """
    def get_name(self):
        return "magphase"

    def apply(self, data):
        magnitudes = np.absolute(data)
        phases = np.angle(data)
        return np.concatenate((magnitudes, phases), axis=1)


class Log10:
    """
    Apply Log10
    """
    def get_name(self):
        return "log10"

    def apply(self, data):
        # 10.0 * log10(re * re + im * im)
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)


class Stats:
    """
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
    """
    def get_name(self):
        return "stats"

    def apply(self, data):
        # data[ch][dim]
        shape = data.shape
        out = np.empty((shape[0], 3))
        for i in range(len(data)):
            ch_data = data[i]
            ch_data = data[i] - np.mean(ch_data)
            outi = out[i]
            outi[0] = np.std(ch_data)
            outi[1] = np.min(ch_data)
            outi[2] = np.max(ch_data)

        return out


class Resample:
    """
    Resample time-series data.
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data


class ResampleHanning:
    """
    Resample time-series data using a Hanning window
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%dhanning" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        out = resample(data, self.f, axis=axis, window=hann(M=data.shape[axis]))
        return out


class DaubWaveletStats:
    """
    Daubechies wavelet coefficients. For each block of co-efficients
    take (mean, std, min, max)
    """
    def __init__(self, n):
        self.n = n

    def get_name(self):
        return "dwtdb%dstats" % self.n

    def apply(self, data):
        # data[ch][dim0]
        shape = data.shape
        out = np.empty((shape[0], 4 * (self.n * 2 + 1)), dtype=np.float64)

        def set_stats(outi, x, offset):
            outi[offset*4] = np.mean(x)
            outi[offset*4+1] = np.std(x)
            outi[offset*4+2] = np.min(x)
            outi[offset*4+3] = np.max(x)

        for i in range(len(data)):
            outi = out[i]
            new_data = pywt.wavedec(data[i], 'db%d' % self.n, level=self.n*2)
            for i, x in enumerate(new_data):
                set_stats(outi, x, i)

        return out


class UnitScale:
    """
    Scale across the last axis.
    """
    def get_name(self):
        return 'unit-scale'

    def apply(self, data):
        return preprocessing.scale(data, axis=data.ndim-1)


class UnitScaleFeat:
    """
    Scale across the first axis, i.e. scale each feature.
    """
    def get_name(self):
        return 'unit-scale-feat'

    def apply(self, data):
        return preprocessing.scale(data, axis=0)


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply(self, data):
        return np.corrcoef(data)


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigenvalues'

    def apply(self, data):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


class OverlappingFFTDeltas:
    """
    Calculate overlapping FFT windows. The time window will be split up into num_parts,
    and parts_per_window determines how many parts form an FFT segment.

    e.g. num_parts=4 and parts_per_windows=2 indicates 3 segments
    parts = [0, 1, 2, 3]
    segment0 = parts[0:1]
    segment1 = parts[1:2]
    segment2 = parts[2:3]

    Then the features used are (segment2-segment1, segment1-segment0)

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "overlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data):
        axis = data.ndim - 1

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            combined_parts = parts[i:i+self.parts_per_window]
            if self.parts_per_window > 1:
                d = np.concatenate(combined_parts, axis=axis)
            else:
                d = combined_parts
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        diffs = []
        for i in range(1, len(partials)):
            diffs.append(partials[i] - partials[i-1])

        return np.concatenate(diffs, axis=axis)


class FFTWithOverlappingFFTDeltas:
    """
    As above but appends the whole FFT to the overlapping data.

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "fftwithoverlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data):
        axis = data.ndim - 1

        full_fft = np.fft.rfft(data, axis=axis)
        full_fft = Magnitude().apply(full_fft)
        full_fft = Log10().apply(full_fft)

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            d = np.concatenate(parts[i:i+self.parts_per_window], axis=axis)
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        out = [full_fft]
        for i in range(1, len(partials)):
            out.append(partials[i] - partials[i-1])

        return np.concatenate(out, axis=axis)


class FreqCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, start, end, scale_option='none', with_fft=False, with_corr=True, with_eigen=True,
                 window='None', subsample=1):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        self.window = window
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen
        self.subsample = subsample

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if self.window is not None:
            selections.append(self.window)
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        if self.subsample == 1:
            return 'freq-correlation-%d-%d-%s-%s%s' % (self.start, self.end, 'withfft' if self.with_fft else 'nofft',
                                                       self.scale_option, selection_str)
        else:
            return 'freq-correlation-%d-%d-%d-%s-%s%s' % (self.start, self.end, self.subsample, 'withfft' if self.with_fft else 'nofft',
                                                       self.scale_option, selection_str)

    def apply(self, data):
        data1 = FFT(self.window).apply(data)
        data1 = Slice(self.start, self.end).apply(data1)
        data1 = Magnitude().apply(data1)
        data1 = Log10().apply(data1)
        if self.subsample != 1:
            n = data1.shape[-1]
            r = []
            for i in range(0,n,self.subsample):
                r.append(data1[:,i:(i+self.subsample)].mean(axis=-1))
            r = np.vstack(r).T
            data1 = r

        data2 = data1
        if self.scale_option == 'usf':
            data2 = UnitScaleFeat().apply(data2)
        elif self.scale_option == 'us':
            data2 = UnitScale().apply(data2)

        data2 = CorrelationMatrix().apply(data2)

        if self.with_eigen:
            w = Eigenvalues().apply(data2)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)

class Bands:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, bands, scale_option='none',
                 window='None'):
        self.bands = bands
        self.scale_option = scale_option
        self.window = window
        assert scale_option in ('us', 'usf', 'none')

    def get_name(self):
        selections = []
        if self.window is not None:
            selections.append(self.window)
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        name = 'bands-%s-%s%s' % (self.scale_option, selection_str)
        name += '-' + '-b'.join(map(str,self.bands))
        return name

    def apply(self, data):
        data1 = FFT(self.window).apply(data)
        data1 = Magnitude().apply(data1)
        data1 = Band(self.bands).apply(data1)
        data1 = Log10().apply(data1)
        return data1.ravel()


class BandsCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, bands, scale_option='none', with_fft=False, with_corr=True, with_eigen=True,
                 window='None'):
        self.bands = bands
        self.scale_option = scale_option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        self.window = window
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if self.window is not None:
            selections.append(self.window)
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        name = 'bands-correlation-%s-%s%s' % ('withfft' if self.with_fft else 'nofft',
                                                   self.scale_option, selection_str)
        name += '-' + '-b'.join(map(str,self.bands))
        return name

    def apply(self, data):
        data1 = FFT(self.window).apply(data)
        data1 = Magnitude().apply(data1)
        data1 = Band(self.bands).apply(data1)
        data1 = Log10().apply(data1)

        data2 = data1
        if self.scale_option == 'usf':
            data2 = UnitScaleFeat().apply(data2)
        elif self.scale_option == 'us':
            data2 = UnitScale().apply(data2)

        data2 = CorrelationMatrix().apply(data2)

        if self.with_eigen:
            w = Eigenvalues().apply(data2)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeCorrelation:
    """
    Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.

    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, max_hz=0, scale_option='none', with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'time-correlation-r%d-%s%s' % (self.max_hz, self.scale_option, selection_str)

    def apply(self, data):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001

        data1 = data
        if self.max_hz and data1.shape[1] > self.max_hz:
            data1 = Resample(self.max_hz).apply(data1)

        if self.scale_option == 'usf':
            data1 = UnitScaleFeat().apply(data1)
        elif self.scale_option == 'us':
            data1 = UnitScale().apply(data1)

        data1 = CorrelationMatrix().apply(data1)

        if self.with_eigen:
            w = Eigenvalues().apply(data1)

        out = []
        if self.with_corr:
            data1 = upper_right_triangle(data1)
            out.append(data1)
        if self.with_eigen:
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeFreqCorrelation:
    """
    Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz=0, scale_option=''):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert scale_option in ('us', 'usf', 'none')

    def get_name(self):
        return 'time-freq-correlation-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(self.start, self.end, self.scale_option).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim-1)


class FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz=0, scale_option=''):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option

    def get_name(self):
        return 'fft-with-time-freq-corr-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        #TODO looks like there is a bug in FreqCorrelation because it did not took the samplingfrequence/window size into consideration
        data2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim-1)


class WindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one windw, results are combined using average, min and max.
    """
    def __init__(self, start, end, max_hz=0, scale_option='', nwindows=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.nwindows = nwindows

    def get_name(self):
        return 'window-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                     self.scale_option, self.nwindows)

    def apply(self, data):
        window1avg = None
        window2avg = None
        window1min = None
        window2min = None
        window1max = None
        window2max = None

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True).apply(window)

            if window1avg is None:
                window1avg = np.copy(window1)
                window1min = np.copy(window1)
                window1max = np.copy(window1)
                window2avg = np.copy(window2)
                window2min = np.copy(window2)
                window2max = np.copy(window2)
            else:
                window1avg += window1
                window1min = np.minimum(window1min,window1)
                window1max = np.maximum(window1max,window1)
                window2avg += window2
                window2min = np.minimum(window2min,window2)
                window2max = np.maximum(window2max,window2)

        window1avg /= self.nwindows
        window2avg /= self.nwindows

        if self.nwindows > 1:
            return np.concatenate((window1avg, window1min, window1max, window2avg, window2min, window2max), axis=-1)
        else:
            return np.concatenate((window1avg, window2avg), axis=-1)

class StdWindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one windw, results are combined using average and std.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.nwindows = nwindows

    def get_name(self):
        return 'stdwindow-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                     self.scale_option, self.nwindows)

    def apply(self, data):
        window1avg = None
        window2avg = None
        window1std = None
        window2std = None

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True).apply(window)

            if window1avg is None:
                window1avg = np.copy(window1)
                window1std = window1 * window1
                window2avg = np.copy(window2)
                window2std = window2 * window2
            else:
                window1avg += window1
                window2avg += window2
                window1std += window1*window1
                window2std += window2*window2

        window1avg /= self.nwindows
        window2avg /= self.nwindows
        window1std = np.sqrt(window1std / self.nwindows - window1avg * window1avg)
        window2std = np.sqrt(window2std / self.nwindows - window2avg * window2avg)


        if self.nwindows > 1:
            return np.concatenate((window1avg, window1std, window2avg, window2std), axis=-1)
        else:
            return np.concatenate((window1avg, window2avg), axis=-1)

class MedianWindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'medianwindow-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'medianwindow-fft-with-time-freq-corr-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True, window=self.window,
                                      subsample=self.subsample).apply(window)
            windows.append(np.concatenate((window1,window2)))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

class Variance:
    def __init__(self, nwindows, percentile=None, nunits=1):
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile

    def get_name(self):
        name = 'variance-w%d' % (self.nwindows)
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            window = UnitScaleFeat().apply(window)
            windows.append(window.var(axis=1))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features


class BoxWindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    the information from the windows is summarized into a box-plot
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 1
        self.nwindows = nwindows

    def get_name(self):
        return 'boxwindow-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                     self.scale_option, self.nwindows)

    def apply(self, data):
        windows = []

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True).apply(window)
            window = np.concatenate((window1,window2))
            windows.append(window)

        windows = np.sort(np.array(windows), axis=0)

        q1 = windows[int(0.25*self.nwindows),:]
        q2 = windows[int(0.5*self.nwindows),:]
        q3 = windows[int(0.75*self.nwindows),:]

        iqr = q3 - q1
        h = q3 + 1.5*iqr
        l = q1 - 1.5*iqr

        mask_h = windows > h
        n_high_outliers = np.sum(mask_h, axis=0)

        mask_l = windows < l
        n_low_outliers = np.sum(mask_l, axis=0)

        windows[mask_h] = -np.inf
        high_whiskers = windows.max(axis=0)

        windows[mask_h | mask_l] = np.inf
        low_whiskers = windows.min(axis=0)

        return np.concatenate((n_low_outliers,low_whiskers,q1,q2,q3,high_whiskers,n_high_outliers),
                              axis=-1)

class CleanMedianWindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'cleanmedianwindow-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'cleanmedianwindow-fft-with-time-freq-corr-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        clean data from people from all harmonies of 60Hz and from DC
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """

        # we have two Fs 399 and 5000 but data_length_sec is always 600.
        data_length_sec = 600.
        nsamples = data.shape[1]
        Fs = nsamples/data_length_sec

        if Fs > 400: # clean only people data
            # optimize speed by using power of 2
            nfft = 8192
            while nfft/2 < nsamples:
                nfft *= 2

            # notch filter
            Y = np.fft.rfft(data*np.hamming(nsamples), n=nfft) # add zero padding
            # find base frequency 60Hz
            base = (60./Fs)*nfft
            # clean
            for harmony in range(1,10):
                notch = int(base*harmony)
                if notch > Y.shape[1]:
                    break
                Y[:,notch-1] = 0.
                Y[:,notch] = 0.
                Y[:,notch+1] = 0.
            # remove DC
            Y[:,0] = 0.

            data = np.fft.irfft(Y,n=nfft)
            data = data[:,:nsamples]

        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True, window=self.window,
                                      subsample=self.subsample).apply(window)
            windows.append(np.concatenate((window1,window2)))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

class CleanCorMedianWindowFFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'cleancormedianwindow-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'cleancormedianwindow-fft-with-time-freq-corr-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        clean data from people from all harmonies of 60Hz and from DC
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """

        # we have two Fs 399 and 5000 but data_length_sec is always 600.
        data_length_sec = 600.
        nsamples = data.shape[1]
        Fs = nsamples/data_length_sec

        if Fs > 400: # clean only people data
            # optimize speed by using power of 2
            nfft = 8192
            while nfft/2 < nsamples:
                nfft *= 2

            # notch filter
            Y = np.fft.rfft(data*np.hamming(nsamples), n=nfft) # add zero padding
            # find base frequency 60Hz
            base = (60./Fs)*nfft
            # clean
            for harmony in range(1,10):
                notch = int(base*harmony)
                if notch > Y.shape[1]:
                    break
                Y[:,notch-1] = 0.
                Y[:,notch] = 0.
                Y[:,notch+1] = 0.
            # remove DC
            Y[:,0] = 0.

            data = np.fft.irfft(Y,n=nfft)
            data = data[:,:nsamples]

        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            window1 = TimeCorrelation(self.max_hz, self.scale_option).apply(window)
            window2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True, window=self.window,
                                      subsample=self.subsample).apply(window)
            windows.append(np.concatenate((window1,window2)))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                windows = np.array(windows)
                sorted_windows = np.sort(windows, axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                score = windows.mean(axis=0)/windows.std(axis=0)
                var = nw/(nw-1.)*(windows[:-1,:]*windows[1:,:]).sum(axis=0)/(windows*windows).sum(axis=0)
                features = np.concatenate((features, score, var), axis=-1)
                windows = []
        return features

class MedianWindow1FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'medianwindow1-fft-with-time-freq-corr-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'medianwindow1-fft-with-time-freq-corr-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if self.max_hz and window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            window1 = TimeCorrelation().apply(window)
            window2 = FreqCorrelation(self.start, self.end, with_fft=True, window=self.window,
                                      subsample=self.subsample).apply(window)
            windows.append(np.concatenate((window1,window2)))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

class MedianWindowFFTWithTimeFreqCov:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'medianwindow-fft-with-time-freq-cov-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'medianwindow-fft-with-time-freq-cov-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            if self.max_hz and window.shape[1] > self.max_hz:
                window = Resample(self.max_hz).apply(window)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            out = []
            ##############
            # window2 = FreqCorrelation(self.start, self.end, with_fft=True, window=self.window,
            #                           subsample=self.subsample).apply(window)
            data_frq = FFT(self.window).apply(window)
            data_frq[:,59:62] = 0.
            data_frq_slice = Slice(self.start, self.end).apply(data_frq)
            data_power = Magnitude().apply(data_frq_slice)
            data_power = Log10().apply(data_power)

            data2 = CorrelationMatrix().apply(data_power)

            out.append(upper_right_triangle(data2))
            out.append(Eigenvalues().apply(data2))

            out.append(data_power.ravel())

            ##############
            # window1 = TimeCorrelation().apply(window)

            data_frq[:,:self.start] = 0.
            data_frq[:,self.end:] = 0.
            filtered_window =  np.fft.irfft(data_frq)

            data1 = CorrelationMatrix().apply(filtered_window)

            out.append(upper_right_triangle(data1))

            out.append(Eigenvalues().apply(data1))

            ##############

            out = np.concatenate(out, axis=0)
            windows.append(out)

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

class MedianWindowFFTWithTimeFreqCov2:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, max_hz, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'medianwindow-fft-with-time-freq-cov2-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.max_hz,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'medianwindow-fft-with-time-freq-cov2-%d-%d-%d-r%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                                 self.max_hz,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)
            resample = self.max_hz and window.shape[1] > self.max_hz
            if resample:
                window = Resample(self.max_hz).apply(window)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            out = []
            ##############
            # window2 = FreqCorrelation(self.start, self.end, with_fft=True, window=self.window,
            #                           subsample=self.subsample).apply(window)
            data_frq = FFT(self.window).apply(window)
            data_power = Log10().apply(Magnitude().apply(Slice(self.start, self.end).apply(data_frq)))

            data2 = CorrelationMatrix().apply(data_power)

            out.append(upper_right_triangle(data2))
            out.append(Eigenvalues().apply(data2))

            out.append(data_power.ravel())

            ##############
            # window1 = TimeCorrelation().apply(window)
            if resample:
                data_frq[:,60] = 0.

            # data_frq[:,:self.start] = 0.
            # data_frq[:,self.end:] = 0.
            filtered_window =  np.fft.irfft(data_frq, window.shape[1])

            data1 = CorrelationMatrix().apply(filtered_window)

            out.append(upper_right_triangle(data1))

            out.append(Eigenvalues().apply(data1))

            ##############

            out = np.concatenate(out, axis=0)
            windows.append(out)

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

"""
partitioned into non-overlapping 1-minute blocks, each block Fourier transformed, and
the resulting power spectrum (0.1-200 Hz) is divided into 6 frequency bands:
delta 0.1-4, theta 4-8, alpha 8-12, beta 12-30, low-gamma 30-70, and high-gamma 70-180.
Within each frequency band the power was summed over band frequencies to produce a power-in-band (PIB) feature.
These features were aggregated into a feature vector containing 96 PIB values (16 channels * 6 bands)
"""
class MedianWindow2FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, start, end, scale_option, nwindows, percentile=None, nunits=1, window=None,subsample=1):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.nunits = nunits # windows are grouped into units
        self.percentile = percentile
        self.window = window
        self.subsample = subsample

    def get_name(self):
        if self.subsample == 1:
            name = 'medianwindow2-fft-with-time-freq-corr-%d-%d-%s-w%d' % (self.start, self.end,
                                                                             self.scale_option, self.nwindows)
        else:
            name = 'medianwindow2-fft-with-time-freq-corr-%d-%d-%d-%s-w%d' % (self.start, self.end, self.subsample,
                                                                             self.scale_option, self.nwindows)
        if self.window is not None:
            name += '-' + self.window
        if self.nunits != 1:
            name += '-u%d'%self.nunits
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        split samples to nwindows
        Downsample them to max_hz (400) samples
        generate Time/Freq Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        if data.shape[1] > 5*60*5000:
            data = resample(data, 239766, axis=-1)
        windows = []
        unit_skip = self.nwindows / self.nunits
        features = None
        percentile = [0.1,0.5,0.9] if self.percentile is None else self.percentile

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            window1 = TimeCorrelation().apply(window)
            window2 = FreqCorrelation(self.start, self.end, with_fft=True, window=self.window,
                                      subsample=self.subsample).apply(window)
            windows.append(np.concatenate((window1,window2)))

            nw = len(windows)
            if nw >= unit_skip or i == self.nwindows-1:
                sorted_windows = np.sort(np.array(windows), axis=0)
                unit_features = np.concatenate([sorted_windows[int(p*nw),:] for p in percentile], axis=-1)
                if features is None:
                    features = unit_features
                else:
                    features = np.concatenate((features, unit_features), axis=-1)
                windows = []

        return features

"""
partitioned into non-overlapping 1-minute blocks, each block Fourier transformed, and
the resulting power spectrum (0.1-200 Hz) is divided into 6 frequency bands:
delta 0.1-4, theta 4-8, alpha 8-12, beta 12-30, low-gamma 30-70, and high-gamma 70-180.
Within each frequency band the power was summed over band frequencies to produce a power-in-band (PIB) feature.
These features were aggregated into a feature vector containing 96 PIB values (16 channels * 6 bands)
"""
class MedianWindowBands:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, scale_option, nwindows, percentile=[0.1,0.5,0.9], bands=[0.2,4,8,12,30,70]):
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.percentile = percentile
        self.bands = bands

    def get_name(self):
        name = 'medianwindow-bands-%s-w%d' % (self.scale_option, self.nwindows)
        name += '-b' + '-b'.join(map(str,self.bands))
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        Downsample to 400Hz, and notch 60Hz
        split samples to nwindows
        generate Band Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        if data.shape[1] > 5*60*5000:
            def mynotch(fftfreq, notchfreq=60., notchwidth=1., Fs=5000.):
                return np.double(np.abs(np.abs(fftfreq) - notchfreq/Fs) > (notchwidth/2.)/Fs)
            data = resample(data, 239766, axis=-1, window=mynotch)
        windows = []

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            window1 = Bands(self.bands).apply(window)
            windows.append(window1)

        sorted_windows = np.sort(np.array(windows), axis=0)
        features = np.concatenate([sorted_windows[int(p*self.nwindows),:] for p in self.percentile], axis=-1)

        return features

class MedianWindowBandsCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    The above is performed on windows which is resmapled to max_hz
    if there is more than one window, the median (50% percentile), 10% perecentile and 90% perecentile are taken.
    """
    def __init__(self, scale_option, nwindows, percentile=[0.1,0.5,0.9], bands=[0.2,4,8,12,30,70]):
        self.scale_option = scale_option
        assert nwindows > 0
        self.nwindows = nwindows # data is divided into windows
        self.percentile = percentile
        self.bands = bands

    def get_name(self):
        name = 'medianwindow-bandscorr-%s-w%d' % (self.scale_option, self.nwindows)
        name += '-b' + '-b'.join(map(str,self.bands))
        if self.percentile is not None:
            name += '-' + '-'.join(map(str,self.percentile))
        return name

    def apply(self, data):
        """data[channels,samples]
        Downsample to 400Hz, and notch 60Hz
        split samples to nwindows
        generate Band Correlation features from each window.
        for each feature find the pecentile values (e.g. 10%,50%,90%) over all windows
        """
        if data.shape[1] > 5*60*5000:
            def mynotch(fftfreq, notchfreq=60., notchwidth=1., Fs=5000.):
                return np.double(np.abs(np.abs(fftfreq) - notchfreq/Fs) > (notchwidth/2.)/Fs)
            data = resample(data, 239766, axis=-1, window=mynotch)
        windows = []

        istartend = np.linspace(0.,data.shape[1],self.nwindows+1)
        for i in range(self.nwindows):
            window = data[:,int(istartend[i]):int(istartend[i+1])].astype(float)

            if self.scale_option == 'usf':
                window = UnitScaleFeat().apply(window)
            elif self.scale_option == 'us':
                window = UnitScale().apply(window)

            window1 = BandsCorrelation(self.bands,with_fft=True).apply(window)
            windows.append(window1)

        sorted_windows = np.sort(np.array(windows), axis=0)
        features = np.concatenate([sorted_windows[int(p*self.nwindows),:] for p in self.percentile], axis=-1)

        return features

