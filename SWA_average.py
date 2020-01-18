import numpy as np
import pyedflib
import matplotlib.pylab as plt
from scipy import signal
from scipy import ndimage
#import scipy.io
#f = scipy.io.loadmat('Subject1.mat')
import copy

def iir_band_filter(ite_data, fs, btype, ftype, order=None, Quality=None , window=None, lowcut=None, highcut=None, zerophase=None, rps=None):
    fe = fs / 2.0
    if not lowcut == ''  and not highcut == '':
        wn = [lowcut / fe, highcut / fe]
    elif not lowcut == '':
        wn = lowcut / fe
    elif not highcut == '':
        wn = highcut / fe

    if rps[0]=='':
        rps[0] =10
    if rps[1]=='':
        rps[1] =10

    if btype in ["butter","bessel","cheby1","cheby2","ellip"] :
        z, p, k = signal.iirfilter(order, wn, btype=ftype, ftype=btype, output="zpk", rp=rps[0], rs=rps[1])
        try:
            sos = signal.zpk2sos(z, p, k)
            ite_data = signal.sosfilt(sos, ite_data)
            if zerophase:
                ite_data = signal.sosfilt(sos, ite_data[::-1])[::-1]
        except:
            print('A filter had an issue: ','btype', btype,'ftype', ftype,'order', order ,'Quality', Quality , 'window',window ,'lowcut', lowcut ,'highcut', highcut  , 'rps',rps  )
    elif btype == "iirnotch":
        b, a = signal.iirnotch(lowcut, Quality, fs)
        y = signal.filtfilt(b, a, ite_data)
    elif btype == "Moving average":
        z2 = np.cumsum(np.pad(ite_data, ((window, 0) ), 'constant', constant_values=0), axis=0)
        z1 = np.cumsum(np.pad(ite_data, ((0, window) ), 'constant', constant_values=ite_data[-1]), axis=0)
        ite_data= (z1 - z2)[(window - 1):-1] / window
    return ite_data

def signalfilterbandpass(Sigs_dict,fs,Filter_info):
    N = len(Sigs_dict[list(Sigs_dict.keys())[0]])
    btype = Filter_info[0]
    ftype = Filter_info[1]
    order = Filter_info[2]
    Quality = Filter_info[3]
    window= Filter_info[4]
    lowcut= Filter_info[5]
    highcut= Filter_info[6]
    rps = [Filter_info[7],Filter_info[8]]

    if not order == '':
        if order >10:
            order=10
        if order <0:
            order=0
    if not lowcut == '':
        if lowcut <=0:
            lowcut=1/fs
    if not highcut == '':
        if highcut >= fs/2:
            highcut = fs/2-1

    if not window == '':
        window = int(window*fs)
        if window <= 0:
            window = 1
        elif window > N:
            window = N

    for idx_lfp, key in enumerate(list(Sigs_dict.keys())):
        Sigs_dict[key] = iir_band_filter(Sigs_dict[key], fs, btype, ftype, order=order, Quality=Quality , window=window, lowcut=lowcut, highcut=highcut, zerophase=0, rps=rps)
    return Sigs_dict


Fs =500
file = pyedflib.EdfReader('band0.5-30hz.edf')

n = file.signals_in_file
signal_labels = file.getSignalLabels()


sigbufs = np.zeros((n, file.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = file.readSignal(i)

#ts= int(233 * Fs)
#te= int((233+5) * Fs)
#sigbufs = sigbufs[:,ts:te]
t = np.arange(sigbufs.shape[1])/Fs

window = int(Fs * 4)
overlap = int(Fs * 3)
YSWO_list = []
for i,EEG in enumerate(sigbufs):
    print(i,signal_labels[i])
    f, t2, Sxx = signal.spectrogram(EEG, Fs, nperseg= window, nfft= window, noverlap=overlap, scaling  ="spectrum",mode ='magnitude')
    Sxx_o = copy.deepcopy(Sxx)

    fmax = 50
    findex =np.where(f>fmax)[0][0]

    SWO_f =[np.where(f>0.25)[0][0],np.where(f>1.5)[0][0]]
    SSWO = Sxx_o[SWO_f[0]:SWO_f[1],:]
    YSWO = np.median(SSWO,axis=0)
    YSWO_list.append(YSWO)

YSWO_mean = np.mean(np.array(YSWO_list),axis=0)
plt.figure()
plt.subplot(1,1,1)
for swa in YSWO_list:
    plt.plot(t2/60,swa,'k' )
plt.plot(t2/60, YSWO_mean, 'r')
plt.legend()
plt.show()