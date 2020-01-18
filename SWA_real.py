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


# sigbufs = np.zeros((n, f.getNSamples()[0]))
#for i in np.arange(n):
#     sigbufs[i, :] = f.readSignal(i)
#
# #ts= int(233 * Fs)
# #te= int((233+5) * Fs)
# #sigbufs = sigbufs[:,ts:te]
# t = np.arange(sigbufs.shape[1])/Fs/60
#
# L = sigbufs[4, :]

#['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T7', 'EEG T8', 'EEG P7', 'EEG P8', 'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG Oz', 'EEG FC1', 'EEG FC2', 'EEG CP1', 'EEG CP2', 'EEG FC5', 'EEG FC6', 'EEG CP5', 'EEG CP6', 'EEG TP9', 'EEG TP10', 'EEG POz', 'ECG ECG', 'ECG ECG2', 'EOG HEOG', 'EOG VEOG']
i=0
EEG= file.readSignal(i)
# L = L[Fs*40*60:Fs*60*60]
t = np.arange(len(EEG))/Fs
#M = (L - R) /2
window = int(Fs * 4)
overlap = int(Fs * 3)

#f, t2, Sxx = signal.spectrogram(L, Fs, nperseg= k, nfft= k, noverlap=int(Fs*0.5), scaling  ="density",mode ='psd')
# f, t2, Sxx = signal.spectrogram(EEG, Fs, nperseg= window, nfft= window, noverlap=overlap, scaling  ="spectrum",mode ='psd')

f, t2, Sxx = signal.spectrogram(EEG, Fs, nperseg= window, nfft= window, noverlap=overlap, scaling  ="spectrum",mode ='magnitude')
Sxx_o =  copy.deepcopy(Sxx)
Sxx   = 10*np.log10(Sxx)
Sxx[Sxx>16]=16
Sxx[Sxx<-19]=-19
# Sxx = ndimage.filters.gaussian_filter(Sxx,1)
fmax = 50
findex =np.where(f>fmax)[0][0]

#Sxx = Sxx/ np.mean(Sxx,axis=0)


ALPHA_f =[np.where(f>8)[0][0],np.where(f>12)[0][0]]
SALPHA = Sxx_o[ALPHA_f[0]:ALPHA_f[1],:]
YALPHA = np.sum(SALPHA,axis=0)


DELTA_f =[np.where(f>0.5)[0][0],np.where(f>4)[0][0]]
SDELTA = Sxx_o[DELTA_f[0]:DELTA_f[1],:]
YDELTA = np.sum(SDELTA,axis=0)


SWO_f =[np.where(f>0.25)[0][0],np.where(f>1.5)[0][0]]
SSWO = Sxx_o[SWO_f[0]:SWO_f[1],:]
YSWO = np.sum(SSWO,axis=0)


THETA_f =[np.where(f>4)[0][0],np.where(f>8)[0][0]]
STHETA = Sxx_o[THETA_f[0]:THETA_f[1],:]
YTHETA = np.sum(STHETA,axis=0)


BETA_f =[np.where(f>12)[0][0],np.where(f>30)[0][0]]
SBETA = Sxx_o[BETA_f[0]:BETA_f[1],:]
YBETA = np.sum(SBETA,axis=0)


Filter_list = [['Moving average', '', '', '', 1.0, '', '', '', '']]
for Filter_info in Filter_list:
    YSWO_filter = signalfilterbandpass({'YSWO':YSWO}, Fs, Filter_info)['YSWO']

plt.figure()
plt.subplot(3,1,1)
plt.title(signal_labels[i])
#plt.title('Test_2 : 14fev2019')
plt.plot(t,EEG)
plt.subplot(3,1,2)
#plt.pcolormesh(t2, f[:findex], np.log10(Sxx[:findex,:])** 0.9,cmap='hot')
# plt.pcolormesh(t2, f[:findex],  Sxx[:findex,:] ** 1, cmap='jet')
plt.pcolormesh(t2, f[:findex], Sxx[:findex,:],cmap='jet')
# plt.pcolormesh(t2, f[SWO_f[0]:SWO_f[1]], Sxx_o[SWO_f[0]:SWO_f[1],:],cmap='jet')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.subplot(3,1,3)
# plt.plot(t2,YALPHA,label='YALPHA')
# plt.plot(t2,YDELTA,label='YDELTA')
plt.plot(t2,YSWO,label='YSWO')
plt.plot(t2,YSWO_filter,label='YSWO_filter',color='r')
# plt.plot(t2,YTHETA,label='YTHETA')
# plt.plot(t2,YBETA,label='YBETA')
plt.legend()
plt.show()