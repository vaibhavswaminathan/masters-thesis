import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state)
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

# random_state = 43
# n_timestamps = 192
# min_window_size = 1
# n_windows = 5

# rng = check_random_state(random_state)
# start = rng.randint(0, n_timestamps - min_window_size, size=n_windows)
# end = rng.randint(start + min_window_size, n_timestamps + 1,
#                     size=n_windows)
# print(start + min_window_size)
# # print(end)
# indices = np.c_[start, end]

# print(indices)

### Time-lagged Cross-correlation

data = pd.read_csv(r'/home/vaibhavs/Master_Thesis/ma-vaibhav/Data/CO_valve_15m.csv')
data_filled = data.dropna()
# setpoint = data_filled['ADS.fAHUHRBypValveSetADSInternalValuesMirror'].to_numpy()
# actual = data_filled['ADS.fAHUHRBypValveAct1ADSInternalValuesMirror'].to_numpy()

no_windows = 30
samples_per_window = data_filled.shape[0]/no_windows

### compute Cross-Correlation between two time series
# def cross_correlation_using_fft(x, y):
#     f1 = fft(x)
#     f2 = fft(np.flipud(y))
#     cc = np.real(ifft(f1 * f2))
#     return fftshift(cc)

# # shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
# def compute_shift(x, y):
#     assert len(x) == len(y)
#     c = cross_correlation_using_fft(x, y)
#     assert len(c) == len(x)
#     zero_index = int(len(x) / 2) - 1
#     shift = zero_index - np.argmax(c)
#     return shift

# for t in range(0, no_windows):
#     setpoint = data_filled['ADS.fAHUCOValveSetADSInternalValuesMirror'].loc[(t)*samples_per_window:(t+1)*samples_per_window]
#     actual = data_filled['ADS.fAHUCOValveActADSInternalValuesMirror'].loc[(t)*samples_per_window:(t+1)*samples_per_window]
#     shift = compute_shift(actual, setpoint)
#     print("Window: %d, Shift: %d" %(t,shift))


### other implementation
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

setpoint_data = data_filled['ADS.fAHUCOValveSetADSInternalValuesMirror']
actual_data = data_filled['ADS.fAHUCOValveActADSInternalValuesMirror']
seconds = 5
fps = 30
# rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
rs = [crosscorr(setpoint_data,actual_data,lag) for lag in range(-4,4)]
offset = np.ceil(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nSetpoint leads <> Actual leads',ylim=[0.0,1.0],xlim=[0,8], xlabel='Offset',ylabel='Pearson r')
ax.set_xticklabels([int(item-4) for item in ax.get_xticks()])
plt.legend()
plt.show()


### Windowed time lagged cross correlation
# seconds = 5
# fps = 20
# no_windows = 30
# samples_per_window = data_filled.shape[0]/no_windows
# rss=[]
# for t in range(0, no_windows):
#     d1 = data_filled['ADS.fAHUCOValveSetADSInternalValuesMirror'].loc[(t)*samples_per_window:(t+1)*samples_per_window]
#     d2 = data_filled['ADS.fAHUCOValveActADSInternalValuesMirror'].loc[(t)*samples_per_window:(t+1)*samples_per_window]
#     # rs = [crosscorr(d1,d2,lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
#     rs = [crosscorr(d1,d2,lag) for lag in range(-4,4)]
#     rss.append(rs)
# rss = pd.DataFrame(rss)
# f,ax = plt.subplots(figsize=(10,5))
# sns.heatmap(rss,cmap='RdBu_r',ax=ax)
# ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,9], xlabel='Offset',ylabel='Window epochs')
# ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
# ax.set_xticklabels([-4, -3, -2, -1, 0, 1, 2, 3, 4]);
# plt.show()

#---------------- StellarGraph (load dataset) -----------------#

import stellargraph as sg
from stellargraph import datasets