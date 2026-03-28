from load_data import *
from augmentations import *
import numpy as np
from scipy.signal import coherence
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import welch, csd
from joblib import Parallel, delayed
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import coherence
from load_data import *
from augmentations import *
import numpy as np
from scipy.signal import coherence, welch
from joblib import Parallel, delayed
import time

fs = 0.5    
nperseg = 64 
noverlap = nperseg // 2


def dataset_time(X, Y, pert=False):
    config = Config_augmentation()  
    X_featgraph, X_adjgraph = [], []
    for i in range(len(Y)):
        signals = X[i]
        if pert == True: 
            signals = DataTransform_TD_bank(signals, config )
        window_data = np.corrcoef(signals)
        knn_graph = compute_KNN_graph(window_data)

        X_featgraph.append(window_data)
        X_adjgraph.append(knn_graph)
        
    return X_featgraph, X_adjgraph, Y

def compute_psd(time_series_data, fs, nperseg):
    nperseg = min(len(time_series_data), nperseg)
    if nperseg < len(time_series_data) // 2:
        nperseg = len(time_series_data) // 2 
    freqs, psd = welch(time_series_data, fs=fs, nperseg=nperseg)
    return freqs, psd
  
def compute_fft_segments(time_series_data, fs, nperseg, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    N = len(time_series_data)

    if N < nperseg:
        nperseg = N
        noverlap = nperseg // 2
        step = max(1, nperseg - noverlap)

    n_segments = 1 + (N - nperseg) // step
    if n_segments <= 0:
        n_segments = 1
        segments = time_series_data[:nperseg][None, :]
    else:
        segments = np.array([
            time_series_data[k * step : k * step + nperseg]
            for k in range(n_segments)
        ])

    window = np.hanning(nperseg)
    segments = segments * window[None, :]

    fft_data = np.fft.rfft(segments, axis=1)
    freqs = np.fft.rfftfreq(nperseg, d=1/fs)

    return freqs, fft_data

def compute_coherence_matrix_FFT(fft_data, freqs=None, fmin=0.01, fmax=0.1, eps=1e-12):

    N, S, F = fft_data.shape
    coherence_matrix = np.eye(N)

    if freqs is not None and fmin is not None and fmax is not None:
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        freq_mask = np.ones(F, dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            Xi = fft_data[i]  
            Xj = fft_data[j]

            Sxx = np.mean(np.abs(Xi) ** 2, axis=0)
            Syy = np.mean(np.abs(Xj) ** 2, axis=0)
            Sxy = np.mean(Xi * np.conj(Xj), axis=0)

            Cxy_f = np.abs(Sxy) ** 2 / (Sxx * Syy + eps)
            coherence_value = np.mean(Cxy_f[freq_mask])

            coherence_matrix[i, j] = coherence_value
            coherence_matrix[j, i] = coherence_value

    coherence_matrix[coherence_matrix < 0.3] = 0
    return coherence_matrix  


def dataset_freq(X, Y, fs, nperseg, pert=False, noverlap=None, fmin=0.01, fmax=0.1):
    config = Config_augmentation()
    X_featgraph, X_adjgraph = [], []
    total_start_time = time.time()

    if noverlap is None:
        noverlap = nperseg // 2
    for i in range(len(Y)):
        signals = X[i]  
        if isinstance(signals, torch.Tensor):
            signals = signals.cpu().detach().numpy()
        fft_list = []
        freqs = None
        for signal in signals:
            freqs, fft_segments = compute_fft_segments(signal, fs, nperseg, noverlap)
            fft_list.append(fft_segments)
        fft_data = np.array(fft_list)  
        if pert == True:
            aug_fft = []
            for ch in range(fft_data.shape[0]):
                ch_fft = DataTransform_FD(fft_data[ch], config)  
                if isinstance(ch_fft, torch.Tensor):
                    ch_fft = ch_fft.cpu().detach().numpy()
                aug_fft.append(ch_fft)
            fft_data = np.array(aug_fft)
        coherence_matrix = compute_coherence_matrix_FFT(
            fft_data, freqs=freqs, fmin=fmin, fmax=fmax
        )
        psd_data = np.array([
            compute_psd(signal, fs=fs, nperseg=nperseg)[1]
            for signal in signals
        ])

        X_featgraph.append(psd_data)
        X_adjgraph.append(coherence_matrix)

    total_elapsed_time = time.time() - total_start_time
    return X_featgraph, X_adjgraph, Y  