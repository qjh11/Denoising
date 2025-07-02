import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from scipy.signal import butter, lfilter, lfilter_zi
from scipy import ndimage, misc


# MATLAB's tic and toc functions in Python
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, zo = lfilter(b, a, data, zi=zi * data[0])
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Reject outliers function
def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 1.
    return data[s < m]


# Custom loss function in PyTorch
def MASK_LOSS(y_true, y_pred):
    nois = y_true - y_pred
    maskbiS = 1 / (1 + torch.abs(nois) / torch.abs(y_pred))
    maskbiN = (torch.abs(nois) / torch.abs(y_pred)) / (1 + torch.abs(nois) / torch.abs(y_pred))
    return torch.abs(1 - maskbiS) + torch.abs(maskbiN)


# Attention layer in PyTorch
class Attention(nn.Module):
    def __init__(self, input_channels):
        super(Attention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Equivalent to GlobalAvgPool1D
        self.dense1 = nn.Linear(input_channels, input_channels)
        self.dense2 = nn.Linear(input_channels, input_channels)

    def forward(self, inp1):
        # Global Average Pooling
        inp2 = self.global_avg_pool(inp1.transpose(1, 2))  # Shape adjustment for PyTorch
        inp2 = inp2.squeeze(-1)

        # Reshape and apply Dense layers (Linear in PyTorch)
        x = F.relu(self.dense1(inp2))
        x = torch.sigmoid(self.dense2(x))

        # Reshape x to match inp1 dimensions and apply element-wise multiplication
        x = x.unsqueeze(-1).expand_as(inp1)  # Broadcasting in PyTorch
        out = inp1 * x

        # Add original input (Skip connection)
        out = out + inp1

        return out


# Block layer in PyTorch
class Block(nn.Module):
    def __init__(self, input_dim, D):
        super(Block, self).__init__()
        self.fc1 = nn.Linear(input_dim, D)
        self.fc2 = nn.Linear(D, D)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))

        # Reshaping: From (batch_size, D) to (batch_size, D, 1)
        x = x.unsqueeze(-1)

        return x


# Compact layer in PyTorch
class CompactLayer(nn.Module):
    def __init__(self, input_dim, D):
        super(CompactLayer, self).__init__()
        self.block1 = Block(input_dim, D)
        self.block2 = Block(input_dim, D)
        self.attention = Attention(D * 2)  # Attention expects concatenated size

    def forward(self, y):
        # Equivalent to Lambda slicing in Keras
        s0 = y[:, :, 0]
        s1 = y[:, :, 1]

        # Apply the Block function on each slice
        B1 = self.block1(s0)
        B2 = self.block2(s1)

        # Concatenate along the last dimension
        B = torch.cat((B1, B2), dim=-1)

        # Apply attention mechanism
        Batt = self.attention(B)

        return Batt


# yc_patch function for patch extraction
def yc_patch(A, l1, l2, o1, o2):
    n1, n2 = np.shape(A)
    tmp = np.mod(n1 - l1, o1)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((o1 - tmp, n2))], axis=0)

    tmp = np.mod(n2 - l2, o2)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((A.shape[0], o2 - tmp))], axis=-1)

    N1, N2 = np.shape(A)
    X = []
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            tmp = np.reshape(A[i1:i1 + l1, i2:i2 + l2], (l1 * l2, 1))
            X.append(tmp)
    X = np.array(X)
    return X[:, :, 0]


# yc_snr function to calculate SNR
def yc_snr(g, f):
    psnr = 20. * np.log10(np.linalg.norm(g) / np.linalg.norm(g - f))
    return psnr


# yc_patch_inv function for patch inversion
import torch
# yc_patch_inv 函数定义（保持不变）
def yc_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)

    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    elif (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    elif (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    elif (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = A.shape
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            A[i1:i1 + l1, i2:i2 + l2] += np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] += np.ones((l1, l2))
            ids += 1

    A = A/mask
    A = A[0:n1, 0:n2]

    return A