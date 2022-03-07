import numpy as np
import torch
import math


def spectralplot(x_mat,frameshift,framelength, Fre):
    fn = x_mat.shape[0]
    frameTime1 = np.zeros(fn)
    for i in range(fn):
        frameTime1[i] = ((i+1-1)*frameshift + framelength/2)/Fre
    frameTime1 = np.reshape(frameTime1,[fn,1]).T
    return frameTime1

def lenwin(wlen):
    w2 = math.ceil(wlen/2+1)
    nn3 = [n2 for n2 in range(1, w2+1)]
    nn31 = np.array(nn3)
    #nn31 = np.reshape(nn31, [w2, 1]).T
    return nn31

def compfreq(n2,Fs,framelength):
    num = n2.shape[0]
    freq =np.zeros(num)
    for ii in range(num):
        freq[ii] = (n2[ii] - 1) * Fs/framelength
    freq = np.reshape(freq, [num, 1]).T
    return freq