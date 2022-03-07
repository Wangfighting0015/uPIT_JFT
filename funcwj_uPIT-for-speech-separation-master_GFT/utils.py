#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import warnings
import yaml
import logging

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import torch
import torch.nn as nn

import gsp as gspfunction
import torchaudio
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import spectr
import math

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

config_keys = [
    "trainer", "model", "spectrogram_reader", "dataloader", "train_scp_conf",
    "valid_scp_conf", "debug_scp_conf"
]


def nfft(window_size):
    return int(2**np.ceil(int(np.log2(window_size))))


def stftdefine(samps,frame_length, frame_shift, center,window, return_samps,apply_abs,apply_log,apply_pow,transpose):
    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat if not return_samps else (samps, stft_mat)


def stft_GFT(file,
         frame_length=1024,
         frame_shift=256,
         center=False,
         window="hann",
         return_samps=False,
         apply_abs=False,
         apply_log=False,
         apply_pow=False,
         transpose=True):
    if not os.path.exists(file):
        raise FileNotFoundError("Input file {} do not exists!".format(file))
    if apply_log and not apply_abs:
        apply_abs = True
        warnings.warn(
            "Ignore apply_abs=False cause function return real values")
    samps, Fs = audio_lib.load(file, sr=None)

    STFT_mat=stftdefine(samps, frame_length, frame_shift, center,window, return_samps,apply_abs,apply_log,apply_pow,transpose)
    GFT_mat = gft_fun(samps)

    #STFT_mat = torch.from_numpy(STFT_mat)
    STFT_mat = torch.Tensor(STFT_mat)
   # print(STFT_mat.shape)
    print(GFT_mat.shape)
    stft_gft_mat = torch.cat((STFT_mat, GFT_mat), 1)
    print(stft_gft_mat.shape)
    '''inputnum = gft_stft.shape[1]
    outputnum = STFT_mat.shape[1]
    gft_stft = torch.unsqueeze(gft_stft, 2)

    gft_stft_out = nn.Conv1d(inputnum, outputnum, 1, bias=False)
    stft_mat1 = gft_stft_out(gft_stft)
    stft_mat2 = torch.squeeze(stft_mat1, 2)
    stft_gft_mat = stft_mat2.detach().numpy()'''
    return stft_gft_mat if not return_samps else (samps, stft_gft_mat)


def gft_fun(samps):
    frame_length = 256
    frame_shift = 128
    center = False
    window = "hann"
    #samps, Fs = audio_lib.load(file, sr=None)
    gft_mat = enframe_gft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    gft_mat = gft_mat.astype(np.float32)
    gft_mat = torch.from_numpy(gft_mat)
    gft_mat = 100 * np.abs(((gft_mat)))
    return gft_mat


# return F x T or T x F
def stft(file,
         frame_length=1024,
         frame_shift=256,
         center=False,
         window="hann",
         return_samps=False,
         apply_abs=False,
         apply_log=False,
         apply_pow=False,
         transpose=True):
    if not os.path.exists(file):
        raise FileNotFoundError("Input file {} do not exists!".format(file))
    if apply_log and not apply_abs:
        apply_abs = True
        warnings.warn(
            "Ignore apply_abs=False cause function return real values")
    samps, Fs = audio_lib.load(file, sr=None)

    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)

    gft_mat = enframe_gft(
            samps,
            nfft(frame_length),
            frame_shift,
            frame_length,
            window=window,
            center=center)

    gft_mat = gft_mat.astype(np.float32)
    gft_mat = torch.from_numpy(gft_mat)
    stft_mat = torch.from_numpy(stft_mat)
    stft_gft_mat = torch.cat((stft_mat, gft_mat), 1)
    #print(stft_gft_mat.dtype)
    #print(stft_mat.shape)
    #print(gft_mat.shape)
    #print(stft_gft_mat.shape)
    return stft_gft_mat if not return_samps else (samps, stft_gft_mat)

    #plot spectrum of signals with STFT
    #FrequencyScale = [i * Fs / frame_length for i in range(frame_length+1 // 2)]  # 频率刻度
    #frameTime = FrameTimeC(stft_mat.shape[0], frame_length, frame_shift, Fs)  # 每帧对应的时间
    #LogarithmicSpectrogramData = 10 * np.log10((np.abs(y) * np.abs(y)))  # 取对数后的数据

    #plt.pcolormesh(frameTime, FrequencyScale, stft_mat)
    #plt.colorbar()
    # plt.savefig('语谱图22.png')
    #plt.show()

    ''''frameTime = spectal.spectralplot(stft_mat, frame_shift,frame_length,Fs)
    print(frameTime)
    n2 = spectal.lenwin(frame_length)
    freq = spectal.compfreq(n2, Fs, frame_length)
    print(freq)
    ww2 = math.ceil(frame_length/2 + 1)
    nn2 = np.reshape(n2, [ww2, 1]).T
    print(nn2)
    plt.figure()
    plt.pcolormesh(frameTime,freq,stft_mat)
    #plt.plot(frameTime,stft_mat[nn2,:])
    #plt.plot(stft_mat)
    plt.show()'''

    '''gft_mat = enframe1(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)

    gft_mat = gft_mat.astype(np.float32)

    FrequencyScale = [i * Fs / frame_length for i in range((frame_length//2)+1)]  # 频率刻度
    frameTime = FrameTimeC(stft_mat.shape[0], frame_length, frame_shift, Fs)  # 每帧对应的时间
    # LogarithmicSpectrogramData = 10 * np.log10((np.abs(y) * np.abs(y)))  # 取对数后的数据
    plt.figure()
    plt.pcolormesh(frameTime, FrequencyScale, stft_mat.T)
    plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Traditinal Spectrum')
    plt.show

    gft_mat = torch.from_numpy(gft_mat)
    gft_mat = 1 * np.abs(((gft_mat)))
    #gft_mat = 1*np.abs((torch.fliplr(gft_mat)))
    samlesnumss = gft_mat.shape[1]
    _, graphfreq = gspfunction.graph_frouier_basisA3(samlesnumss)
    plt.figure()
    plt.pcolormesh(frameTime, graphfreq, gft_mat.T)
    plt.colorbar()
    plt.ylabel('Graph Frequency')
    plt.xlabel('Time(s)')
    plt.title('Graph Spectrum')
    plt.show()

    stft_mat = torch.from_numpy(stft_mat)
    gft_stft = torch.cat((stft_mat, gft_mat), 1)
    inputnum = gft_stft.shape[1]
    outputnum = stft_mat.shape[1]
    gft_stft = torch.unsqueeze(gft_stft, 2)
    # print(gft_stft.shape)
    gft_stft_out = nn.Conv1d(inputnum, outputnum, 1, bias=False)
    stft_mat1 = gft_stft_out(gft_stft)
    stft_mat2 = torch.squeeze(stft_mat1, 2)
    stft_mat3 = stft_mat2.detach().numpy()
    #plt.figure()
    #plt.pcolormesh(frameTime, FrequencyScale, stft_mat3.T)
    #plt.colorbar()
    #plt.ylabel('STFT_GFT_Frequency')
    #plt.xlabel('Time(s)')
    #plt.title('STFT_GFT Spectrum')
    #plt.show()
    # print(stft_mat3.dtype)

    ###print(stft_mat1.dtype)
    # print(gft_mat.dtype)
    # print(gft_stft.dtype)

    # print(gft_mat.shape)
    # print(stft_mat2.shape)
    # print(gft_stft.shape)

    return stft_mat3 if not return_samps else (samps, stft_mat3)'''



def enframe_gft(yy, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window =audio_lib.filters.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = audio_lib.util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    audio_lib.util.valid_audio(yy)

    # Pad the time series so that frames are centered
    if center:
        yy = np.pad(yy, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    yy_frames = audio_lib.util.frame(yy, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    sampsnum = yy_frames.shape[0]
    framenu = yy_frames.shape[1]
    U, _ = gspfunction.graph_frouier_basisA3(sampsnum)
    GFT_samps = np.zeros(shape=(sampsnum, framenu))
    for i in range(framenu):
    # print((U).shape)
        GFT_samps[:, i] = np.dot(U, (yy_frames[:, i]))
    GFT_samps = np.transpose(GFT_samps)
    return GFT_samps

def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

def istft(file,
          stft_mat,
          frame_length=1024,
          frame_shift=256,
          center=False,
          window="hann",
          transpose=True,
          norm=None,
          fs=16000,
          nsamps=None):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    samps = audio_lib.istft(
        stft_mat,
        frame_shift,
        frame_length,
        window=window,
        center=center,
        length=nsamps)
    # renorm if needed
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / samps_norm
    # same as MATLAB and kaldi
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    fdir = os.path.dirname(file)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(file, fs, samps_int16)


def apply_cmvn(feats, cmvn_dict):
    #print(feats.shape)
    if type(cmvn_dict) != dict:
        raise TypeError("Input must be a python dictionary")
    if 'mean' in cmvn_dict:
        feats = feats - cmvn_dict['mean']
    if 'std' in cmvn_dict:
        feats = feats / cmvn_dict['std']
    return feats


def parse_scps(scp_path):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f:
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr
    return scp_dict


def filekey(path):
    fname = os.path.basename(path)
    if not fname:
        raise ValueError("{}(Is directory path?)".format(path))
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])


def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find configure files...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)

    for key in config_keys:
        if key not in config_dict:
            raise KeyError("Missing {} configs in yaml".format(key))
    batch_size = config_dict["dataloader"]["batch_size"]
    if batch_size <= 0:
        raise ValueError("Invalid batch_size: {}".format(batch_size))
    num_frames = config_dict["spectrogram_reader"]["frame_length"]
    num_bins = nfft(num_frames) // 2 + 1
    if len(config_dict["train_scp_conf"]) != len(
            config_dict["valid_scp_conf"]):
        raise ValueError("Check configures in train_scp_conf/valid_scp_conf")
    num_spks = 0
    for key in config_dict["train_scp_conf"]:
        if key[:3] == "spk":
            num_spks += 1
    if num_spks != config_dict["model"]["num_spks"]:
        warnings.warn(
            "Number of speakers configured in trainer do not match *_scp_conf, "
            " correct to {}".format(num_spks))
        config_dict["model"]["num_spks"] = num_spks
    return num_bins, config_dict


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
