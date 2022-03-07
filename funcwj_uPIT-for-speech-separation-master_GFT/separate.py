#!/usr/bin/env python
# coding=utf-8
# wujian@2018
"""
Using speaker mask produced by neural networks to separate single channel speech
"""

import argparse
import os
import pickle

import numpy as np
import torch as th
import scipy.io as sio

from utils import stft, istft, parse_scps, apply_cmvn, parse_yaml, EPSILON, stft_GFT
from model import PITNet

import torch.nn as nn



class Separator(object):
    def __init__(self, nnet, state_dict, cuda=False):
        if not os.path.exists(state_dict):
            raise RuntimeError(
                "Could not find state file {}".format(state_dict))
        self.nnet = nnet

        self.location = None if args.cuda else "cpu"
        self.nnet.load_state_dict(
            th.load(state_dict, map_location=self.location))
        self.nnet.eval()

    def seperate(self, spectra1, cmvn=None, apply_log=True):
        """
            spectra: stft complex results T x F
            cmvn: python dict contains global mean/std
            apply_log: using log-spectrogram or not
        """
        #if not np.iscomplexobj(spectra):
            #raise ValueError("Input must be matrix in complex value")
        # compute (log)-magnitude spectrogram

        stft_spectra, gft_spectra = spectra1.split([129, 256], 1)
        stft_gft_mat11 = th.cat((np.abs(stft_spectra), np.abs(gft_spectra)), 1)
        #print(stft_gft_mat11.dtype)
        inputnum = stft_gft_mat11.shape[1]
        outputnum = stft_spectra.shape[1]
        stft_gft_mat11 = th.unsqueeze(stft_gft_mat11, 2)
        #print(stft_gft_mat11.dtype)
        gft_stft_out = nn.Conv1d(inputnum, outputnum, 1, bias=False)
        stft_gft_mat1 = gft_stft_out(stft_gft_mat11)
        #print(stft_gft_mat1.shape)
        stft_gft_mat2 = th.squeeze(stft_gft_mat1, 2)
        #spectra11 = stft_gft_mat2
        spectra11 = stft_gft_mat2.detach().numpy()
        #print(spectra11.shape)

        input_spectra = np.log(np.maximum(
            np.abs(spectra11), EPSILON)) if apply_log else np.abs(spectra11)

        # apply cmvn or not
        input_spectra = apply_cmvn(input_spectra,
                                   cmvn) if cmvn else input_spectra

        out_masks = self.nnet(
            th.tensor(input_spectra, dtype=th.float32, device=self.location),
            train=False)

        spk_masks = [spk_mask.cpu().data.numpy() for spk_mask in out_masks]
        #print(input_spectra.shape)
        #print(input_spectra.dtype)
        return spk_masks, [stft_spectra * spk_mask for spk_mask in spk_masks]


def run(args):
    num_bins, config_dict = parse_yaml(args.config)
    dataloader_conf = config_dict["dataloader"]
    spectrogram_conf = config_dict["spectrogram_reader"]
    # Load cmvn
    dict_mvn = dataloader_conf["mvn_dict"]
    if dict_mvn:
        if not os.path.exists(dict_mvn):
            raise FileNotFoundError("Could not find mvn files")
        with open(dict_mvn, "rb") as f:
            dict_mvn = pickle.load(f)
    # default: True
    apply_log = dataloader_conf[
        "apply_log"] if "apply_log" in dataloader_conf else True

    dcnet = PITNet(num_bins, **config_dict["model"])

    frame_length = spectrogram_conf["frame_length"]
    frame_shift = spectrogram_conf["frame_shift"]
    window = spectrogram_conf["window"]

    separator = Separator(dcnet, args.state_dict, cuda=args.cuda)

    utt_dict = parse_scps(args.wave_scp)
    num_utts = 0
    for key, utt in utt_dict.items():
        try:
            samps, stft_gft_mat = stft(
                utt,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                return_samps=True)
        except FileNotFoundError:
            print("Skip utterance {}... not found".format(key))
            continue
        print("Processing utterance {}".format(key))

        num_utts += 1
        norm = np.linalg.norm(samps, np.inf)
        spk_mask, spk_spectrogram = separator.seperate(
            stft_gft_mat, cmvn=dict_mvn, apply_log=apply_log)

        #stft_mat11, gft_spect = stft_gft_mat.split([129, 256], 1)
        #stft_mat11 = stft_mat11.numpy()
        #print(stft_mat11.dtype)
        for index, stft_mat1 in enumerate(spk_spectrogram):
            stft_mat11 = stft_mat1.numpy()
            istft(
                os.path.join(args.dump_dir, '{}.spk{}.wav'.format(
                    key, index + 1)),
                stft_mat11,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                norm=norm,
                fs=8000,
                nsamps=samps.size)
            #print(stft_mat11.shape)
            if args.dump_mask:
                sio.savemat(
                    os.path.join(args.dump_dir, '{}.spk{}.mat'.format(
                        key, index + 1)), {"mask": spk_mask[index]})
    print("Processed {} utterance!".format(num_utts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to seperate single-channel speech using masks generated by neural networks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./conf/train.yaml",
        help="Location of training configure files")
    parser.add_argument(
        "--state_dict",
        type=str,
        default="./tune/2spk_pit_a/epoch.2.pkl",
        help="Location of networks state file")
    parser.add_argument(
        "--wave_scp",
        type=str,
        default="/media/dell/wtt_resources/singapore/projects/con_tasNet/data/create_scp/tt_mix.scp",
        help="Location of input wave scripts in kaldi format")
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        dest="cuda",
        help="If true, inference on GPUs")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="cache",
        dest="dump_dir",
        help="Location to dump seperated speakers")
    parser.add_argument(
        "--dump-mask",
        default=False,
        action="store_true",
        dest="dump_mask",
        help="If true, dump mask matrix")
    args = parser.parse_args()
    run(args)
