########################################################
#        mix noise to LRS2, LRS3 trainval / test 
########################################################

import argparse
import os
import numpy as np
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

import numpy as np
import torch
import time
import torchaudio

import pdb
import glob
from tqdm import tqdm

def compare_noise(noise_wav_list):
    noise_len_list = []

    for i in range(len(noise_wav_list)):
        noise_len_list.append(noise_wav_list[i].shape[1])
    noise_len_list.sort()
    cut_f = noise_len_list[0]
    for i in range(len(noise_wav_list)):
        start = (noise_wav_list[i].shape[1]-cut_f)//2
        noise_wav_list[i]=noise_wav_list[i][:,start:start+cut_f]

    mixing_noise = 0
    for i in range(len(noise_wav_list)):
        mixing_noise += noise_wav_list[i]
    return mixing_noise
    
    
def compare_src_noise(src, noisy):
    src_len = src.shape[1]
    noisy_len = noisy.shape[1]
    k = np.random.randint(low=0, high = noisy_len - src_len)
    np_src = src.squeeze(0)
    np_noisy = noisy.squeeze(0)
    if src_len > noisy_len :
        p = src_len//noisy_len
        np_noisy = torch.cat((np_noisy.repeat(p), np_noisy[0:src_len%noisy_len]),dim=-1)

    elif src_len < noisy_len :
        np_noisy = np_noisy[k:k+src_len]
    else :
        np_noisy = np_noisy
        np_src = np_src
        
    return np_src.numpy(), np_noisy.numpy()
    

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))
    

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms



def data_augment_onthefly(src,noise_paths,SNR):
    
    snr = SNR
    clean_wav , _ = torchaudio.load(src)

    ############ noisy audio prepare ######################
    #pdb.set_trace()
    noise_file_list = []
    k = np.random.randint(low=0, high=len(noise_paths))
    noise_file_list.append(noise_paths[k]) # select one noise file out of 20s

    noise_wav_list = []
    for i in range(len(noise_file_list)):
        noise_wav , _ = torchaudio.load(noise_file_list[i])
        # noise_wav is 2 channel -> select one channel
        noise_wav = noise_wav[0]
        noise_wav = noise_wav.unsqueeze(0)
        noise_wav_list.append(noise_wav)
        
    if len(noise_file_list) > 1:
        print("222")
        mixing_wav = compare_noise(noise_wav_list)
    else :
        mixing_wav = noise_wav
    clean_amp, noise_amp = compare_src_noise(clean_wav,mixing_wav)

    clean_rms = cal_rms(clean_amp)
    noise_rms = cal_rms(noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise_amp = noise_amp * (adjusted_noise_rms / noise_rms)
    mixed_amp = (clean_amp + adjusted_noise_amp)
    mixed_amp=np.expand_dims(mixed_amp,axis=0)
    mixed_amp = torch.from_numpy(mixed_amp)

    return clean_wav, mixed_amp


def save_mixed_audio(mixed_audio, clean_path, SNR):
    # find saving_path
    original_dir = "/home/nas/DB/[DB]_LIPREADING/LRS_con_wav/"
    save_dir = "/home/nas3/DB/[DB]_AVSR/LRS_con_noisy/"
    save_full_path = clean_path.replace(original_dir, save_dir)
    save_path_list = save_full_path.split("/")
    save_path_list.insert(-2,str(SNR))
    save_full_path = "/".join(save_path_list)

    # if there's no directory, make directory
    save_root_dir = os.path.dirname(save_full_path)
    if not os.path.exists(save_root_dir):
        try:
            os.makedirs(save_root_dir)
        except OSError as exc:
            print("error")

    # save
    torchaudio.save(save_full_path, mixed_audio, 16000)



########################################################
#                         MAIN
########################################################

if __name__ == '__main__':
    
    # 1. get clean audio_path_list
    clean_paths = np.loadtxt("./clean_audio.txt", str)
    start_idx = 0 # for parallel procesesing
    end_idx = 10000
    clean_paths = clean_paths[start_idx : end_idx]
    
    # 2. get noise_path_list
    noise_dir_path = "/home/nas/user/jungwook/Noise"
    noise_paths = glob.glob(noise_dir_path + "/" + "*.wav")
    #pdb.set_trace()

    # 3. hyperparameters
    fs = 16000
    win_len = int(1024*(fs/16))
    window = torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    
    for clean_path in tqdm(clean_paths):
        
        SNR_list = [-5, 0, 5, 10, 15, 20]    
        for SNR in SNR_list:
            pdb.set_trace()
            # 4. mix noise
            _, mix_wav = data_augment_onthefly(src = clean_path, noise_paths = noise_paths, SNR = SNR)

            # 5. save mixed audio 
            save_mixed_audio(mix_wav, clean_path, SNR)
        
