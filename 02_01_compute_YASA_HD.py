#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:15:34 2024

@author: arthurlecoz

02_01_compute_YASA_HD.py
"""
# %% paths & packages

from glob import glob

import os
import mne
import yasa

import pandas as pd
import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable


root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
preproc_dir = root_dir+'/Preproc'
raw_dir = root_dir+'/Raw'
fig_dir = root_dir+'/Figs' 
stages_dir = root_dir + '/Stages'

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

valid_codes = np.asarray(df_demographics.code)

files = glob(os.path.join(raw_dir, "*PSG1.edf"))

# %%

channel_oi = ['C3','EOG G','EMG 1','EMG 2','A2']
str_to_int = {"N1" : 1, "N2" : 2, "N3" : 3, "W" : 0, "R" : 5}

def map_values(x):
    return str_to_int[x]

for i, file in enumerate(files) :
    sub_id = file.split('Raw/')[-1][:6]
    if sub_id not in valid_codes : continue

    this_hypnodensity_savepath = os.path.join(
        fig_dir, "yasa", f"{sub_id}_hypnodensity_yasa.csv"
        )
    
    if os.path.exists(this_hypnodensity_savepath) : continue

    raw = mne.io.read_raw_edf(file, include = channel_oi, preload = True)
    mne.set_bipolar_reference(
        raw, "EMG 1", "EMG 2", ch_name = "EMG", copy = False
        )
    raw.set_channel_types({'EMG':'emg'})
    mne.set_eeg_reference(raw, ["A2"], copy = False)
    raw.drop_channels(['A2'])
    
    sfreq = raw.info["sfreq"] 
    raw.resample(100, npad="auto")
    
    sls = yasa.SleepStaging(
        raw, 
        eeg_name="C3", 
        eog_name="EOG G", 
        emg_name="EMG", 
        )
    y_pred = sls.predict()
    hypnodensity = sls.predict_proba()
    confidence = sls.predict_proba().max(1)
    
    hypnodensity.insert(0, "confidence", confidence)
    
    vectorized_map_values = np.vectorize(map_values)
    y_pred_mapped = vectorized_map_values(y_pred)
    
    hypnodensity.insert(0, "scorred_stage", y_pred)
    hypnodensity.insert(0, "int_stage", y_pred_mapped)
    hypnodensity.reset_index(inplace = True)
    
    hypnodensity.to_csv(this_hypnodensity_savepath)
    
