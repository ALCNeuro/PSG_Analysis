#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Dec 21 10:55:29 2023

@author: arthurlecoz

ALC_SS_detect.py

"""
# %% Pathways

import mne
import os
import pandas as pd
import numpy as np
from yasa import spindles_detect
from glob import glob
# import seaborn as sns, matplotlib.pyplot as plt
from datetime import date
todaydate = date.today().strftime("%d%m%y")

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}{os.sep}Raw"
preproc_dir = f"{root_dir}{os.sep}Preproc"
fig_dir = f"{root_dir}{os.sep}Figs"

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

stage_str_to_int = {
    "WAKE" : 0,
    "N1" : 1,
    "N2" : 2,
    "N3" : 3,
    "REM" : 4
    }

# %% Loop over subjects

sub_id_l = [] ; subtype_l = []; listdf = []

files = glob(f"{preproc_dir}{os.sep}*PSG1*.fif")

for i, file in enumerate(files):
    sub_id = file.split('Preproc/')[-1].split('_PSG1_clean')[0]
    if sub_id.startswith('H') :
        continue
    
    thisSpindleSavename = (
        f"{fig_dir}/yasa/ALL_NoRelPow_sleepspindles_{sub_id}_040924.npy"
        )
    
    if os.path.exists(thisSpindleSavename) :
        continue
    
    print(f"...Processing {sub_id} : {i+1}/{len(files)}")
    if sub_id.startswith("C") :
        subtype = "CTL"
    elif sub_id.startswith("N") :
        subtype = "NT1"
    
    #### Load data
    epochs = mne.read_epochs(file, preload = True)
    sf = epochs.info['sfreq']
    metadata = epochs.metadata
    
    #### Load Hypnogram
    metadata = metadata.sort_values(by = "n_epoch")
    metadata.reset_index(inplace = True)
    
    # Sub metadata to only pick the nrem epochs
    nrem_meta = metadata.loc[
        (metadata['scoring'] == "N2")
        | (metadata['scoring'] == "N3")
        ]
    
    #### SW detect 
    
    _, nchan, nsamples = epochs._data.shape
    dataframeList = []
    
    for i_epoch, n_epoch in enumerate(nrem_meta.n_epoch.unique()) :
        thisStage = nrem_meta.scoring.loc[
            nrem_meta['n_epoch'] == n_epoch].iloc[0]
        this_eeg_chan = epochs.copy()[
            epochs.metadata.n_epoch == n_epoch
            ].get_data(units = 'uV')[0]
    
        sp_det = spindles_detect(
            data = this_eeg_chan,
            sf = sf,
            hypno = None,
            include = None,
            ch_names = ["F3", "C3", "O1"],
            freq_sp = (12,15),
            freq_broad = (1,30),
            duration = (0.5, 2.5),
            min_distance = 500,
            thresh = {'corr': 0.65, 'rel_pow': None, 'rms': 1.5},
            multi_only = False,
            remove_outliers = False,
            verbose = 0
            )
        if sp_det == None :
            continue
        sp_df = sp_det.summary()
        sp_df.insert(
            sp_df.shape[1], "n_epoch", [n_epoch for i in range(len(sp_df))]
            )
        sp_df.insert(
            sp_df.shape[1], "stage", [thisStage for i in range(len(sp_df))]
            )
        sp_df.insert(0,"sub_id", [sub_id for i in range(len(sp_df))])
        sp_df.insert(1,"subtype", [subtype for i in range(len(sp_df))])
        dataframeList.append(sp_df)
    df = pd.concat(dataframeList)
    df.to_csv(thisSpindleSavename)
        
# df = pd.concat(listdf)
# df.to_csv(
#     f"{fig_dir}{os.sep}yasa{os.sep}ALL_NoRelPow_SpDet_long_N2N3_{todaydate}.csv"
#     )
print("\n! All done !")

