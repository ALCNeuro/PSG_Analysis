#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:53:05 2023

@author: arthurlecoz

compute_STAGES_incertitudes.py

i need to get the proba of each sleep stage for every epochs
-> And the entropy
-> And the dKL

One big df with 
Wake ==> But for wake also the information "before" or "during" night
N1 => Which would be np nan for other stage
N2
N3
REM


"""
# %% paths & packages

from glob import glob

import os
import mne

import pandas as pd
import numpy as np

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
stages = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]
stage_strtoint = {"WAKEd" : 0, "N1" : 1, "N2" : 2, "N3" : 3, "REM" : 5}
# stage_strtoint = {"WAKE" : 0, "N1" : 1, "N2" : 2, "N3" : 3, "REM" : 5}

stage_pos = {
    0 : 0, 
    1 : 1, 
    2 : 2, 
    3 : 3, 
    5 : 4}

channels = ['F3', 'C3', 'O1']
subtypes = ['N1', 'C1', 'HI']

basic_params = [
    "sub_id", "age", "genre", "subtype", "stage", "channel", 
    "duration_wakeb", "duration_waked", "duration_N1", "duration_N2",
    "duration_N3", "duration_REM", "entropy", "dKL",
    "p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM"
    ]

tradi_stage = ["WAKE", "N1", "N2", "N3", "REM"]

# %% script 

big_dic = {param : [] for param in basic_params}
files = glob(f"{raw_dir}{os.sep}PSG4_Hypnogram_Export*PSG1.txt")

thisSavingPath = os.path.join(
    fig_dir, "nt1vcns_df_stages_entropy_dkl.csv"
    )
            
for i_file, file in enumerate(files) :
    
    sub_id = file.split('Export_')[-1][:-4]
    exam = sub_id[-4:]
    code = sub_id[:-5]
    subtype = code[:2]
    
    if subtype == 'HI' : continue
    
    if code not in valid_codes :
        print(f"\n...{code} skipped...\n")
        continue 
    print(f"...Processing {sub_id} : {i_file+1} / {len(files)}")
    
    thisDemo = df_demographics.loc[df_demographics.code == code]
    age = thisDemo.age.iloc[0]
    genre = thisDemo.sexe.iloc[0]
    
    thisEpochs = glob(
        os.path.join(preproc_dir, f"*{code}*PSG1*clean_meta_epo.fif"
                     ))[0]
    
    epochs = mne.read_epochs(thisEpochs, preload = True)
    metadata = epochs.metadata.reset_index()
    orderedMetadata = metadata.sort_values("n_epoch")
    
    epochsWakeAfterNight = []
    while orderedMetadata.divBin2.iloc[-1] == "noBin" :
        epochsWakeAfterNight.append(orderedMetadata.n_epoch.iloc[-1])
        orderedMetadata = orderedMetadata.iloc[:-1]
        
    for thisEpoch in epochsWakeAfterNight :
        metadata = metadata.loc[metadata['n_epoch'] != thisEpoch]
    epochs = epochs[np.asarray(metadata.index)]
    
    bigstagelist = []
    for i_st, stage in enumerate(metadata.scoring):
        if stage == 'WAKE' and metadata.night_status.iloc[i_st] == "outNight" :
            bigstagelist.append('WAKEb')
        elif stage == 'WAKE' and metadata.night_status.iloc[i_st] == "inNight" :
            bigstagelist.append('WAKEd')
        else :
            bigstagelist.append(stage)
    epochs.metadata["newstage"] = bigstagelist
    metadata = epochs.metadata.reset_index()
    orderedMetadata = metadata.sort_values("n_epoch")
    
    thisCount = {}
    for stage in orderedMetadata.newstage.unique() :
        thisCount[stage] = np.count_nonzero(
            orderedMetadata.newstage == stage)/2
    
    sorted_hypnogram = metadata.sort_values('n_epoch').scoring.to_numpy()
    kept_epochs = metadata.sort_values('n_epoch').n_epoch.to_numpy()
    
    hypnodensity_file = glob(
        f"{stages_dir}{os.sep}*{code}*{exam}*e30.hypnodensity.txt"
        )[0]
    hypnopred_file = glob(
        f"{stages_dir}{os.sep}*{code}*{exam}*e30.hypnogram.txt"
        )[0]
    
    hypnodensity = np.loadtxt(
        hypnodensity_file, dtype = float, delimiter = ','
        )
    hypnopred = np.loadtxt(
        hypnopred_file, dtype = int, delimiter = ','
        )
    
    # Entropy
    this_hd_entropy = [-sum(p * np.log2(p) 
                            for p in probas if p != 0) 
                       for probas in hypnodensity]
    
    # dKL
    log_ratio = np.log2(hypnodensity[:-1] / hypnodensity[1:])
    aux_td_dkl = np.sum(hypnodensity[:-1] * log_ratio, axis=1)
    
    long_aux_td_dkl = np.append(aux_td_dkl, np.nan)

    for stage in metadata.newstage.unique() :
        for channel in channels :
            big_dic['sub_id'].append(code)
            big_dic['subtype'].append(subtype)
            big_dic['age'].append(age)
            big_dic['genre'].append(genre)
            big_dic['stage'].append(stage)
            big_dic['channel'].append(channel)
            
            if np.count_nonzero(orderedMetadata.newstage == "WAKEb") == 0 : 
                big_dic['duration_wakeb'].append(0)
            else :
                big_dic['duration_wakeb'].append(thisCount["WAKEb"])
            if np.count_nonzero(orderedMetadata.newstage == "WAKEd") == 0 : 
                big_dic['duration_waked'].append(0)
            else :
                big_dic['duration_waked'].append(thisCount["WAKEd"])
            if np.count_nonzero(orderedMetadata.newstage == "N1") == 0 :
                big_dic['duration_N1'].append(np.nan)
            else :
                big_dic['duration_N1'].append(thisCount["N1"])
            big_dic['duration_N2'].append(thisCount["N2"])
            big_dic['duration_N3'].append(thisCount["N3"])
            big_dic['duration_REM'].append(thisCount["REM"])
            
            thishypnopred = hypnopred[
                metadata.n_epoch[metadata.newstage == stage]
                ]
            thishypnodensity = hypnodensity[
                metadata.n_epoch[metadata.newstage == stage]
                ]
            thisentropy = np.nanmean(np.asarray(this_hd_entropy)[
                metadata.n_epoch[metadata.newstage == stage]
                ])
            thisdkl = np.nanmean(long_aux_td_dkl[
                metadata.n_epoch[metadata.newstage == stage]
                ])
            
            big_dic["entropy"].append(thisentropy)
            big_dic["dKL"].append(thisdkl)
            
            thisproba = np.mean(thishypnodensity, axis = 0)
            "p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM"
            for i_st, st in enumerate(tradi_stage) :
                big_dic[f'p_{st}'].append(thisproba[i_st])
    
df = pd.DataFrame.from_dict(big_dic)
small_df = df[
    ['sub_id', 'age', 'genre', 'subtype', 'stage', 'entropy',
     'duration_wakeb', 'duration_waked', 'duration_N1', 
     'duration_N2', 'duration_N3', 'duration_REM',
     'dKL', 'p_WAKE', 'p_N1', 'p_N2', 'p_N3', 'p_REM']
    ].groupby(
        ['sub_id', 'age', 'genre', 'subtype', 'stage'],
        as_index = False
        ).mean()
small_df.to_csv(thisSavingPath)

# %% plot
import seaborn as sns
import matplotlib.pyplot as plt

order = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]

fig, ax = plt.subplots()

sns.boxplot(
    data = small_df, 
    x = "stage", 
    order = order, 
    y = "entropy", 
    hue = "subtype",
    ax = ax
    )

# %% 

# from scipy.stats import zscore

# kept_entropy = np.asarray(this_hd_entropy)[kept_epochs]
# kept_dkl = long_aux_td_dkl[kept_epochs]
# kept_hypnopred = hypnopred[kept_epochs]
# actual_hypnogram_int = np.array(
#     [stage_strtoint[stage] for stage in sorted_hypnogram]
#     )

# # Creating the plots
# plt.figure(figsize=(15, 10))

# # Plotting the Hypnograms
# plt.subplot(2, 1, 1)
# plt.plot(actual_hypnogram_int, label='Actual Hypnogram', linewidth=2)
# plt.plot(agree_hypnopred, label='Predicted Hypnogram', linestyle='dotted', color='red')
# plt.ylabel('Sleep Stage')
# plt.title('Sleep Hypnograms')
# plt.legend()

# # Plotting the Features
# plt.subplot(2, 1, 2)
# plt.plot(zscore(agree_entropy), label='Entropy', linewidth=2)
# plt.plot(zscore(agree_dkl), label='dKL', linewidth=2)
# plt.ylabel('Feature Value')
# plt.title('Features Extracted Based on Hypnodensity')
# plt.legend()

# plt.xlabel('Epoch')
# plt.tight_layout()
# plt.show()