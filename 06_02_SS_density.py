#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/08/23

@author: Arthur_LC

ALC_SS_density.py

Currently for HI + NT1 + CNS

"""

# %%% Paths & Packages

from glob import glob
import numpy as np
import pandas as pd
import mne
import os

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = os.path.join(root_dir, "Raw")
preproc_dir = os.path.join(root_dir, "Preproc")
fig_dir = os.path.join(root_dir, "Figs")

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

# %%% Script

n1_ss = glob(os.path.join(root_dir, "Figs", "yasa", "*N1*130124.npy"))
hi_cns_ss = glob(os.path.join(root_dir, "Figs", "yasa", "hi_cns", "*.npy"))

files = n1_ss + hi_cns_ss

df = pd.concat([pd.read_csv(file) for file in files])
del df['Unnamed: 0']

subidlist = []
agelist = []
sexelist = []
countlist = []
stagelist = []
chanlist = []
subtypelist = []
duration_N2 = []
duration_N3 = []

densitylist = []
freqlist = []
durationlist = []
amplitudelist = []
RMSlist = []
abspowerlist = []
relpowerlist = []
oscillationlist = []
symmetrylist = []


for i, sub_id in enumerate(df.sub_id.unique()) :
    print(f"...Processing {sub_id} : {i+1}/{len(files)}")
    
    this_demographics = df_demographics.loc[
        df_demographics.code == sub_id
        ]
    
    this_df = df.loc[df["sub_id"] == sub_id]
    subtype = this_df.subtype.unique()[0]
    
    epochs = mne.read_epochs(glob(
        os.path.join(preproc_dir, f"*{sub_id}*")
        )[0])
    
    hypno = np.asarray(epochs.metadata.scoring)
    
    thisCount = {}
    for stage in this_df.stage.unique() :
        thisCount[stage] = np.count_nonzero(hypno == stage)/2
    
    for stage in this_df.stage.unique() :
            for chan in this_df.Channel.unique() :
                n_epoch = np.count_nonzero(hypno == stage)
                n_wave = np.count_nonzero(
                    (this_df.stage == stage)
                    & (this_df.Channel == chan)
                    )
                # if n_wave == 0:
                #     continue;
                subidlist.append(sub_id)
                agelist.append(this_demographics.age.iloc[0])
                sexelist.append(this_demographics.sexe.iloc[0])
                subtypelist.append(subtype)
                stagelist.append(stage)
                chanlist.append(chan)
                duration_N2.append(thisCount["N2"])
                duration_N3.append(thisCount["N3"])
                
                countlist.append(n_wave)
                densitylist.append(n_wave / n_epoch)
                freqlist.append(
                    this_df.Frequency.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                durationlist.append(
                    this_df.Duration.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                amplitudelist.append(
                    this_df.Amplitude.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                RMSlist.append(
                    this_df.RMS.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                abspowerlist.append(
                    this_df.AbsPower.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                relpowerlist.append(
                    this_df.RelPower.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                oscillationlist.append(
                    this_df.Oscillations.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                symmetrylist.append(
                    this_df.Symmetry.loc[
                        (this_df["stage"] == stage)
                        & (this_df["Channel"] == chan)
                        ].mean()
                    )
                
df = pd.DataFrame(
    {
     "sub_id" : subidlist,
     "age" : agelist,
     "sexe" : sexelist,
     "subtype" : subtypelist,
     "stage" : stagelist,
     "channel" : chanlist,
     "duration_N2" : duration_N2,
     "duration_N3" : duration_N3,
     "count" : countlist,
     "density" : densitylist,
     "abs_power" : abspowerlist,
     "rel_power" : relpowerlist,
     "frequency" : freqlist,
     "duration" : durationlist,
     "amplitude" : amplitudelist,
     "oscillations" : oscillationlist,
     "RMS" : RMSlist,
     "symmetry" : symmetrylist,
     }
    )

df.to_csv(os.path.join(
    fig_dir, "yasa", "df_nt1_hi_cns_ss_yasa.csv"
    ))
