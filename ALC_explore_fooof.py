#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 01:03:30 2023

@author: arthurlecoz

explore_fooof.py

"""
# %% Paths

import os
import mne

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pingouin as pg

from scipy.stats import sem
from glob import glob

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from datetime import date
todaydate = date.today().strftime("%d%m%y")

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
preproc_dir = root_dir+'/Preproc'
raw_dir = root_dir+'/Raw'
fig_dir = root_dir+'/Figs' 

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

subtypes = ["C1", "N1"]
channels = ["F3", "C3", "O1"]
stages = ['N1', 'N2', 'N3', 'REM', 'WAKEb', 'WAKEd']
psd_palette = ["#000000", "#d00000", "#faa307"]
sem_palette = ['#999999', '#fca5a5', '#fce4ad']

freqs = np.linspace(.5, 40, 159)

    # %% Script offset & exponent

subidlist = []
subtypelist = []
agelist = []
genrelist = []
stagelist = []
chanlist = []
exponentlist = []
offsetlist = []

duration_N1 = []
duration_N2 = []
duration_N3 = []
duration_WAKEb = []
duration_WAKEd = []
duration_REM = []

files = glob(os.path.join(
    fig_dir, "fooof", "*.pkl"
    ))

thisSavingPath = os.path.join(
    fig_dir, "fooof", "nt1_cns_offset_exponent.csv"
    )
if os.path.exists(thisSavingPath):
    print("DataFrame is already computed and saved... Loading...")
    df = pd.read_csv(thisSavingPath)
    del df['Unnamed: 0']
else : 
    for i_f, file in enumerate(files) :
        dic_fooof = pd.read_pickle(file)
        sub_id = file.split("fooof/")[-1].split("_foo")[0]
        if sub_id.startswith("H") : 
            continue
        print(f"...Processing {sub_id} : {i_f+1} / {len(files)}")
        
        epochs = mne.read_epochs(
            glob(os.path.join(preproc_dir, f"{sub_id}*.fif"))[0], 
            preload = True)
        
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
        
        thisCount = {}
        for stage in metadata.newstage.unique() :
            thisCount[stage] = np.count_nonzero(
                metadata.newstage == stage)/2
        
        thisDemo = df_demographics.loc[df_demographics.code == sub_id]
        age = thisDemo.age.iloc[0]
        genre = thisDemo.sexe.iloc[0]
        subtype = thisDemo.diag.iloc[0]
        
        for i_st, stage in enumerate(dic_fooof.keys()) :
            for i_ch, chan in enumerate(channels) :
                subidlist.append(sub_id)
                subtypelist.append(subtype)
                agelist.append(age)
                genrelist.append(genre)
                stagelist.append(stage)
                chanlist.append(chan)
                
                if np.count_nonzero(metadata.newstage == "WAKEb") == 0 : 
                    duration_WAKEb.append(0)
                else :
                    duration_WAKEb.append(thisCount["WAKEb"])
                if np.count_nonzero(metadata.newstage == "WAKEd") == 0 : 
                    duration_WAKEd.append(0)
                else :
                    duration_WAKEd.append(thisCount["WAKEd"])
                if np.count_nonzero(metadata.newstage == "N1") == 0 :
                    duration_N1.append(0)
                else :
                    duration_N1.append(thisCount["N1"])
                duration_N2.append(thisCount["N2"])
                duration_N3.append(thisCount["N3"])
                duration_REM.append(thisCount["REM"])
                
                exponentlist.append(dic_fooof[stage]['Exponent'][i_ch])
                offsetlist.append(dic_fooof[stage]['Offset'][i_ch])
            
    df = pd.DataFrame({
        "sub_id" : subidlist,
        "subtype" : subtypelist,
        "age" : agelist,
        "genre" : genrelist,
        "stage" : stagelist,
        "channel" : chanlist,
        "duration_wakeb" : duration_WAKEb,
        "duration_waked" : duration_WAKEd,
        "duration_N1" : duration_N1,
        "duration_N2" : duration_N2,
        "duration_N3" : duration_N3,
        "duration_REM" : duration_REM,
        "exponent" : exponentlist,
        "offset" : offsetlist,
        })
    
    df.to_csv(thisSavingPath)
    
# %% exponent subplot

x = "stage"
order = ['WAKEd', 'N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["C1", "N1"]
channels = ['F3', 'C3', 'O1']
y = "exponent"

palette = ["#8d99ae", "#d00000"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (5, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title(f"{chan}")
    # ax[i_ch].set_ylim(-0.75, 3.5)
    # for i in range(3) :
    #     ax[i].set_xlabel("")
    #     if i == 2 :
    #         ax[i].set_xticks(
    #             ticks = np.arange(0, 6, 1), 
    #             labels = ["" for i in np.arange(0, 6, 1)]
    #             )
    #     else : 
    #         ax[i].set_xticks(
    #             ticks = [], 
    #             labels = []
    #             )
    #     ax[i].set_yticks(
    #         ticks = np.arange(0, 4, 1), 
    #         labels = ["" for i in np.arange(0, 4, 1)]
    #         )
    #     ax[i].set_ylabel("")
    ax[i_ch].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/fooof/nt1_cns_exponent_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
#%% LME 

this_df = df.loc[df.stage!="WAKEb"]

foi = ["Stage", "Channel", "ß (NT1 vs HS)", "p_val"]
dic = {f : [] for f in foi}

for stage in this_df.stage.unique():
    for channel in this_df.channel.unique() :
        # model_formula = f'permutation_entropy_theta ~ age + C(genre) + duration_waked + duration_N1 + duration_N2 + duration_N3 + duration_REM + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
        model_formula = f'offset ~ age + C(genre) + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
        model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'], missing = 'drop')
        model_result = model.fit()
        
        dic["Stage"].append(stage)
        dic["Channel"].append(channel)
        dic["ß (NT1 vs HS)"].append(
            model_result.params['C(subtype, Treatment("C1"))[T.N1]'])
        dic["p_val"].append(
            model_result.pvalues['C(subtype, Treatment("C1"))[T.N1]']) 
        
corr_pval = multipletests(dic["p_val"], method='fdr_tsbh')
print(np.asarray(dic["Channel"])[corr_pval[0]])
print(np.asarray(dic["Stage"])[corr_pval[0]])
print(np.asarray(corr_pval[1])[corr_pval[0]])
dic['p_corr'] = list(corr_pval[1])
stats_df = pd.DataFrame.from_dict(dic)
print(stats_df)
    
# %% offset subplots

x = "stage"
order = ['WAKEd', 'N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["C1", "N1"]
channels = ['F3', 'C3', 'O1']
y = "offset"

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (5, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title(f"{chan}")
    # ax[i_ch].set_ylim(-12.5, -8)
    # for i in range(3) :
    #     ax[i].set_xlabel("")
    #     if i == 2 :
    #         ax[i].set_xticks(
    #             ticks = np.arange(0, 6, 1), 
    #             labels = ["" for i in np.arange(0, 6, 1)]
    #             )
    #     else : 
    #         ax[i].set_xticks(
    #             ticks = [], 
    #             labels = []
    #             )
    #     ax[i].set_yticks(
    #         ticks = np.arange(-12, -8, 1), 
    #         labels = ["" for i in np.arange(-12, -8, 1)]
    #         )
    #     ax[i].set_ylabel("")
    ax[i_ch].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/fooof/nt1_cns_offset_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )

# %% LME

index = 'C(subtype)[T.N1]'
ape_feats = ['exponent', 'offset']

for i_f, ape_feat in enumerate(ape_feats):
    
    columns = ['sub_id', 'subtype', 'age', 'genre', 'stage', 'channel',
           'duration_wakeb', 'duration_waked', 'duration_N1', 'duration_N2',
           'duration_N3', 'duration_REM', f'{ape_feat}']
    
    formula = (f"{ape_feat} ~ age + genre + duration_wakeb + duration_waked" + 
               "+ duration_N1 + duration_N2 + duration_N3 + duration_REM" + 
               " + C(subtype)*C(stage)*C(channel)")
    
    temp_tval = []; temp_pval = []; chan_l = []; stage_l = []
    subdf = df[columns].dropna()
    md = smf.mixedlm(formula, df, groups = df['sub_id'])
    mdf = md.fit()
    print(mdf.summary())
             
    # _, corrected_pval = fdrcorrection(temp_pval)


#%%
# %% compute difference subtype - mean CTL

this_saving_diff_path = f"{fig_dir}{os.sep}fooof{os.sep}diff_fooof_params.csv"

if os.path.exists(this_saving_diff_path) :
    df_differences = pd.read_csv(this_saving_diff_path)
else : 
    stages_oi = ['N1', 'N2', 'N3', 'REM']
    
    agelist = []
    genrelist = []
    subidlist = []
    subtypelist = []
    stagelist = []
    channellist = []
    diffexponentlist = []
    diffoffsetlist = []
    
    eds_patients = np.asarray(df.sub_id.loc[df.subtype != "C1"].unique())
    mean_expo_ctl = {stage : {} for stage in stages_oi}
    mean_offset_ctl = {stage : {} for stage in stages_oi}
    for stage in stages_oi :
        for chan in channels :
            mean_expo_ctl[stage][chan] = df.exponent.loc[
                (df.subtype == 'C1') 
                & (df.stage ==  stage) 
                & (df.channel == chan)
                ].mean()
            mean_offset_ctl[stage][chan] = df.offset.loc[
                (df.subtype == 'C1')
                & (df.stage ==  stage) 
                & (df.channel == chan)
                ].mean()
    
    
    for sub_id in eds_patients :
        for stage in stages_oi :
            for chan in channels :
                if df.loc[
                    (df.sub_id == sub_id)
                    & (df.stage == stage)
                    ].empty : continue
                agelist.append(df.loc[df.sub_id == sub_id].age.unique()[0])
                genrelist.append(df.loc[df.sub_id == sub_id].genre.unique()[0])
                subidlist.append(sub_id)
                subtypelist.append(df.loc[df.sub_id == sub_id].subtype.unique()[0])
                stagelist.append(stage)
                channellist.append(chan)
                
                diffexponentlist.append(
                    df.exponent.loc[
                        (df.sub_id == sub_id)
                        & (df.stage == stage)
                        & (df.channel == chan)
                        ].iloc[0] - mean_expo_ctl[stage][chan]
                    )
                diffoffsetlist.append(
                    df.offset.loc[
                        (df.sub_id == sub_id)
                        & (df.stage == stage)
                        & (df.channel == chan)
                        ].iloc[0] - mean_offset_ctl[stage][chan]
                    )
    
    df_differences = pd.DataFrame({
        "age" : agelist,
        "genre" : genrelist,
        "sub_id" : subidlist,
        "subtype" : subtypelist,
        "stage" : stagelist,
        "channel" : channellist,
        "diffexponent" : diffexponentlist,
        "diffoffset" : diffoffsetlist,
        })
    df_differences.to_csv(this_saving_diff_path)

# %% plot differences exponent

x = "stage"
order = ['N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["N1", "HI"]
channels = ['F3', 'C3', 'O1']
y = "diffexponent"


palette = ["#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (5, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df_differences.loc[df_differences['channel'] == chan], 
        palette = palette, inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    ax[i_ch].axhline(y = 0, c = 'grey', ls = "dotted", alpha = .5)
    
    ax[i_ch].set_title("")
    ax[i_ch].set_ylim(-1.75, 1.5)
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 4, 1), 
                labels = ["" for i in np.arange(0, 4, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(-1, 2, 1), 
            labels = ["" for i in np.arange(-1, 2, 1)]
            )
        
        ax[i].set_ylabel("")
    ax[i_ch].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/fooof/nolegend_differences_exponent_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )

# %% plot differences offset

x = "stage"
order = ['N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["N1", "HI"]
channels = ['F3', 'C3', 'O1']
y = "diffoffset"


palette = ["#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (5, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df_differences.loc[df_differences['channel'] == chan], 
        palette = palette, inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    ax[i_ch].axhline(y = 0, c = 'grey', ls = "dotted", alpha = .5)
    
    ax[i_ch].set_title("")
    ax[i_ch].set_ylim(-1.5, 2)
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 4, 1), 
                labels = ["" for i in np.arange(0, 4, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(-1, 3, 1), 
            labels = ["" for i in np.arange(-1, 3, 1)]
            )
        
        ax[i].set_ylabel("")
    ax[i_ch].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/fooof/nolegend_differences_offset_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )


# %% script flat spectra

n_h = len(glob(f"{fig_dir}{os.sep}fooof{os.sep}*HI*.pkl"))
n_n = len(glob(f"{fig_dir}{os.sep}fooof{os.sep}*N1*.pkl"))
n_c = len(glob(f"{fig_dir}{os.sep}fooof{os.sep}*C1*.pkl"))

i_st = {"HI" : 0, "N1" : 0, "C1" : 0}

stages = ['WAKEb', 'WAKEd', 'NREM', 'REM']

# list df
subidlist = []
agelist = []
sexelist = []
chanlist = []
subtypelist = []
stagelist = []
flatspecvaluelist = []
freqlist = []

dic_flatspectra = {
    stage : {
        "HI" : np.nan * np.ones((n_h, 3, 159)),
        "N1" : np.nan * np.ones((n_n, 3, 159)),
        "C1" : np.nan * np.ones((n_c, 3, 159))
        } for stage in stages
    }

files = glob(f"{fig_dir}{os.sep}fooof{os.sep}*.pkl")

for i_file, file in enumerate(files):
    sub_id = file.split("of/")[-1][:6]
    print(f"...Processing {sub_id}")
    subtype = sub_id[:2]
    
    thisdemo = df_demographics.loc[df_demographics.code == sub_id]
    
    temp_dic = pd.read_pickle(file)
    
    if "WAKEb" in temp_dic.keys() and "WAKEd" in temp_dic.keys() :
        thisStages = ["WAKEb", "WAKEd", "NREM", "REM"]
    elif "WAKEb" in temp_dic.keys() and not "WAKEd" in temp_dic.keys() :
        thisStages = ["WAKEb", "NREM", "REM"]
    elif "WAKEd" in temp_dic.keys() and not "WAKEb" in temp_dic.keys() :
        thisStages = ["WAKEd", "NREM", "REM"]
    
    for stage in thisStages :
        for i_ch, chan in enumerate(["F3", "C3", "O1"]):
            if stage == "NREM" :
                for freq in freqs :
                    subidlist.append(sub_id)
                    agelist.append(thisdemo.age.iloc[0])
                    sexelist.append(thisdemo.sexe.iloc[0])
                    chanlist.append(chan)
                    subtypelist.append(subtype)
                    stagelist.append(stage)
                    freqlist.append(freq)
                    flatspecvaluelist.append(
                        np.nanmean(
                            [temp_dic["N2"]['Flat_spectra'][
                                i_ch, freqs == freq], 
                             temp_dic["N3"]['Flat_spectra'][
                                 i_ch, freqs == freq]],
                            axis = 0
                            )[0]
                        )
                
                dic_flatspectra["NREM"][subtype][
                    i_st[subtype], i_ch, :] = np.nanmean(
                        [temp_dic["N2"]['Flat_spectra'][i_ch, :], 
                         temp_dic["N3"]['Flat_spectra'][i_ch, :]],
                        axis = 0
                        )
            elif stage == "REM" :
                for freq in freqs :
                    subidlist.append(sub_id)
                    agelist.append(thisdemo.age.iloc[0])
                    sexelist.append(thisdemo.sexe.iloc[0])
                    chanlist.append(chan)
                    subtypelist.append(subtype)
                    stagelist.append(stage)
                    freqlist.append(freq)
                    flatspecvaluelist.append(
                        temp_dic["REM"]['Flat_spectra'][i_ch, freqs == freq][0]
                        )
                dic_flatspectra["REM"][subtype][
                    i_st[subtype], :, :] = temp_dic["REM"]['Flat_spectra'][i_ch, :]
            elif stage == "WAKEb" :
                for freq in freqs :
                    subidlist.append(sub_id)
                    agelist.append(thisdemo.age.iloc[0])
                    sexelist.append(thisdemo.sexe.iloc[0])
                    chanlist.append(chan)
                    subtypelist.append(subtype)
                    stagelist.append(stage)
                    freqlist.append(freq)
                    flatspecvaluelist.append(
                        temp_dic["WAKEb"]['Flat_spectra'][i_ch, freqs == freq][0]
                        )
                dic_flatspectra["WAKEb"][subtype][
                    i_st[subtype], :, :] = temp_dic["WAKEb"]['Flat_spectra'][i_ch, :]
            elif stage == "WAKEd" :
                for freq in freqs :
                    subidlist.append(sub_id)
                    agelist.append(thisdemo.age.iloc[0])
                    sexelist.append(thisdemo.sexe.iloc[0])
                    chanlist.append(chan)
                    subtypelist.append(subtype)
                    stagelist.append(stage)
                    freqlist.append(freq)
                    flatspecvaluelist.append(
                        temp_dic["WAKEd"]['Flat_spectra'][i_ch, freqs == freq][0]
                        )
                dic_flatspectra["WAKEd"][subtype][
                    i_st[subtype], :, :] = temp_dic["WAKEd"]['Flat_spectra'][i_ch, :]
                    
    i_st[subtype] += 1

df = pd.DataFrame({
    "sub_id" : subidlist,
    "age" : agelist,
    "sexe" : sexelist,
    "chan" : chanlist,
    "subtype" : subtypelist,
    "stage" : stagelist,
    "freq" : freqlist,
    "flatspectra" : flatspecvaluelist,
    })
                    
dic_flat = {"C1" : {}, "HI" : {}, "N1" : {}}
dic_flatsem = {"C1" : {}, "HI" : {}, "N1" : {}}

for subtype in subtypes :
    for stage in stages :
        dic_flat[subtype][stage] = np.nanmean(
            dic_flatspectra[stage][subtype], axis = 0
            )
        dic_flatsem[subtype][stage] = sem(
            dic_flatspectra[stage][subtype], axis = 0, nan_policy = 'omit'
            )
        
# %%
from mne.stats import permutation_cluster_test
import scipy

# hsi_flat = np.dstack(
#     [i for i in dic_flatspectra['NREM']['HI']])
# nti_flat = np.dstack(
#     [i for i in dic_flatspectra['NREM']['C1']])
# ctl_flat = np.dstack(
#     [i for i in dic_flatspectra['NREM']['N1']])

alpha_cluster_forming = 0.05
n_conditions = 3
n_observations = 159
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [
     # dic_flatspectra['WAKEb']['HI'], 
     dic_flatspectra['WAKEb']['C1'], 
     dic_flatspectra['WAKEb']['N1']],
    out_type="mask",
    n_permutations=1000,
    threshold=f_thresh,
    tail=0,
    )   

clusterfreqs_O1 = np.append(freqs[clusters[0][0, :]], freqs[clusters[1][0, :]])

# %% Plot flat spectra$
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(6, 12), sharey=True, layout = "constrained")

# Loop through each channel
# for i, channel in enumerate(channels):
#     ax = axs[i]

i = 1
channel = 'O1'
stage = 'WAKEb'

# Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = dic_flat[subtype][stage][i]

    # Calculate the SEM
    sem_db = dic_flatsem[subtype][stage][i]

    # Plot the PSD and SEM
    ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
    ax.fill_between(
        freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
        color = sem_palette[j]
        )

for i_c, c in enumerate(clusters):
    c = c[0,:]
    if cluster_p_values[i_c] <= 0.05:
        h = ax.axvspan(freqs[c].min(), freqs[c].max(), color="r", alpha=0.1)


# hf = plt.plot(freqs, T_obs, "g")
ax.legend((h,), ("cluster p-value < 0.05",))

# Set the title and labels
ax.set_title('Channel: ' + channel)
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim([0.5, 40])
# ax.set_ylim([-30, 60])
ax.legend()

# Add the condition name as a title for the entire figure
fig.suptitle('Condition: ' + stage)

# Add a y-axis label to the first subplot
ax.set_ylabel('dB')
plt.show(block = False)


# %% Stats
palette = ["#5e6472", "#d00000", "#faa307"]

subtype_A = []
subtype_B = []
corr_freqlist = []
corr_p_value = []

for freq in clusterfreqs_O1 :
    ancova = pg.ancova(
        data = df.loc[
            (df.stage == "NREM") & (df.freq == freq)
            ],
        dv = "flatspectra",
        between = "subtype",
        covar = ["age", "sexe"]
        )
    
    tukey = pg.pairwise_tests(
        data = df.loc[
            (df.stage == "NREM") & (df.freq == freq)
            ],
        dv = "flatspectra",
        between = "subtype",
        subject = "sub_id",
        parametric = False,
        alpha = 0.05,
        padjust = 'fdr_bh',
        correction = 'auto'
        )
    
    if ancova['p-unc'].loc[ancova.Source == 'subtype'].iloc[0] < .05 :
        print(f"\nNREM : {freq} :\nTUKEY - FDR :")
        for i_p, p_val in enumerate(tukey['p-corr']) :
            subtype_A.append(tukey['A'].iloc[i_p])
            subtype_B.append(tukey['B'].iloc[i_p])
            corr_freqlist.append(freq)
            corr_p_value.append(np.round(tukey['p-corr'].iloc[i_p], 3))
            # if p_val < 0.1 and p_val > 0.05 :
            #     print(f"{tukey['A'].iloc[i_p]} vs {tukey['B'].iloc[i_p]} - p = {np.round(tukey['p-corr'].iloc[i_p], 3)} ns")
            # if p_val < .05 and p_val > 0.01 :
            #     print(f"{tukey['A'].iloc[i_p]} vs {tukey['B'].iloc[i_p]} - p = {np.round(tukey['p-corr'].iloc[i_p], 3)} *")
            # elif p_val < .01 and p_val > 0.001 :
            #     print(f"{tukey['A'].iloc[i_p]} vs {tukey['B'].iloc[i_p]} - p = {np.round(tukey['p-corr'].iloc[i_p], 3)} **")
            # elif p_val < .001  :
            #     print(f"{tukey['A'].iloc[i_p]} vs {tukey['B'].iloc[i_p]} - p = {np.round(tukey['p-corr'].iloc[i_p], 3)} ***")
        

df_fdr = pd.DataFrame({
    "subtypeA" : subtype_A,
    "subtypeB" : subtype_B,
    "frequency" : corr_freqlist,
    "fdr_pval" : corr_p_value
    })

df_signif = df_fdr.loc[df_fdr.fdr_pval < .05]
df_ct_ni = df_signif.loc[df_signif.subtypeB == "N1"]
df_ct_hi = df_signif.loc[df_signif.subtypeB == "HI"]

# %% Plot Flat C3 FDR

# %% Plot flat spectra$
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(4, 8), sharey=True, layout = "constrained")

# Loop through each channel
# for i, channel in enumerate(channels):
#     ax = axs[i]

i = 1
channel = 'O1'
stage = 'NREM'

# Loop through each population and plot its PSD and SEM
for j, subtype in enumerate(subtypes):
    # Convert power to dB
    psd_db = dic_flat[subtype][stage][i]

    # Calculate the SEM
    sem_db = dic_flatsem[subtype][stage][i]

    # Plot the PSD and SEM
    ax.plot(freqs, psd_db, label = subtype, color = psd_palette[j])
    ax.fill_between(
        freqs, psd_db - sem_db, psd_db + sem_db, # alpha=0.3, 
        color = sem_palette[j]
        )

ax.axhline(
    .75, 13.5/40, 18.25/40, 
    color = psd_palette[1], alpha = 1, linewidth = 3, ls = "-"
    )
ax.axhline(
    .77, 13.5/40, 16.25/40, 
    color = psd_palette[2], alpha = 1, linewidth = 3
    )

# Set the title and labels
# ax.set_title('Channel: ' + channel)
# ax.set_xlabel('Frequency (Hz)')
ax.set_xticks(
    ticks = np.arange(5, 45, 5), 
    labels = ["" for i in np.arange(5, 45, 5)]
    )
ax.set_yticks(
    ticks = np.arange(-.2, 1, .2), 
    labels = ["" for i in np.arange(-.2, 1, .2)]
    )
ax.set_xlim([1, 40])
ax.set_ylim([-.2, .8])
ax.legend_ = None

# Add the condition name as a title for the entire figure
# fig.suptitle('Condition: ' + stage)

# Add a y-axis label to the first subplot
# ax.set_ylabel('dB')
plt.show(block = False)
plt.savefig(
    f"{fig_dir}{os.sep}fooof{os.sep}NOLEGEND_flat_spectra_wdifferences_O1.png", 
    dpi = 200)

# %% Plot flat spectras

for stage in stages :
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 9), sharey=True, layout = "constrained")
   
    # Loop through each channel
    for i, channel in enumerate(channels):
        # Loop through each population and plot its PSD and SEM
        for j, subtype in enumerate(subtypes):
            # Convert power to dB
            psd_db = dic_flat[subtype][stage][i]
        
            # Calculate the SEM
            sem_db = dic_flatsem[subtype][stage][i]
        
            # Plot the PSD and SEM
            ax[i].plot(
                freqs, 
                psd_db, 
                label = subtype, 
                color = psd_palette[j]
                )
            ax[i].fill_between(
                freqs, 
                psd_db - sem_db, 
                psd_db + sem_db,  alpha=0.3, 
                color = psd_palette[j]
                )
    
        # Set the title and labels
        ax[i].set_title('Channel: ' + channel)
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_xlim([0.5, 40])
    # ax.set_ylim([-30, 60])
    ax[2].legend()
    
    # Add the condition name as a title for the entire figure
    fig.suptitle('Condition: ' + stage)
    
    # Add a y-axis label to the first subplot
    ax[0].set_ylabel('dB')
    plt.show(block = False)
    plt.savefig(os.path.join(
        fig_dir, "fooof", f"periodic_PSD_{stage}.png"
        ), dpi = 200)
