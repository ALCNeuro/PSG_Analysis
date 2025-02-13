#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues 4 July 2023

@author: Arthur_LC

SW_threshold_stage_subject.py

"""
# %%% Paths & Packages

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import mne
from glob import glob
from scipy.stats import exponnorm
from datetime import date
import os
todaydate = date.today().strftime("%d%m%y")

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}{os.sep}Raw"
preproc_dir = f"{root_dir}{os.sep}Preproc"
fig_dir = f"{root_dir}{os.sep}Figs"

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

# %%% Script

inspect = False

stage_int_to_str = {1 : "N1", 2 : "N2", 3 : "N3", 4 : "REM"}

slope_range = [0.25, 1] # in uV/ms
positive_amp = [75] # in uV
amplitude_max = 150
filt_range = [0.2, 7] # in Hz
thr_value = 0.1

files = glob(f"{fig_dir}/allwaves/*clean_epochs.csv")

subidlist = []
genrelist = []
agelist = []
esslist = []
sollist = [] # Sleep Onset Latency: 'lat.end.N1'
tstlist = [] # Total Sleep Time: 'TST.N1'
efflist = [] # Sleep Efficiency: 'eff.N1'
wasolist = [] # Wake After Sleep Onset: 'WASO.N1'
melist = [] # Micro Awakenings index: 'ME.N1'
hyperonilist = []
blackoutlist = []
# soremtotlist = []
dursemtot = []
dursemwe = []
ivresse = []
sieste = []

duration_N1 = []
duration_N2 = []
duration_N3 = []
duration_WAKEb = []
duration_WAKEd = []
duration_REM = []

countlist = []
stagelist = []
densitylist = []
ptplist = []
frequencylist = []
dslopelist = []
uslopelist = []

chanlist = []
subtypelist = []

if inspect :
    palette_bins = ["#9E2A2B", "#E09F3E", "#335C67"]
    palette_fit = ["#4F1516", "#7B5214", "#192E33"]

for i, file in enumerate(files) :
    filename = file.split(os.sep)[-1]
    sub_id = '_'.join(filename.split('_')[0:2])
    
    thisDemographics = df_demographics.loc[
        df_demographics['code'] == sub_id
        ]
    
    thisEpochsFile = glob(
        os.path.join(
            preproc_dir, f"*{sub_id}*PSG1*clean_meta*epo.fif"
            )
        )[0]
    print(f"\n...Processing {sub_id} : {i+1}/{len(files)}")
    
    epochs = mne.read_epochs(thisEpochsFile, preload = False)
    df_metadata = epochs.metadata
    orderedMetadata = df_metadata.sort_values("n_epoch")
     
    if sub_id.startswith("C") :
        subtype = "CTL"
    elif sub_id.startswith("N") :
        subtype = "NT1"
    else : 
        continue

    df_sw = pd.read_csv(file)
    del df_sw['Unnamed: 0']
    
    df_sw = df_sw.loc[
        (df_sw["stage"] != 3)
        & (df_sw['PTP'] < 150)
        | (df_sw["stage"] == 3)
        & (df_sw['PTP'] < 250)
        ]
    
    df_sw = df_sw.loc[
        (df_sw["pos_halfway_period"] < slope_range[1])
        & (df_sw["pos_halfway_period"] > slope_range[0])
        ]
    
    if inspect :
        report = mne.Report(title=f"PTP inspections of {sub_id} | {subtype}")

    thresh_dic = {0 : {}, 1 : {}, 2 : {}, 3 : {}, 4 : {}}
    
    for stage in df_sw.stage.unique() :
        for i, chan in enumerate(df_sw.chan_name.unique()) :
            temp_p2p = np.asarray(
                df_sw.PTP.loc[
                    (df_sw['chan_name'] == chan)
                    & (df_sw['stage'] == stage)]
                )
            if len(temp_p2p) == 0 :
                continue
            params = exponnorm.fit(temp_p2p)#, floc=temp_sw[:,9].min())
            mu, sigma, lam = params
            bins = np.arange(0, temp_p2p.max(), 0.1)
            y = exponnorm.pdf(bins, mu, sigma, lam)
            max_gaus = bins[np.where(y == max(y))][0] * 2

            if inspect :
                fig1 = plt.figure(f"GaussianFit_{stage}_{sub_id}")
                plt.hist(
                    temp_p2p, bins = 100, density = True, 
                    alpha = 0.5, label = f"PTP_SW_{chan}",
                    color = palette_bins[i]
                    )
                plt.plot(bins, y, color = palette_fit[i], 
                         label = f"Ex-GaussFit_{chan}")
                plt.axvline(
                    x = max_gaus, color = palette_bins[i], 
                    label = f"Threshold_{chan}", ls = '--')
                plt.xlabel('Values')
                plt.ylabel('Density')
                plt.title('Ex-Gaussian Fit')
                plt.legend()
                plt.show(block=False)
                plt.close(fig1)
            if stage == 3 :
                if max_gaus > 75 :
                    max_gaus = 75
            
            thresh_dic[stage][chan] = max_gaus
            
        if inspect :
            report.add_figure(
                fig=fig1,
                title=f"Ex Gaussian Fit on PTP distrib during stage {stage}",
                image_format="PNG"
                )
    if inspect :
        report.save(
            f"{fig_dir}/Reports/PTP_report_{sub_id}.html", 
            overwrite=True,
            open_browser = False)    
    
    df_clean = pd.concat(
        [df_sw[
            (df_sw.stage == stage) 
            & (df_sw.chan_name == chan) 
            & (df_sw.PTP > thresh_dic[stage][chan])]
            for chan in thresh_dic[stage].keys()
            for stage in thresh_dic.keys() 
            if len(thresh_dic[stage].keys()) > 0
            ]
        )
    
    bigstagelist = []
    for i_st, stage in enumerate(df_clean.stage):
        if stage == 0 and df_clean.night_status.iloc[i_st] == 0 :
            bigstagelist.append('WAKEb')
        elif stage == 0 and df_clean.night_status.iloc[i_st] == 1 :
            bigstagelist.append('WAKEd')
        else :
            bigstagelist.append(stage_int_to_str[stage])
    
    df_clean.insert(df_clean.shape[1], "newstages", bigstagelist)
    
    savename = f"{fig_dir}/slowwaves/hi_cns/SW_{sub_id}_epochs.csv"
    df_clean.to_csv(savename)
    
    
    epochsWakeAfterNight = []
    while orderedMetadata.divBin2.iloc[-1] == "noBin" :
        epochsWakeAfterNight.append(orderedMetadata.n_epoch.iloc[-1])
        orderedMetadata = orderedMetadata.iloc[:-1]
     
    bigstagelist = []
    for i_st, stage in enumerate(orderedMetadata.scoring) :
        if stage == 'WAKE' and orderedMetadata.night_status.iloc[i_st] == "inNight" :
            bigstagelist.append("WAKEd")
        elif stage == 'WAKE' and orderedMetadata.night_status.iloc[i_st] == "outNight":
            bigstagelist.append("WAKEb")
        else :
            bigstagelist.append(stage)
    orderedMetadata.insert(0, "newstage", bigstagelist)
        
    for thisEpoch in epochsWakeAfterNight :
        df_clean = df_clean.loc[df_clean['n_epoch'] != thisEpoch]
    
    thisCount = {}
    for stage in orderedMetadata.newstage.unique() :
        thisCount[stage] = np.count_nonzero(
            orderedMetadata.newstage == stage)/2
    
    for stage in df_clean.newstages.unique() :
        for chan in df_clean.chan_name.unique() :
            n_epoch = orderedMetadata[
                orderedMetadata.newstage == stage].shape[0]
            n_wave = np.count_nonzero(
                (df_clean.newstages == stage)
                & (df_clean.chan_name == chan)
                )
            subidlist.append(sub_id)
            subtypelist.append(subtype)
            sollist.append(thisDemographics['lat.end.N1'].iloc[0])
            tstlist.append(thisDemographics['TST.N1'].iloc[0])
            efflist.append(thisDemographics['eff.N1'].iloc[0])
            wasolist.append(thisDemographics['WASO.N1'].iloc[0])
            melist.append(thisDemographics['ME.N1'].iloc[0])
            genrelist.append(thisDemographics['sexe'].iloc[0])
            agelist.append(thisDemographics['age'].iloc[0])
            esslist.append(thisDemographics['ESS'].iloc[0])
            hyperonilist.append(thisDemographics['hyper.o'].iloc[0])
            blackoutlist.append(thisDemographics['black.o'].iloc[0])
            # soremtotlist.append(thisDemographics['SOREM.tot'].iloc[0])
            dursemtot.append(thisDemographics['dur.som.sem'].iloc[0])
            dursemwe.append(thisDemographics['dur.som.we'].iloc[0])
            ivresse.append(thisDemographics['ivresse'].iloc[0])
            sieste.append(thisDemographics['sieste.reg'].iloc[0])
            
            if np.count_nonzero(orderedMetadata.newstage == "WAKEb") == 0 : 
                duration_WAKEb.append(np.nan)
            else :
                duration_WAKEb.append(thisCount["WAKEb"])
            if np.count_nonzero(orderedMetadata.newstage == "WAKEd") == 0 : 
                duration_WAKEd.append(np.nan)
            else :
                duration_WAKEd.append(thisCount["WAKEd"])
            if np.count_nonzero(orderedMetadata.newstage == "N1") == 0 :
                duration_N1.append(np.nan)
            else :
                duration_N1.append(thisCount["N1"])
            duration_N2.append(thisCount["N2"])
            duration_N3.append(thisCount["N3"])
            duration_REM.append(thisCount["REM"])
            
            stagelist.append(stage)
            chanlist.append(chan)
            countlist.append(n_wave)
            densitylist.append(n_wave / n_epoch)
            ptplist.append(
                df_clean.PTP.loc[
                    (df_clean["newstages"] == stage)
                    & (df_clean["chan_name"] == chan)
                    ].mean()
                )
            frequencylist.append(
                np.mean(1/df_clean.period.loc[
                    (df_clean["newstages"] == stage)
                    & (df_clean["chan_name"] == chan)
                        ])
                )
            dslopelist.append(
                np.nanmean(df_clean.inst_neg_1st_segment_slope.loc[
                    (df_clean["newstages"] == stage)
                    & (df_clean["chan_name"] == chan)
                    ])
                )
            uslopelist.append(
                np.nanmean(df_clean.max_pos_slope_2nd_segment.loc[
                    (df_clean["newstages"] == stage)
                    & (df_clean["chan_name"] == chan)
                    ])
                )

df = pd.DataFrame(
    {
     "sub_id" : subidlist,
     "subtype" : subtypelist,
     "SOL" : sollist,
     "TST" : tstlist,
     "SE" : efflist,
     "WASO" : wasolist,
     "MAI" : melist,
     ""
     "genre" : genrelist,
     "age" : agelist,
     "ess" : esslist,
     "hyperonirisme" : hyperonilist,
     "blackout" : blackoutlist,
     # "nsorem" : soremtotlist,
     "dur_som_tot" : dursemtot,
     "dur_som_we" : dursemwe,
     "ivresse" : ivresse,
     "sieste" : sieste,
     "stage" : stagelist,
     "duration_wakeb" : duration_WAKEb,
     "duration_waked" : duration_WAKEd,
     "duration_N1" : duration_N1,
     "duration_N2" : duration_N2,
     "duration_N3" : duration_N3,
     "duration_REM" : duration_REM,
     "channel" : chanlist,
     "count" : countlist,
     "density" : densitylist,
     "ptp" : ptplist,
     "frequency" : frequencylist,
     "downward_slope" : dslopelist,
     "upward_slope" : uslopelist
     }
    )

df.to_csv(f"{fig_dir}/df_allsw_exgausscrit_nobin_{todaydate}.csv")
