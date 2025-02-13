#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:44:41 2024

@author: arthurlecoz

ALC_pcompute_bandpower.py

"""
# %% Paths
import os, numpy as np, pandas as pd
from glob import glob
import statsmodels.formula.api as smf

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}/Raw"
preproc_dir = f"{root_dir}/Preproc/"
fig_dir = f"{root_dir}/Figs/"

aperiodic_files = glob(os.path.join(
    fig_dir, "fooof", "*_aperiodic_psd_nolowess.pickle"
    ))

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

subtypes = ["C1", "N1", "HI"]
# subtypes = ["C1", "HI"]
channels = ["F3", "C3", "O1"]
stages = ['WAKE', 'N1', 'N2', 'N3', 'REM']
palette = ["#8d99ae", "#d00000", "#ffb703"]
# palette = ["#8d99ae", "#ffb703"]
freqs = np.linspace(0.5, 40, 159)
bands = {
    "delta" : (1,  4),
    "theta" : (4 ,  8),
    "alpha" : (8 , 12),
    "sigma" : (11, 16),
    "beta"  : (16, 30)
    }


# %% Bandpower

basic_params = [
    "sub_id", "subtype", "stage", "channel", "age", "genre",
    # "dur_WAKEb", "dur_WAKEd", "dur_N1", "dur_N2", "dur_N3", "dur_REM",
    "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
    "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
    ]

big_dic = {param : [] for param in basic_params}

thisSavingPath = os.path.join(
    fig_dir, "bandpower_df.csv"
    )

for i, file in enumerate(aperiodic_files) :
    this_dic = pd.read_pickle(file)
    sub_id = file.split('fooof/')[-1].split('_aper')[0]
    if sub_id == 'N1_016' : continue
    subtype = sub_id[:2]
    # if len(subtypes)<3 :
    #     if subtype == "N1" : continue
    
    print(f"Processing : {sub_id}... [{i+1} / {len(aperiodic_files)}]")
    
    # for stage in stages:
    #     for channel in channels:
    #         if len(this_dic[stage][channel]) < 1 :
    #             big_dic[subtype][stage][channel].append(
    #                 np.nan * np.empty(159))
    #         else : 
    #             big_dic[subtype][stage][channel].append(
    #                 10 * np.log10(this_dic[stage][channel][0]))
                
                
    thisDemo = df_demographics.loc[df_demographics.code == sub_id]
    age = thisDemo.age.iloc[0]
    genre = thisDemo.sexe.iloc[0]
    
    for i_st, stage in enumerate(stages) :
        for i_ch, chan in enumerate(channels) : 
            if not len(this_dic[stage][chan]) : continue
            thischan_power = this_dic[stage][chan][0]
            thisabs_power = sum([
                np.mean(thischan_power[
                    np.logical_and(freqs >= borders[0], freqs <= borders[1])
                    ], axis = 0) for band, borders in bands.items()
                ])
            big_dic["sub_id"].append(sub_id)
            big_dic["age"].append(age)
            big_dic["genre"].append(genre)
            big_dic["subtype"].append(subtype)
            big_dic["channel"].append(chan)
            big_dic["stage"].append(stage)
            
            # if np.count_nonzero(metadata.newstage == "WAKEb") == 0 : 
            #     big_dic["dur_WAKEb"].append(np.nan)
            # else :
            #     big_dic["dur_WAKEb"].append(thisCount["WAKEb"])
            # if np.count_nonzero(metadata.newstage == "WAKEd") == 0 : 
            #     big_dic["dur_WAKEd"].append(np.nan)
            # else :
            #     big_dic["dur_WAKEd"].append(thisCount["WAKEd"])
            # if np.count_nonzero(metadata.newstage == "N1") == 0 :
            #     big_dic["dur_N1"].append(np.nan)
            # else :
            #     big_dic["dur_N1"].append(thisCount["N1"])
            # big_dic["dur_N2"].append(thisCount["N2"])
            # big_dic["dur_N3"].append(thisCount["N3"])
            # big_dic["dur_REM"].append(thisCount["REM"])
            
            for band, borders in bands.items() :
                abs_bandpower = np.mean(thischan_power[
                    np.logical_and(freqs >= borders[0], freqs <= borders[1])
                    ], axis = 0)
                rel_bandpower = abs_bandpower/thisabs_power
                    
                big_dic[f"abs_{band}"].append(10 * np.log(abs_bandpower))
                big_dic[f"rel_{band}"].append(rel_bandpower)
     
df = pd.DataFrame.from_dict(big_dic)
# for col in df.columns[-4:] :
#     df[col] = 10 * np.log10(df[col])
df.to_csv(thisSavingPath)

# %% LME - 

including = 'N1'

temp_df = df.loc[df.subtype.isin(['C1', including])]
interests = [
    'abs_delta','abs_theta', 'abs_alpha', 'abs_sigma', 'abs_beta', 
    'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta'
    ]

highlight = f'C(subtype, Treatment("C1"))[T.{including}]'

for feature in interests :
    for stage in stages:
        if stage == 'WAKE': continue
        for channel in channels :
            model_formula = f'{feature} ~ age + C(genre) + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
            model = smf.mixedlm(model_formula, temp_df, groups=temp_df['sub_id'], missing='drop')
            model_result = model.fit()
            
            if model_result.pvalues[highlight] < .05 :
                print(f"\n\nSomething in {stage}, at {channel} for {feature}")
