#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:52:30 2023

@author: arthurlecoz

explore_complexity.py

"""
# %% Paths & Variables

from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import mne

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}/Raw"
preproc_dir = f"{root_dir}/Preproc"
fig_dir = f"{root_dir}/Figs"
complexity_dir = os.path.join(root_dir,"Figs", "complexity")

epochs_files = glob(os.path.join(
    preproc_dir, "*PSG1*.fif")
    )
complexity_files = glob(f"{complexity_dir}{os.sep}*_complexity_dic.pkl")

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

# %% Script

this_saving_path = f"{fig_dir}{os.sep}complexity{os.sep}nt1_cns_conform_dfcomplex.csv"

if os.path.exists(this_saving_path) :
    print("Dataframe already exists... Loading it...")
    bigdf = pd.read_csv(this_saving_path)
    del bigdf['Unnamed: 0']
    # analysis_df = pd.read_csv(f"{fig_dir}{os.sep}mean_complexitydf.csv")
    # del analysis_df['Unnamed: 0']
else : 
    list_df = []
    
    subidlist = []
    subtypelist = []
    agelist = []
    genrelist = []
    stagelist = []
    chanlist = []
    
    approxentropylist = []
    sampentropylist = []
    kolmogorovlist = []   
    permdeltalist = []
    permthetalist = []
    permalphalist = []
    permbetalist = []
    permgammalist = []
    
    duration_WAKEb = []
    duration_WAKEd = []
    duration_N1 = []
    duration_N2 = []
    duration_N3 = []
    duration_REM = []
    
    for i_file, file in enumerate(complexity_files) :
        sub_id = file.split("complexity/")[-1].split('_PSG')[0]
        if sub_id.startswith('H') : 
            continue
        print(f"...Processing {sub_id} : {i_file+1}/{len(complexity_files)}")
        
        thisDemo = df_demographics.loc[df_demographics.code == sub_id]
        age = thisDemo.age.iloc[0]
        genre = thisDemo.sexe.iloc[0]
        subtype = thisDemo.diag.iloc[0]
        
        thisDic = pd.read_pickle(file)
        thisEpochs = mne.read_epochs(
            glob(f"{preproc_dir}{os.sep}*{sub_id}*PSG1*.fif")[0]
            )
        metadata = thisEpochs.metadata
        
        thisJointdf = metadata.copy()
        for key in thisDic :
            for i_ch, chan in enumerate(["F3", "C3", "O1"]) :
                thisJointdf.insert(
                    14, f"{key}_{chan}", thisDic[key][:, i_ch]
                    )
        # thisJointdf.to_csv(f"{fig_dir}{os.sep}{sub_id}_complexitydf.csv")
        list_df.append(thisJointdf)

        orderedMetadata = thisJointdf.sort_values("n_epoch")
        
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
        
        # hypno = np.asarray(epochs.metadata.scoring)
        
        thisCount = {}
        for stage in orderedMetadata.newstage.unique() :
            thisCount[stage] = np.count_nonzero(
                orderedMetadata.newstage == stage)/2
        
        thisMeandf = orderedMetadata[[
            'newstage', 'subid', 'age', 'sex',
            'Permutation_Entropy_gamma_O1',
            'Permutation_Entropy_gamma_C3', 'Permutation_Entropy_gamma_F3',
            'Permutation_Entropy_beta_O1', 'Permutation_Entropy_beta_C3',
            'Permutation_Entropy_beta_F3', 'Permutation_Entropy_alpha_O1',
            'Permutation_Entropy_alpha_C3', 'Permutation_Entropy_alpha_F3',
            'Permutation_Entropy_theta_O1', 'Permutation_Entropy_theta_C3',
            'Permutation_Entropy_theta_F3', 'Permutation_Entropy_delta_O1',
            'Permutation_Entropy_delta_C3', 'Permutation_Entropy_delta_F3',
            'Sample_Entropy_O1', 'Sample_Entropy_C3', 'Sample_Entropy_F3',
            'Approximative_Entropy_O1', 'Approximative_Entropy_C3',
            'Approximative_Entropy_F3', 'Kolmogorov_O1', 'Kolmogorov_C3',
            'Kolmogorov_F3']].groupby(
                ['newstage', 'subid', 'age', 'sex'], as_index = False
                ).mean()
                
        for i_stage, stage in enumerate(thisMeandf.newstage.unique()) :
            for i_ch, channel in enumerate(["F3", "C3", "O1"]):
                subidlist.append(sub_id)
                subtypelist.append(subtype)
                agelist.append(age)
                genrelist.append(genre)
                stagelist.append(stage)
                chanlist.append(channel)
                
                approxentropylist.append(
                    thisMeandf[f"Approximative_Entropy_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                sampentropylist.append(
                    thisMeandf[f"Sample_Entropy_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                kolmogorovlist.append(
                    thisMeandf[f"Kolmogorov_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )   
                permdeltalist.append(
                    thisMeandf[f"Permutation_Entropy_delta_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                permthetalist.append(
                    thisMeandf[f"Permutation_Entropy_theta_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                permalphalist.append(
                    thisMeandf[f"Permutation_Entropy_alpha_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                permbetalist.append(
                    thisMeandf[f"Permutation_Entropy_beta_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                permgammalist.append(
                    thisMeandf[f"Permutation_Entropy_gamma_{channel}"].loc[
                        thisMeandf.newstage == stage].iloc[0]
                    )
                
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
    
    df_complexfeats = pd.DataFrame({
        "sub_id" : subidlist,
        "subtype" : subtypelist,
        "age" : agelist,
        "genre" : genrelist,
        "stage" : stagelist,
        "channel" : chanlist,
        "approximative_entropy" : approxentropylist,
        "sample_entropy" : sampentropylist,
        "kolmogorov" : kolmogorovlist,   
        "permutation_entropy_delta" : permdeltalist,
        "permutation_entropy_theta" : permthetalist,
        "permutation_entropy_alpha" : permalphalist,
        "permutation_entropy_beta" : permbetalist,
        "permutation_entropy_gamma" : permgammalist,
        "duration_wakeb" : duration_WAKEb,
        "duration_waked" : duration_WAKEd,
        "duration_N1" : duration_N1,
        "duration_N2" : duration_N2,
        "duration_N3" : duration_N3,
        "duration_REM" : duration_REM,
        })
    df_complexfeats.to_csv(
        os.path.join(complexity_dir, "conform_dfcomplex.csv")
        )
    
    bigdf = pd.concat(list_df)   
    
    # %% Results Stats

"""
KOLMOGOROV

channel = C3, stage = WAKEb:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.018312 0.00872 618  -2.101  0.0361

channel = F3, stage = WAKEb:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.012553 0.00872 618  -1.440  0.1503

channel = O1, stage = WAKEb:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.018930 0.00872 618  -2.172  0.0303
 
 channel = C3, stage = REM:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.015981 0.00872 618  -1.833  0.0672

channel = F3, stage = REM:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.013333 0.00872 618  -1.530  0.1266

channel = O1, stage = REM:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.019093 0.00872 618  -2.190  0.0289
 
 channel = C3, stage = N3:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.018691 0.00872 618  -2.144  0.0324

channel = F3, stage = N3:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.021468 0.00872 618  -2.463  0.0141

channel = O1, stage = N3:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.019221 0.00872 618  -2.205  0.0278
 
 channel = C3, stage = N2:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.020141 0.00872 618  -2.311  0.0212

channel = F3, stage = N2:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.020602 0.00872 618  -2.364  0.0184

channel = O1, stage = N2:
 contrast  estimate      SE  df t.ratio p.value
 C1 - N1  -0.019715 0.00872 618  -2.262  0.0240
 
 fdr_correction([0.0361, 0.1503, 0.0303])
 Out[8]: (array([False, False, False]), 
          array([0.05415, 0.1503 , 0.05415]))

 fdr_correction([0.0672, 0.1266, 0.0289])
 Out[9]: (array([False, False, False]), 
          array([0.1008, 0.1266, 0.0867]))

 fdr_correction([0.0324, 0.0141, 0.0278])
 Out[10]: (array([ True,  True,  True]), 
           array([0.0324, 0.0324, 0.0324]))

 fdr_correction([0.0212, 0.0184, 0.0240])
 Out[11]: (array([ True,  True,  True]), 
           array([0.024, 0.024, 0.024]))

"""
    
# %% Plots

df_complexfeats = bigdf.copy()

x = "stage"
order = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["C1", "N1"]
channels = ['F3', 'C3', 'O1']
y = "sample_entropy"

complexity_features = ['approximative_entropy','sample_entropy', 'kolmogorov']
palette = ["#8d99ae", "#d00000"]

fig, ax = plt.subplots(
    nrows=3, ncols=3, sharex=False, sharey=True, 
    figsize = (12, 16), layout = "tight")
for i_cf, complexity_feature in enumerate(complexity_features) : 
    for i_ch, chan in enumerate(channels):
        sns.violinplot(
            x = x, 
            order = order, 
            y = y, 
            hue = hue, 
            hue_order = hue_order,
            data = df_complexfeats.loc[df_complexfeats['channel'] == chan], 
            palette = palette,
            inner = "quartile",
            linecolor = "white",
            cut = 1, 
            ax = ax[i_ch][i_cf]
            )
        
        ax[0][i_cf].set_title(f"{complexity_feature}")
        ax[i_ch][i_cf].legend_ = None
            
    plt.show(block = False)
plt.savefig(
    f"{fig_dir}/complexity/nt1_cns_complex_feats.png", 
    dpi = 300
    )
    

# %% LME

this_df = bigdf.loc[bigdf.stage!="WAKEb"]

foi = ["Stage", "Channel", "ß (NT1 vs HS)", "p_val"]
dic = {f : [] for f in foi}

for stage in this_df.stage.unique():
    for channel in this_df.channel.unique() :
        model_formula = f'permutation_entropy_theta ~ age + C(genre) + duration_waked + duration_N1 + duration_N2 + duration_N3 + duration_REM + C(subtype, Treatment("C1")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
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

# %% Kolmogorov

x = "stage"
order = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
hue = "subtype"
hue_order = ["C1", "N1"]
# channels = ['F3', 'C3', 'O1']
y = "permutation_entropy_beta"

# palette = ["#000000", "#d00000"]

fig, ax = plt.subplots(
    nrows=1, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 4), layout = "tight")
sns.violinplot(
    x = x, 
    order = order, 
    y = y, 
    hue = hue, 
    hue_order = hue_order,
    data = df_complexfeats, 
    palette = palette,
    inner = "quartile",
    linecolor = "white",
    cut = 1, 
    ax = ax
    )
    
ax.set_title(f"{y}")
# ax.legend_ = None
        
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/complexity/nt1_cns_{y}_avchan.png", 
    dpi = 300
    )

# %% Kolmogorov

for feat in df_complexfeats.columns[6:-6]:

    x = "stage"
    order = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
    hue = "subtype"
    hue_order = ["C1", "N1"]
    channels = ['F3', 'C3', 'O1']
    y = feat
    
    # palette = ["#000000", "#d00000"]
    
    fig, ax = plt.subplots(
        nrows=3, ncols=1, sharex=False, sharey=True, 
        figsize = (6, 16), layout = "tight")
    for i_ch, chan in enumerate(channels):
        sns.violinplot(
            x = x, 
            order = order, 
            y = y, 
            hue = hue, 
            hue_order = hue_order,
            data = df_complexfeats.loc[df_complexfeats['channel'] == chan], 
            palette = palette,
            inner = "quartile",
            linecolor = "white",
            cut = 1, 
            ax = ax[i_ch]
            )
        
        ax[i_ch].set_title(f"{chan}")
        ax[0].legend_ = None
        ax[1].legend_ = None
            
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/complexity/nt1_cns_{y}.png", 
        dpi = 300
        )

# %% compute difference subtype - mean CTL

# analysis_df = df_complexfeats[
#     ['subtype', 'stage', 'channel',
#      'approximative_entropy', 'sample_entropy', 'kolmogorov',
#      'permutation_entropy_delta', 'permutation_entropy_theta',
#      'permutation_entropy_alpha', 'permutation_entropy_beta',
#      'permutation_entropy_gamma']
#     ].groupby(['subtype', 'stage', 'channel'], as_index = False).mean()
# analysis_df = analysis_df.loc[analysis_df["subtype"] == "C1"]

# this_saving_diff_path = f"{fig_dir}{os.sep}diff_complexity_nt1_cns.csv"

# if os.path.exists(this_saving_diff_path) :
#     df_differences = pd.read_csv(this_saving_diff_path)
# else : 
#     stages_oi = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
#     channels = ["F3", "C3", "O1"]
#     complexity_features = ['approximative_entropy','sample_entropy', 'kolmogorov']
    
#     subidlist = []
#     stagelist = []
#     channellist = []
#     kolmogorov = []
#     sample_entropy = []
#     approximative_entropy = []
    
#     nt1_patients = np.asarray(df_complexfeats.sub_id.loc[df_complexfeats.subtype == "N1"].unique())
    
#     # dic_complex = {}
#     # for complexity_feature in  :
#     #     for stage in stages_oi :
#     #         for channel in channels :
#     #             dic_complex[complexity_feature][stage][channel] = analysis_df[complexity_feature].loc[
#     #                 (analysis_df[stage] == stage) 
#     #                 & (analysis_df[channel] == channel)
#     #                 ].iloc[0]    
    
#     # mean_kolmo_ctl = {stage : {channel : {}} for stage in stages_oi}
#     # mean_approx_ctl = {stage : {channel : {}} for stage in stages_oi}
#     # mean_sample_ctl = {stage : {channel : {}} for stage in stages_oi}
    
#     # for stage in stages_oi :
#     #     for chan in channels :
#     #         mean_kolmo_ctl[stage][chan] = analysis_df[f"Kolmogorov"].loc[
#     #             (analysis_df.groupID == 'C1') 
#     #             & (analysis_df.stage == stage)
#     #             & (analysis_df.channel == channel)
#     #             ].mean()
    
#     for sub_id in nt1_patients :
#         for stage in stages_oi :
#             for chan in channels :
#                 if df_complexfeats.loc[
#                     (df_complexfeats.sub_id == sub_id)
#                     & (df_complexfeats.stage == stage)
#                     ].empty : continue
                
#                 subidlist.append(sub_id)
#                 stagelist.append(stage)
#                 channellist.append(chan)
                
#                 kolmogorov.append(
#                     df_complexfeats.kolmogorov.loc[
#                         (df_complexfeats["sub_id"] == sub_id)
#                         & (df_complexfeats["stage"] == stage)
#                         & (df_complexfeats["channel"] == channel)
#                         ].iloc[0] - analysis_df.kolmogorov.loc[
#                             (analysis_df.stage == stage)
#                             & (analysis_df.channel == channel)
#                             ].iloc[0]
#                     )
#                 sample_entropy.append(
#                     df_complexfeats.sample_entropy.loc[
#                         (df_complexfeats["sub_id"] == sub_id)
#                         & (df_complexfeats["stage"] == stage)
#                         & (df_complexfeats["channel"] == channel)
#                         ].iloc[0] - analysis_df.sample_entropy.loc[
#                             (analysis_df.stage == stage)
#                             & (analysis_df.channel == channel)
#                             ].iloc[0]
#                     )
#                 approximative_entropy.append(
#                     df_complexfeats.approximative_entropy.loc[
#                         (df_complexfeats["sub_id"] == sub_id)
#                         & (df_complexfeats["stage"] == stage)
#                         & (df_complexfeats["channel"] == channel)
#                         ].iloc[0] - analysis_df.approximative_entropy.loc[
#                             (analysis_df.stage == stage)
#                             & (analysis_df.channel == channel)
#                             ].iloc[0]
#                     )
    
#     df_differences = pd.DataFrame({
#         "sub_id" : subidlist,
#         "stage" : stagelist,
#         "channel" : channellist,
#         "kolmogorov" : kolmogorov,
#         "approximative_entropy" : approximative_entropy,
#         "sample_entropy" : sample_entropy
#         })
    
#     df_differences.to_csv(this_saving_diff_path)

# %% plot differences kolmo

# x = "stage"
# order = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
# hue = "channel"
# hue_order = ['F3', 'C3', 'O1']
# channels = ['F3', 'C3', 'O1']
# y = "kolmogorov"

# # palette = ["#d00000", "#faa307"]

# fig, ax = plt.subplots(
#     nrows=3, ncols=1, sharex=False, sharey=True, 
#     figsize = (5, 16), layout = "tight")
# # for i_ch, chan in enumerate(channels):
#     sns.violinplot(
#         x = x, order = order, y = y, hue = hue, hue_order = hue_order,
#         data = df_differences, 
#         #palette = palette, 
#         inner = "quartile" , cut = 1
#         )
#     ax[i_ch].axhline(y = 0, c = 'grey', ls = "dotted", alpha = .5)
    
#     ax[i_ch].set_title("")
#     ax[i_ch].set_ylim(-.17, .17)
#     for i in range(3) :
#         ax[i].set_xlabel("")
#         if i == 2 :
#             ax[i].set_xticks(
#                 ticks = np.arange(0, 4, 1), 
#                 labels = ["" for i in np.arange(0, 4, 1)]
#                 )
#         else : 
#             ax[i].set_xticks(
#                 ticks = [], 
#                 labels = []
#                 )
#         ax[i].set_yticks(
#             ticks = np.arange(-.15, .25, .10), 
#             labels = ["" for i in np.arange(-.15, .25, .10)]
#             )
        
#         ax[i].set_ylabel("")
#     ax[i_ch].legend_ = None
        
#     plt.show(block = False)
#     plt.savefig(
#         f"{fig_dir}/nolegend_differences_kolmogorov_subplot_subtype_stage.png", 
#         dpi = 300
#         )

