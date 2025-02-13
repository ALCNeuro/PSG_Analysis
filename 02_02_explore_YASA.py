#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:10:42 2024

@author: arthurlecoz

02_02_explore_yasa.py

hypnodensities matrices shows :

    The percentage to score the different sleep stages 
    when expert score a given sleep stage

"""
# %% paths & packages

from glob import glob

import os
import mne

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confmat
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager 

# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')
prop = font_manager.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

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
stage_strtoint = {
    "WAKEd" : 0, "WAKEb" : 0, "N1" : 1, "N2" : 2, "N3" : 3, "REM" : 5
    }
int_to_str = {0:"WAKE", 1:"N1", 2:"N2", 3:"N3", 5:"REM"}

channels = ['F3', 'C3', 'O1']
subtypes = ['N1', 'C1', 'HI']

basic_params = [
    "sub_id", "age", "genre", "subtype", "kappa",
    "duration_wakeb", "duration_waked", "duration_N1", 
    "duration_N2", "duration_N3", "duration_REM",
    "stage", "channel", "entropy", "dKL",
    "p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM",
    "p_agree_WAKE", "p_agree_N1", "p_agree_N2", "p_agree_N3", "p_agree_REM"
    ]

params_hypnodensities = [
    "sub_id", "age", "genre", "subtype", 
    "duration_wakeb", "duration_waked", "duration_N1", 
    "duration_N2", "duration_N3", "duration_REM",
    "stage", "channel", "confidence","kappa",
    "p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM",
    ]

tradi_stages = ["WAKE", "N1", "N2", "N3", "REM"]

palette = ["#5e6472", "#faa307"]

# %% script 

big_dic = {param : [] for param in basic_params}
dic_hypnodensities = {param : [] for param in params_hypnodensities}

files = glob(f"{raw_dir}{os.sep}PSG4_Hypnogram_Export*PSG1.txt")

chan_stage_df = os.path.join(
    fig_dir, "yasa", "df_yasa_hypnodens_all.csv"
    )
thisSavingPath = os.path.join(
    fig_dir, "yasa", "df_yasa_entropy_dkl_noagreement_compumedics_sleepdur.csv"
    )
thisHypnodensitiesSavingPath = os.path.join(
    fig_dir, "yasa", "df_hypnodensities_yasastats_compumedics_sleepdur.csv"
    )

if (os.path.exists(thisSavingPath) 
    and os.path.exists(thisHypnodensitiesSavingPath)) :
    
        small_df = pd.read_csv(thisSavingPath)
        df_hypnodensities = pd.read_csv(thisHypnodensitiesSavingPath)
else :        
    for i_file, file in enumerate(files) :
        
        sub_id = file.split('Export_')[-1][:-4]
        exam = sub_id[-4:]
        code = sub_id[:-5]
        subtype = code[:2]
        
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
        for stage in metadata.newstage.unique() :
            thisCount[stage] = np.count_nonzero(
                metadata.newstage == stage)/2
            
        # thisCount = {}
        # for stage in orderedMetadata.newstage.unique() :
        #     if stage == "WAKEb" :
        #         thisCount[stage] = np.count_nonzero(
        #             orderedMetadata.newstage == stage)/2
        #     elif stage == "WAKEd" :
        #         thisCount[stage] = float(thisDemo['WASO.N1'].iloc[0].replace(",","."))
        #     elif stage == "N1" :
        #         thisCount[stage] = float(thisDemo['NREM1.N1'].iloc[0].replace(",","."))
        #     elif stage == "N2" :
        #         thisCount[stage] = float(thisDemo['NREM2.N1'].iloc[0].replace(",","."))
        #     elif stage == "N3" :
        #         thisCount[stage] = float(thisDemo['NREM3.N1'].iloc[0].replace(",","."))
        #     elif stage == "REM" :
        #         thisCount[stage] = float(thisDemo['REM.N1'].iloc[0].replace(",","."))
        
        sorted_hypnogram = metadata.sort_values('n_epoch').scoring.to_numpy()
        kept_epochs = metadata.sort_values('n_epoch').n_epoch.to_numpy()
        
        df_yasa_path = glob(os.path.join(
            fig_dir, "yasa", f"{code}_hypnodensity_yasa.csv"
            ))[0]
        df_yasa = pd.read_csv(df_yasa_path)
        del df_yasa['Unnamed: 0']
        df_yasa["our_scoring"] = [int_to_str[st] for st 
                                  in df_yasa.int_stage.to_numpy()]
        
        hypnodensity = df_yasa[["W", "N1", "N2", "N3", "R"]].to_numpy()    
        hypnopred = df_yasa['int_stage'].to_numpy()
        str_pred = np.array([int_to_str[stage] for stage in hypnopred])
        yasa_hypno = df_yasa.scorred_stage.to_numpy()
        
        k = kappa(sorted_hypnogram, str_pred[kept_epochs])
        this_cm = confmat(
            sorted_hypnogram, str_pred[kept_epochs], normalize = "true"
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
                big_dic['kappa'].append(k)
                
                if np.count_nonzero(orderedMetadata.newstage == "WAKEb") == 0 : 
                    big_dic['duration_wakeb'].append(0)
                else :
                    big_dic['duration_wakeb'].append(thisCount["WAKEb"])
                if np.count_nonzero(orderedMetadata.newstage == "WAKEd") == 0 : 
                    big_dic['duration_waked'].append(0)
                else :
                    big_dic['duration_waked'].append(thisCount["WAKEd"])
                if np.count_nonzero(orderedMetadata.newstage == "N1") == 0 :
                    big_dic['duration_N1'].append(0)
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
                
                # agreement stages x expert
                this_agreepred = thishypnopred[
                    thishypnopred == stage_strtoint[stage]
                    ]
                this_agreedensity = thishypnodensity[
                    thishypnopred == stage_strtoint[stage]
                    ]
                
                big_dic["entropy"].append(thisentropy)
                big_dic["dKL"].append(thisdkl)
                
                thisproba = np.mean(thishypnodensity, axis = 0)
                this_agreeproba = np.mean(this_agreedensity, axis = 0)
                # "p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM"
                for i_st, st in enumerate(tradi_stages) :
                    big_dic[f'p_{st}'].append(thisproba[i_st])
                    big_dic[f'p_agree_{st}'].append(this_agreeproba[i_st])
        
        df_yasa = df_yasa.loc[df_yasa.epoch.isin(kept_epochs)]
        df_yasa.rename(columns={"R" : "REM", "W" : "WAKE"}, inplace = True)
        yasa_stats = df_yasa[
            ['our_scoring', 'confidence', 'WAKE', 'N1', 'N2', 'N3','REM']
            ].groupby('our_scoring', as_index = False).mean()
        
        for stage in tradi_stages :
            
            sub_stats = yasa_stats.loc[yasa_stats.our_scoring == stage]
            
            dic_hypnodensities['sub_id'].append(code)
            dic_hypnodensities['subtype'].append(subtype)
            dic_hypnodensities['age'].append(age)
            dic_hypnodensities['genre'].append(genre)
            dic_hypnodensities['stage'].append(stage)
            dic_hypnodensities['channel'].append(channel)
            dic_hypnodensities['confidence'].append(sub_stats.confidence.iloc[0])
            dic_hypnodensities['kappa'].append(k)
            
            if np.count_nonzero(orderedMetadata.newstage == "WAKEb") == 0 : 
                dic_hypnodensities['duration_wakeb'].append(0)
            else :
                dic_hypnodensities['duration_wakeb'].append(thisCount["WAKEb"])
            if np.count_nonzero(orderedMetadata.newstage == "WAKEd") == 0 : 
                dic_hypnodensities['duration_waked'].append(0)
            else :
                dic_hypnodensities['duration_waked'].append(thisCount["WAKEd"])
            if np.count_nonzero(orderedMetadata.newstage == "N1") == 0 :
                dic_hypnodensities['duration_N1'].append(0)
            else :
                dic_hypnodensities['duration_N1'].append(thisCount["N1"])
            dic_hypnodensities['duration_N2'].append(thisCount["N2"])
            dic_hypnodensities['duration_N3'].append(thisCount["N3"])
            dic_hypnodensities['duration_REM'].append(thisCount["REM"])
            
            for i_st, st in enumerate(tradi_stages) :
                dic_hypnodensities[f'p_{st}'].append(sub_stats[st].iloc[0])
        
    df = pd.DataFrame.from_dict(big_dic)
    df.to_csv(chan_stage_df)
    small_df = df[
        ['sub_id', 'age', 'genre', 'subtype', 'stage', 'entropy', 'kappa',
         'duration_wakeb', 'duration_waked', 'duration_N1', 
         'duration_N2','duration_N3', 'duration_REM', 'dKL', 
         'p_WAKE', 'p_N1', 'p_N2', 'p_N3', 'p_REM',
         'p_agree_WAKE', 'p_agree_N1', 'p_agree_N2', 'p_agree_N3', 'p_agree_REM',
         ]].groupby(
               ['sub_id', 'age', 'genre', 'subtype', 'stage'],
               as_index = False
               ).mean()
    small_df.to_csv(thisSavingPath)
    
    df_hypnodensities = pd.DataFrame.from_dict(dic_hypnodensities)
    df_hypnodensities.to_csv(thisHypnodensitiesSavingPath)


# %% kappa

df_k = small_df[['sub_id', 'age', 'genre', 'subtype', 'kappa',
       'duration_wakeb', 'duration_waked', 'duration_N1', 'duration_N2',
       'duration_N3', 'duration_REM']].groupby(
           ['sub_id', 'age', 'genre', 'subtype'], as_index = False
           ).mean()
df_k.to_csv(os.path.join(
    fig_dir, 'yasa', 'hi_cns', 'kappa_df.csv'
    ))

# %% Kappa plot
           
subdf = df_k.loc[df_k.subtype != 'N1']
subtypes = ["C1", "HI"]

fig, ax = plt.subplots(figsize = (5, 8))
sns.violinplot(
    data = subdf, 
    hue = "subtype", 
    y = "kappa", 
    ax = ax, 
    fill = True,
    linecolor = "white",
    inner = None,
    palette = palette,
    gap = .05,
    alpha=.2,
    legend = None
    )
sns.pointplot(
    data = subdf, 
    hue = "subtype", 
    y = "kappa", 
    ax = ax, 
    # fill = True,
    # linewidth = 2,
    # inner = None,
    palette = palette,
    dodge = .4,
    capsize=.05,
    errorbar="se",
    linestyle="none",
    legend = None
    )
sns.stripplot(
    data = subdf,
    hue = "subtype", 
    y = "kappa", 
    ax = ax,
    palette = palette,
    size = 5,
    legend = None,
    dodge = True,
    alpha = .5
    )

sns.despine(bottom = True)

ax.set_yticks(np.arange(0, 1.2, 0.2), labels = np.round(np.arange(0, 1.2, 0.2), 1), font = font)
ax.set_ylim(0, 1)
ax.set_ylabel("Cohen's Kappa", fontsize = 20, font = bold_font)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    labelbottom=False
    )
ax.tick_params(axis = 'y', labelsize=16)

# title = """
# <Cohen's Kappa> : 
# <Expert Scorers> vs <YASA> 
# """
# fig_text(
#    0.04, .94,
#    title,
#    fontsize=20,
#    ha='left', va='center',
#    color="k", font=font,
#    highlight_textprops=[
#       {'font': bold_font},
#       {'font': bold_font},
#       {'font': bold_font},
#    ],
#    fig=fig
# )
plt.tight_layout(pad = .8)

plt.savefig(os.path.join(
    fig_dir, "yasa", "all_cohen_kappa_comparison_nolegend.png"
    ))

# %% plot

subtypes = ["C1", "N1"]
order = ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]

fig, ax = plt.subplots()

sns.boxplot(
    data = small_df, 
    x = "stage", 
    order = order, 
    y = "entropy", 
    hue = "subtype",
    hue_order = subtypes,
    ax = ax
    )

# %% Plotting

# Load your dataset
# df = pd.read_csv('path_to_your_csv_file.csv')

# Define the sleep stages and subtypes
stages = ['WAKEd', 'N1', 'N2', 'N3', 'REM']
prob_columns = ['p_agree_WAKE', 'p_agree_N1', 'p_agree_N2', 'p_agree_N3', 'p_agree_REM']

# Extract unique subtypes
subtypes = small_df['subtype'].unique()

# Create a plot for each subtype
for subtype in subtypes:
    # Filter the data for the current subtype
    subtype_data = small_df[small_df['subtype'] == subtype]
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((len(stages), len(prob_columns)))
    
    # Populate the confusion matrix
    for i, stage in enumerate(stages):
        stage_data = subtype_data[subtype_data['stage'] == stage]
        if not stage_data.empty:
            means = stage_data[prob_columns].mean(axis=0).values
            confusion_matrix[i, :] = means# * 100  # Convert to percentage

    # Create the heatmap without automatic annotations
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        confusion_matrix, 
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=prob_columns,
        yticklabels=stages,
        vmin=0,
        vmax = 1
        )
    
    # Manually add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            color = 'white' if confusion_matrix[i, j] > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, f"{confusion_matrix[i, j]:.2f}",
                    ha='center', va='center', color=color, fontsize=24, font = font)
    # Set font for the colorbar
    cbar = ax.collections[0].colorbar
    # cbar.ax.yaxis.set_tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)
    cbar.ax.tick_params(axis = 'y', labelsize=16)
    
    plt.yticks(
        ticks = [.5,1.5,2.5,3.5,4.5], 
        labels = ["WAKE", "N1", "N2", "N3", "REM"], 
        font = font, 
        fontsize = 16
        )
    plt.xticks(
        ticks = [.5,1.5,2.5,3.5,4.5], 
        labels = ["WAKE", "N1", "N2", "N3", "REM"], 
        font = font, 
        fontsize = 16
        )
    
    # plt.title(f'Percentage Hypnodensity Matrix for {subtype}')
    plt.xlabel("YASA's Probabilities", font = bold_font, fontsize = 20)
    plt.ylabel('Expert Scoring', font = bold_font, fontsize = 20)
    plt.show()
    plt.savefig(os.path.join(
        fig_dir, "yasa", f"yasa_confusion_matrices_keptepochs_matrices_{subtype}.png"
        ))
    
# %% hypnodensities
    
# stages = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
prob_columns = ['p_WAKE', 'p_N1', 'p_N2', 'p_N3', 'p_REM']

# Extract unique subtypes
subtypes = df_hypnodensities['subtype'].unique()

# Create a plot for each subtype
for subtype in subtypes:
    # Filter the data for the current subtype
    subtype_data = df_hypnodensities[df_hypnodensities['subtype'] == subtype]
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((len(tradi_stages), len(prob_columns)))
    
    # Populate the confusion matrix
    for i, stage in enumerate(tradi_stages):
        stage_data = subtype_data[subtype_data['stage'] == stage]
        if not stage_data.empty:
            # total = stage_data[prob_columns].sum().sum()  # Total sum of all probabilities for normalization
            means = stage_data[prob_columns].mean(axis=0).values
            confusion_matrix[i, :] = means# * 100  # Convert to percentage

    # Create the heatmap without automatic annotations
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=False, fmt=".2f", cmap="Blues",
                     xticklabels=prob_columns, yticklabels=tradi_stages, vmin=0, vmax=1)
    
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            color = 'white' if confusion_matrix[i, j] > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, f"{confusion_matrix[i, j]:.2f}",
                    ha='center', va='center', color=color, fontsize=24, font = font)
    # Set font for the colorbar
    cbar = ax.collections[0].colorbar
    # cbar.ax.yaxis.set_tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)
        
    cbar.ax.tick_params(axis = 'y', labelsize=16)

    plt.yticks(
        ticks = [.5,1.5,2.5,3.5,4.5], 
        labels = ["WAKE", "N1", "N2", "N3", "REM"], 
        font = font, 
        fontsize = 16
        )
    plt.xticks(
        ticks = [.5,1.5,2.5,3.5,4.5], 
        labels = ["WAKE", "N1", "N2", "N3", "REM"], 
        font = font, 
        fontsize = 16
        )
    
    # plt.title(f'Percentage Hypnodensity Matrix for {subtype}')
    plt.xlabel("YASA's probabilities", font = bold_font, fontsize = 20)
    plt.ylabel("YASA's Scoring", font = bold_font, fontsize = 20)
    plt.show()
    plt.savefig(os.path.join(
        fig_dir, "yasa", f"yasa_hypnodensities_matrices_{subtype}.png"
        ))
    
# %% Plot Raph's 

df_raph = pd.read_csv('/Volumes/DDE_ALC/PhD/NT1_HI/PSG/Figs/dataframe_mean.csv')

stages = ["preN", "inN", "1", "2", "3", "5"]
prob_columns = ['pb_0', 'pb_1', 'pb_2', 'pb_3', 'pb_5']

# Extract unique subtypes
subtypes = df_raph['group'].unique()

# Create a plot for each subtype
for subtype in subtypes:
    # Filter the data for the current subtype
    subtype_data = df_raph[df_raph['group'] == subtype]
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((len(stages), len(prob_columns)))
    
    # Populate the confusion matrix
    for i, stage in enumerate(stages):
        stage_data = subtype_data[subtype_data['stade_expert_2'] == stage]
        if not stage_data.empty:
            # total = stage_data[prob_columns].sum().sum()  # Total sum of all probabilities for normalization
            means = stage_data[prob_columns].mean(axis=0).values
            confusion_matrix[i, :] = means# * 100  # Convert to percentage

    # Create the heatmap without automatic annotations
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=False, fmt=".2f", cmap="Blues",
                     xticklabels=prob_columns, yticklabels=stages)
    
    # Manually add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{confusion_matrix[i, j]:.2f}",
                    ha='center', va='center', color='black', fontsize = "x-large")
    
    plt.title(f'Percentage Hypnodensity Matrix for {subtype}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Expert Scoring')
    plt.show()
    
# %% Stats YASA - pYASA

features = ["stage", "beta", "tvalue", "p_val", "p_corr" ]
this_dic = {feature : [] for feature in features}

this_df = df_hypnodensities[df_hypnodensities.subtype!="N1"]

for proba in ["p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM"] :
    temp_pval = []
    for stage in ['WAKE', 'N1', 'N2', 'N3', 'REM']:
        this_dic["stage"].append(f"{stage} x {proba}")
    
        model_formula = f'{proba} ~ duration_waked + duration_N1 + duration_N2 + duration_N3 + duration_REM + C(subtype, Treatment("C1")) * C(stage, Treatment("{stage}"))'
        model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'])
        model_result = model.fit(reml = False)
        
        this_dic["beta"].append(model_result.params['C(subtype, Treatment("C1"))[T.HI]'])
        this_dic["p_val"].append(model_result.pvalues['C(subtype, Treatment("C1"))[T.HI]'])
        temp_pval.append(model_result.pvalues['C(subtype, Treatment("C1"))[T.HI]'])
        this_dic["tvalue"].append(model_result.tvalues['C(subtype, Treatment("C1"))[T.HI]'])
    
    _, p_corrs = fdrcorrection(temp_pval)
    for i in p_corrs :
        this_dic['p_corr'].append(i)
    
df_stat = pd.DataFrame.from_dict(this_dic)
print(f"{proba}\n{df_stat}")

# %% Stats Expert - pYASA

features = ["stage", "beta", "tvalue", "p_val", "p_corr" ]
this_dic = {feature : [] for feature in features}

this_df = small_df[small_df.subtype!="N1"]

for proba in ["p_WAKE", "p_N1", "p_N2", "p_N3", "p_REM"] :
    temp_pval = []
    for stage in ['WAKEd', 'N1', 'N2', 'N3', 'REM']:
        this_dic["stage"].append(f"{stage} x {proba}")
    
        model_formula = f'{proba} ~ duration_waked + duration_N1 + duration_N2 + duration_N3 + duration_REM + C(subtype, Treatment("C1")) * C(stage, Treatment("{stage}"))'
        model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'])
        model_result = model.fit(reml = False)
        
        this_dic["beta"].append(model_result.params['C(subtype, Treatment("C1"))[T.HI]'])
        this_dic["p_val"].append(model_result.pvalues['C(subtype, Treatment("C1"))[T.HI]'])
        temp_pval.append(model_result.pvalues['C(subtype, Treatment("C1"))[T.HI]'])
        this_dic["tvalue"].append(model_result.tvalues['C(subtype, Treatment("C1"))[T.HI]'])
    
    _, p_corrs = fdrcorrection(temp_pval)
    for i in p_corrs :
        this_dic['p_corr'].append(i)
    
df_stat = pd.DataFrame.from_dict(this_dic)
print(f"{proba}\n{df_stat}")
