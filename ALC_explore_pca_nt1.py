#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:06:30 2023

@author: arthurlecoz

explore_pca.py

- PCA weights & score               OK

for each stage (WAKEb, Waked, N1, N2, N3, REM) :
    -> MACRO 
        * STAGES                    
        - D'                        OK
        - MetaD'                    OK

For each electrode (F3, C3, O1) :
    -> MESO 
        - Puissance                 To Compute
            - Delta                 OK
            - Theta                 OK
            - Alpha                 OK
            - Sigma                 OK
            - Beta                  OK
        * FOOOF 
        - Slope                     OK
        - Exponent                  OK
        * Complexity 
        - Kolmogorov                OK
        - Approximate Entropy       OK
        - Sample Entropy            OK
        - Permutation Entropy       OK
            -> Each bandfreq
    -> MICRO 
        * SW DETECTION 
        - Density                   OK
        - PTP                       OK
        - D Slope                   OK
        - U Slope                   OK
        * YASA SS DET               Verifications to be made (N2/N3 density)
        - Density                   ...
        - Amplitude/Puissance (?)   ...
        - Frequency                 ...

"""
# %% paths

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import localdef

DataType='PSG' #TILE, PSG, JC, TME

local = False

if local :
    if DataType == 'TILE':
        root_dir = localdef.LOCAL_path_TILE
    elif DataType == 'PSG':
        root_dir = localdef.LOCAL_path_PSG
    elif DataType == 'JC':
        print ('Not processed yet') 
    else:
        print ('Data-type to process has not been selected/recognized') 
else :
    if DataType == 'TILE':
        root_dir=localdef.DDE_path_TILE
    elif DataType == 'PSG':
        root_dir=localdef.DDE_path_PSG
    elif DataType == 'JC':
        print ('Not processed yet') 
    else:
        print ('Data-type to process has not been selected/recognized') 

preproc_dir = root_dir+'/Preproc'
raw_dir = root_dir+'/Raw'
fig_dir = root_dir+'/Figs' 
complexity_dir = os.path.join(fig_dir, "complexity")
fooof_dir = os.path.join(fig_dir, "fooof")

df_demographics = pd.read_csv(
    '/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv',
    sep = ";"
    )

# %% PCA Clinic

df = df_demographics[df_demographics["diag.2"].isin(["C1", "N1"])]
# df = df_demographics[df_demographics["diag.2"].isin(["HIw", "C1"])]
n_sub, n_col = df.shape

col = np.squeeze(df.columns)

features = []
for info in col :
    if sum(df[info].isna())/n_sub < .15 :
        features.append(info)
    
df_pca = df[np.squeeze(features)]

df_pca_nonight = df[[
    'code', 'diag.2', 'sexe','taille', 'ESS','ivresse', 
    'sieste.reg', 'dur.som.sem', 'cata',
    'dur.som.we', 'ATCD.dep'
    ]]

for col in df_pca_nonight.columns :
    if df_pca_nonight[col].isna().any() :
        meanValue = df_pca_nonight[col].dropna().mean()
        df_pca_nonight[col].fillna(meanValue, inplace = True)

ids = np.asarray(df_pca_nonight.code)
diag2 = np.asarray(df_pca_nonight['diag.2'])
df_test = df_pca_nonight.drop(columns = ["code", "diag.2"])
features = df_test.columns

first_row = df_test.iloc[0]

for i, value in enumerate(first_row) :
    if type(value) is str :
        df_test[df_test.columns[i]] = [
            float(val.replace(",", ".")) for val in df_test[df_test.columns[i]]
            ] 
x = df_test.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(
    data = principalComponents, columns = ['PC1', 'PC2']
    )
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2'], 
    index=features
    )
save_PCA_CQ = principalDf.copy()
save_loadings_CQ = loadings.copy()

principalDf["code"] = ids
principalDf["subtype"] = diag2

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Min Info', fontsize = 20)

targets = ['N1', 'C1']
colors = ['r', 'b', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['subtype'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC1']
               , principalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# %% creating df

# df_complexity = pd.read_csv(
#     os.path.join(complexity_dir, "conform_dfcomplex.csv")
#     )
df_bandpower = pd.read_csv(
    os.path.join(fig_dir, "bandpower_df.csv")
    )
df_bandpower = df_bandpower.loc[df_bandpower.subtype != 'HI']

df_fooof = pd.read_csv(
    os.path.join(fooof_dir, "all_offset_exponent.csv")
    )
df_fooof = df_fooof.loc[df_fooof.sub_id.isin(df_bandpower.sub_id.unique())]
df_fooof = df_fooof.loc[df_fooof.stage != 'WAKEb']
df_fooof['stage'] = df_fooof['stage'].replace('WAKEd', 'WAKE')


df_sw = pd.read_csv(
    os.path.join(fig_dir, "df_allsw_exgausscrit_nobin_090924.csv")
    )
df_sw = df_sw.loc[df_sw.sub_id.isin(df_bandpower.sub_id.unique())]
df_sw = df_sw.loc[df_sw.stage != 'WAKEb']
df_sw['stage'] = df_sw['stage'].replace('WAKEd', 'WAKE')

df_ss = pd.read_csv(
    os.path.join(fig_dir, "yasa", "df_nt1_hi_cns_ss_yasa.csv")
    )
df_ss = df_ss.loc[df_ss.sub_id.isin(df_bandpower.sub_id.unique())]


df_entropydkl = pd.read_csv(
    os.path.join(fig_dir, "yasa",  "df_yasa_hypnodens_all.csv")
    )
df_entropydkl = df_entropydkl.loc[df_entropydkl.sub_id.isin(df_bandpower.sub_id.unique())]
df_entropydkl = df_entropydkl.loc[df_entropydkl.stage != 'WAKEb']
df_entropydkl['stage'] = df_entropydkl['stage'].replace('WAKEd', 'WAKE')

# df_stage = pd.read_csv(os.path.join(
#     fig_dir, "df_stages_certitudes.csv"
#     ))
# df_bursts = pd.read_csv(os.path.join(
#     fig_dir, "bursts", "theta_alpha_bursts", "features", "allfeat_v2.csv"
#     ))

# del df_complexity['Unnamed: 0'], 
del df_fooof['Unnamed: 0'], df_sw['Unnamed: 0']
del df_ss['Unnamed: 0']
del df_bandpower['Unnamed: 0']
del df_entropydkl['Unnamed: 0']
# del df_stage['Unnamed: 0'], df_bursts['Unnamed: 0']

df_ss.rename(
    columns = {col : f"ss_{col}" for col in df_ss.columns[8:]}, 
    inplace = True
    )

df_sw_ss = pd.merge(
    df_sw, 
    df_ss[['sub_id', 'age', 'subtype', 'stage', 'channel', 
         'ss_count', 'ss_density', 'ss_abs_power', 'ss_rel_power',
         'ss_frequency', 'ss_duration']], 
    on = ['sub_id', 'age', 'subtype', 'stage', 'channel'],
    how = "left"
    )

huge_df = pd.merge(
    df_sw_ss, 
    df_bandpower[
        ['sub_id', 'stage', 'channel',
         'abs_delta','abs_theta', 'abs_alpha', 'abs_sigma', 'abs_beta', 
         'rel_delta','rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta']
        ], 
    on = ['sub_id', 'stage', 'channel'],
    how = "left"
    )

huge_df = pd.merge(
    huge_df, 
    df_fooof[['sub_id', 'stage', 'channel', 'exponent','offset']], 
    on = ['sub_id', 'stage', 'channel'],
    how = "left"
    )

huge_df = pd.merge(
    huge_df, 
    df_entropydkl[['sub_id', 'stage', 'channel', 'entropy', 'dKL']], 
    on = ['sub_id', 'stage', 'channel'],
    how = "left"
    )

# huge_df = pd.concat([
#     df_sw,
#     df_burstalpha[df_burstalpha.columns[8:]],
#     df_bursttheta[df_bursttheta.columns[8:]],
#     df_bandpower[['delta','theta', 'alpha', 'beta']],
#     df_fooof[['exponent','offset']],
#     df_complexity[[
#         'approximative_entropy', 'sample_entropy', 'kolmogorov',
#         'permutation_entropy_delta', 'permutation_entropy_theta',
#         'permutation_entropy_alpha', 'permutation_entropy_beta'
#         ]],
#     df_stage[['probability', 'uncertainty']]
#     ], axis = 1)

# df = df_demographics[df_demographics.code.isin(df_sw.sub_id.unique())]

# huge_df = pd.concat([
#     df_sw_ss,
#     df_bandpower[
#         ['abs_delta','abs_theta', 'abs_alpha', 'abs_sigma', 'abs_beta', 
#          'rel_delta','rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta']
#     ],
#     df_fooof[['exponent','offset']],
#     df_entropydkl[['entropy', 'dKL']]
#     ], axis = 1)

# huge_df = huge_df.loc[huge_df.subtype != "HSI"]

eeg_df = huge_df.copy()

# huge_df['PC1'] = [principalDf.PC1.loc[
#     principalDf.code == row.sub_id
#     ].iloc[0] for i, row in huge_df.iterrows()]
# huge_df['PC2'] = [principalDf.PC2.loc[
#     principalDf.code == row.sub_id
#     ].iloc[0] for i, row in huge_df.iterrows()]

huge_df.to_csv(
    os.path.join(fig_dir, "pca_hugedf_nt1_cns.csv"),
    index = False
    )

# %% spearman

stages = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
channels = huge_df.channel.unique()

PC1_dic = {
    stage : {
        info : np.nan * np.empty((
        huge_df.columns[13:-2].shape[0], huge_df.channel.unique().shape[0] 
        ))
        for info in ['p_val', 'r_val', 'p_corr']
        }
        for stage in stages
    }
PC2_dic = {
    stage : {
        info : np.nan * np.empty((
        huge_df.columns[13:-2].shape[0], huge_df.channel.unique().shape[0] 
        ))
        for info in ['p_val', 'r_val', 'p_corr']
        }
        for stage in stages
    }

for i_f, feature in enumerate(huge_df.columns[13:-2]):
    print(f"Processing... {feature}")
    for stage in huge_df.stage.unique() :
        for i_ch, channel in enumerate(channels) :
            subdf = huge_df.loc[
                (huge_df.stage == stage)
                & (huge_df.channel == channel)
                ]
            (PC1_dic[stage]['r_val'][i_f, i_ch], 
             PC1_dic[stage]['p_val'][i_f, i_ch]) = spearmanr(
                a = subdf.PC1, 
                b = subdf[feature], 
                nan_policy='omit'
                )
            (PC2_dic[stage]['r_val'][i_f, i_ch], 
             PC2_dic[stage]['p_val'][i_f, i_ch]) = spearmanr(
                a = subdf.PC2, 
                b = subdf[feature], 
                nan_policy='omit'
                )

alpha = 0.05
for stage in stages:
    for i_f, feature in enumerate(huge_df.columns[13:-2]):
        reject, p_corr, _, _ = multipletests(
            PC1_dic[stage]['p_val'][i_f, :], 
            alpha=alpha, 
            method='fdr_bh'
            )
        PC1_dic[stage]['p_corr'][i_f, :] = p_corr
        reject, p_corr, _, _ = multipletests(
            PC2_dic[stage]['p_val'][i_f, :], 
            alpha=alpha, 
            method='fdr_bh'
            )
        PC2_dic[stage]['p_corr'][i_f, :] = p_corr
            
# %% heatmap spearman

for stage in huge_df.stage.unique() :
    fig, ax = plt.subplots(
        figsize = (16, 16)
        )
    temp_pc1 = np.transpose(PC1_dic[stage]['p_corr'], (1, 0))
    temp_pc2 = np.transpose(PC2_dic[stage]['p_corr'], (1, 0))
    df_pval = pd.DataFrame(
        np.concatenate((temp_pc1, temp_pc2)),
        columns = huge_df.columns[13:-2]
        )
    temp_pc1 = np.transpose(PC1_dic[stage]['r_val'], (1, 0))
    temp_pc2 = np.transpose(PC2_dic[stage]['r_val'], (1, 0))
    df_rval = pd.DataFrame(
        np.concatenate((temp_pc1, temp_pc2)),
        columns = huge_df.columns[13:-2]
        )
    mask = np.triu(df_pval <= 0.05)
    sns.heatmap(
        df_rval, 
        yticklabels = ["PC1-F3", "PC1-C3", "PC1-O1",
                 "PC2-F3", "PC2-C3", "PC2-O1"],
        annot = True,
        cmap = "vlag",
        mask = mask,
        ax = ax,
        fmt=".2f"
        )
    fig.suptitle(stage)
    fig.tight_layout(pad = 1)
    plt.savefig(os.path.join(
        fig_dir, f"{stage}_PCA_spearmanmatrix_nt1cns.png"
        ),
        dpi = 200)
    
# %% lmm

PC1_dic = {
    stage : {
        info : np.nan * np.empty((
        huge_df.columns[13:-2].shape[0], huge_df.channel.unique().shape[0] 
        )) 
        for info in ['p_val', 't_val']
        }
        for stage in ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
    }
PC2_dic = {
    stage : {
        info : np.nan * np.empty((
        huge_df.columns[13:-2].shape[0], huge_df.channel.unique().shape[0] 
        )) 
        for info in ['p_val', 't_val']
        }
        for stage in ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
    }

for i_f, feature in enumerate(huge_df.columns[13:-2]):
    print(f"Processing... {feature}")
    for stage in huge_df.stage.unique() :
        for i_ch, channel in enumerate(huge_df.channel.unique()) :
            subdf = huge_df.loc[
                (huge_df.stage == stage)
                & (huge_df.channel == channel)
                ]
            
            lm_formula = f"{feature} ~ PC1"
            md = smf.mixedlm(
                lm_formula, 
                subdf, 
                groups = subdf['sub_id'],
                missing = "drop"
                )
            mdf = md.fit()
            
            PC1_dic[stage]['t_val'][i_f, i_ch] = mdf.tvalues.PC1
            PC1_dic[stage]['p_val'][i_f, i_ch] = mdf.pvalues.PC1
                
            lm_formula = f"{feature} ~ PC2"
            md = smf.mixedlm(
                lm_formula, 
                subdf, 
                groups = subdf['sub_id'],
                missing = "drop"
                )
            mdf = md.fit()
            
            PC2_dic[stage]['t_val'][i_f, i_ch] = mdf.tvalues.PC2
            PC2_dic[stage]['p_val'][i_f, i_ch] = mdf.pvalues.PC2


# %% plot_lmm

for stage in huge_df.stage.unique() :
    fig, ax = plt.subplots(
        figsize = (16, 16)
        )
    temp_pc1 = np.transpose(PC1_dic[stage]['p_val'], (1, 0))
    temp_pc2 = np.transpose(PC2_dic[stage]['p_val'], (1, 0))
    df_pval = pd.DataFrame(
        np.concatenate((temp_pc1, temp_pc2)),
        columns = huge_df.columns[13:-2]
        )
    temp_pc1 = np.transpose(PC1_dic[stage]['t_val'], (1, 0))
    temp_pc2 = np.transpose(PC2_dic[stage]['t_val'], (1, 0))
    df_tval = pd.DataFrame(
        np.concatenate((temp_pc1, temp_pc2)),
        columns = huge_df.columns[13:-2]
        )
    mask = np.triu(df_pval <= 0.05)
    sns.heatmap(
        df_rval, 
        yticklabels = ["PC1-F3", "PC1-C3", "PC1-O1",
                 "PC2-F3", "PC2-C3", "PC2-O1"],
        annot = True,
        cmap = sns.diverging_palette(20, 220, as_cmap=True),
        mask = mask,
        ax = ax,
        # vmax = df_rval.max().sort_values().iloc[-4],
        # vmin = df_rval.min().sort_values().iloc[-4]
        )
    fig.suptitle(stage)
    fig.tight_layout(pad = 1)
    
# %% PCA EEG Feat

"""
Here the goal is to obtain PC1 & PC2 scores and their weights 
> For each stage & channel
=> Then correlates the scores to the clinical questionnaires (CQ)
==> Hoping to see correlation w/ the criteria we saw for the PCA on the CQ
"""

# %% Create df PCA 

inspect = 0

# print(eeg_df)
n_sub, n_col = df.shape

col = np.squeeze(eeg_df.columns)

features = []
for info in col :
    if sum(eeg_df[info].isna())/n_sub < .15 :
        features.append(info)
    
df_pca = eeg_df[np.squeeze(features)]

for col in df_pca.columns :
    if df_pca[col].isna().any() :
        meanValue = df_pca[col].dropna().mean()
        df_pca[col].fillna(meanValue, inplace = True)
        
list_pca = []
list_loadings = []
for stage in ["WAKEb", "WAKEd", "N1", "N2", "N3", "REM"]:
    print(f"...Processing {stage}")
    for channel in ['F3', 'C3', 'O1'] :
        
        tempdf = df_pca.loc[
            (df_pca.stage == stage)
            & (df_pca.channel == channel)
            ]
        
        ids = np.asarray(tempdf.sub_id)
        diag2 = np.asarray(tempdf['subtype_2'])
        df_test = tempdf.drop(columns = [
            'sub_id','subtype','subtype_2','subtype_3',
            'genre','age', 'stage', 'channel'
            ]).reset_index(drop = True)
        
        features = df_test.columns
        first_row = df_test.iloc[0]
        
        x = df_test.loc[:, features].values
        x = StandardScaler().fit_transform(x)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        
        principalDf = pd.DataFrame(
            data = principalComponents, columns = ['PC1', 'PC2']
            )
        loadings = pd.DataFrame(
            pca.components_.T, columns=['PC1', 'PC2'], index=features
            )
        
        principalDf['sub_id'] = ids
        
        principalDf = principalDf.rename(
            columns = {col : f"{col}_{stage}_{channel}" 
                       for col in principalDf.columns[:2]}
            )
        list_pca.append(principalDf)
        loadings = loadings.rename(
            columns = {col : f"{col}_{stage}_{channel}" 
                       for col in loadings.columns}
            )
        list_loadings.append(loadings)
        
        # principalDf["code"] = ids
        # principalDf["subtype"] = diag2
        
        if inspect :
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PC1', fontsize = 15)
            ax.set_ylabel('PC2', fontsize = 15)
            ax.set_title('Min Info', fontsize = 20)
            
            targets = ['N1', 'C1']
            colors = ['r', 'b', 'k']
            for target, color in zip(targets,colors):
                indicesToKeep = principalDf['subtype'] == target
                ax.scatter(principalDf.loc[indicesToKeep, 'PC1']
                           , principalDf.loc[indicesToKeep, 'PC2']
                           , c = color
                           , s = 50)
            ax.legend(targets)
            ax.grid()
            fig.suptitle(f"{stage}_{channel}")

final_pca_df = pd.DataFrame()
final_pca_df['sub_id'] = eeg_df.sub_id.unique()
for df in list_pca :
    final_pca_df = pd.merge(
        final_pca_df, df, 
        on='sub_id', 
        how='outer'
        )
final_loadings_df = pd.concat(list_loadings, axis = 1)

# add CQ

sub_demo = df_demographics[
    ['code', 'diag.2', 'ESS','ivresse','sieste.reg', 'dur.som.sem', 
     'dur.som.we', 'cata']
    ].loc[df_demographics["diag.2"].isin(["HIw", "C1", "N1"])]
sub_demo = sub_demo.rename(columns = {'code' : 'sub_id'})

final_df = pd.merge(
    final_pca_df, sub_demo, 
    on='sub_id', 
    how='outer'
    )

# %% Create df PCA 

inspect = 0

n_sub, n_col = df.shape

col = np.squeeze(eeg_df.columns)

features = []
for info in col :
    if sum(eeg_df[info].isna())/n_sub < .15 :
        features.append(info)
    
df_pca = eeg_df[np.squeeze(features)]

for col in df_pca.columns :
    if df_pca[col].isna().any() :
        meanValue = df_pca[col].dropna().mean()
        df_pca[col].fillna(meanValue, inplace = True)
        
list_pca = []
list_loadings = []
        
ids = np.asarray(df_pca.sub_id)
diag2 = np.asarray(df_pca['subtype_2'])
stages = np.asarray(df_pca['stage'])
channels = np.asarray(df_pca['channel'])
df_test = df_pca.drop(columns = [
    'sub_id','subtype','subtype_2','subtype_3',
    'genre','age', 'stage', 'channel'
    ]).reset_index(drop = True)

features = df_test.columns
first_row = df_test.iloc[0]

x = df_test.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(x)

# pca = PCA()
# principalComponents = pca.fit_transform(x)
# ratios = pca.explained_variance_ratio_ * 100

# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))

principalDf = pd.DataFrame(
    data = principalComponents, 
    columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
    )
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], 
    index=features
    )

principalDf['sub_id'] = ids
principalDf["subtype"] = diag2
principalDf["stage"] = stages
principalDf["channel"] = channels

for stage in principalDf.stage.unique() :
    subdf = principalDf.loc[principalDf['stage'] == stage]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('Min Info', fontsize = 20)
    
    targets = ['N1', 'HIw', 'C1']
    colors = ['r', 'b', 'k']
    for target, color in zip(targets,colors):
        indicesToKeep = subdf['subtype'] == target
        ax.scatter(subdf.loc[indicesToKeep, 'PC1']
                   , subdf.loc[indicesToKeep, 'PC2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    fig.suptitle(f"{stage}")


# %% spearman eeg - CQ

stages = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
channels = huge_df.channel.unique()

PC_dic = {
    stage : {comp : {
        stat : np.nan * np.empty((len(final_df.columns[-6:]), len(channels))) 
        for stat in ['p_val', 'r_val', 'p_cor']
        } for comp in ['PC1', 'PC2']} for stage in stages
    }

for i_f, feature in enumerate(final_df.columns[-6:]):
    for i_comp, PC in enumerate(["PC1", "PC2"]) :
        for i_st, stage in enumerate(stages) :
            for i_ch, channel in enumerate(channels) :
                thiscol = final_df.filter(like = f"{PC}_{stage}_{channel}")
                thisfeature = final_df.filter(like = feature)
                
                (PC_dic[stage][PC]['r_val'][i_f, i_ch], 
                 PC_dic[stage][PC]['p_val'][i_f, i_ch]) = spearmanr(
                    a = thiscol, 
                    b = thisfeature, 
                    nan_policy='omit'
                    )

alpha = 0.05
for PC in ["PC1", "PC2"] :
    for stage in stages:
        for i_f, feature in enumerate(final_df.columns[-6:]):
            reject, p_corr, _, _ = multipletests(
                PC_dic[stage][PC]['p_val'][i_f, :], 
                alpha=alpha, 
                method='fdr_bh'
                )
            PC_dic[stage][PC]['p_cor'][i_f, :] = p_corr

# %% heatmap spearman

fig, ax = plt.subplots(
    nrows = 1, 
    ncols = 2,
    figsize = (18, 18),
    sharey=True
    )
for i_c, comp in enumerate(["PC1", "PC2"]) :
    
    vmin = -0.15
    vmax = 0.15
 
    df_p = pd.DataFrame(
        np.asarray(
            [PC_dic[stage][comp]['p_cor'][:,1]
             for stage in stages]
            ),
        columns = final_df.columns[-6:]
        )
    df_r = pd.DataFrame(
        np.asarray(
            [PC_dic[stage][comp]['r_val'][:,1]
             for stage in stages]
            ),
        columns = final_df.columns[-6:]
        )
    mask = np.triu(df_p <= 0.05)
    sns.heatmap(
        df_r, 
        yticklabels = [f"{stage}" 
                       for stage in stages],
        annot = True,
        cmap = "vlag",
        mask = mask,
        ax = ax[i_c],
        fmt=".3f",
        vmin = vmin,
        vmax = vmax
        )
    ax[i_c].tick_params(left = False, bottom = False)
    ax[i_c].set_title(comp)
    fig.suptitle(
        "Spearman Correlation of Principal Component Analysis on EEG features with Clinical Questionnaires",
        fontweight = "heavy"
        )
    fig.tight_layout(pad = 1)
    
plt.savefig(os.path.join(
    fig_dir, "spearman_PCEEG_CQ_heatmap_stade_C3"
    ),
    dpi = 200)

# %% PC & barplot

component = "PC2"

C3_loads = final_loadings_df.filter(like = "C3")
PC1C3_loads = C3_loads.filter(like = component).reset_index()

g = sns.PairGrid(
    PC1C3_loads.sort_values(f"{component}_WAKEb_C3"), 
    x_vars=PC1C3_loads.columns[1:], 
    y_vars=["index"],
    height=10, 
    aspect=.25
    )

g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(-0.35, 0.36), xlabel="Loading Weight", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Wake B", "Wake D", "N1",
          "N2", "N3", "REM"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
g.fig.suptitle(
    f"Loadings - {component}", fontsize = "xx-large", fontweight = "bold"
    )    
g.tight_layout(pad = 1)

plt.savefig(os.path.join(
    fig_dir, f"loadings_{component}"
    ),
    dpi = 200)

# %% MERGE PCA EEG x PCA CQ

save_PCA_CQ.insert(0, 'sub_id', save_PCA_CQ["code"])
del save_PCA_CQ['code']

save_loadings_CQ

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Min Info', fontsize = 20)

targets = ['N1', 'HIw', 'C1']
colors = ['r', 'b', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = save_PCA_CQ['subtype'] == target
    ax.scatter(save_PCA_CQ.loc[indicesToKeep, 'PC1']
               , principalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

PCA_CQ_EEG = pd.merge(
    final_pca_df, save_PCA_CQ, 
    on='sub_id', 
    how='outer'
    )

# %% PC_CQ x PC_EEG Spearman

stages = ['WAKEb', 'WAKEd', 'N1', 'N2', 'N3', 'REM']
channels = huge_df.channel.unique()

PC_dic = {
    stage : {comp : {
        stat : np.nan * np.empty((2, len(channels))) 
        for stat in ['p_val', 'r_val', 'p_cor']
        } for comp in ['PC1', 'PC2']} for stage in stages
    }

for i_f, feature in enumerate(['PC1', 'PC2']):
    for i_comp, PC in enumerate(["PC1", "PC2"]) :
        for i_st, stage in enumerate(stages) :
            for i_ch, channel in enumerate(channels) :
                thiscol = PCA_CQ_EEG.filter(like = f"{PC}_{stage}_{channel}")
                thisfeature = PCA_CQ_EEG[feature]
                
                (PC_dic[stage][PC]['r_val'][i_f, i_ch], 
                 PC_dic[stage][PC]['p_val'][i_f, i_ch]) = spearmanr(
                    a = thiscol, 
                    b = thisfeature, 
                    nan_policy='omit'
                    )

alpha = 0.05
for PC in ["PC1", "PC2"] :
    for stage in stages:
        for i_f, feature in enumerate(['PC1', 'PC2']):
            reject, p_corr, _, _ = multipletests(
                PC_dic[stage][PC]['p_val'][i_f, :], 
                alpha=alpha, 
                method='fdr_bh'
                )
            PC_dic[stage][PC]['p_cor'][i_f, :] = p_corr

# %% PC_CQ x PC_EEG Heatmap

fig, ax = plt.subplots(
    nrows = 2, 
    ncols = 1,
    figsize = (4, 16),
    sharey=True
    )
for i_c, comp in enumerate(["PC1", "PC2"]) :
    
    df_p = pd.DataFrame(
        np.asarray(
            [PC_dic[stage][comp]['p_cor'][:,1]
             for stage in stages]
            ),
        columns = ['PC1', 'PC2']
        )
    df_r = pd.DataFrame(
        np.asarray(
            [PC_dic[stage][comp]['r_val'][:,1]
             for stage in stages]
            ),
        columns = ['PC1', 'PC2']
        )
    mask = np.triu(df_p <= 0.05)
    sns.heatmap(
        df_r, 
        yticklabels = [f"{stage}" 
                       for stage in stages],
        annot = True,
        cmap = "vlag",
        mask = mask,
        ax = ax[i_c],
        fmt=".3f",
        vmin = vmin,
        vmax = vmax
        )
    plt.yticks(rotation = 0)
    ax[i_c].tick_params(left = False, bottom = False)
    ax[i_c].set_title(f'{comp} EEG')
    fig.suptitle(
        "Spearman r-values\nPCA EEG x PCA Clinical Questionnaire",
        fontweight = "heavy"
        )
    fig.tight_layout(pad = 1)
    
plt.savefig(os.path.join(
    fig_dir, "spearman_PCEEG_PCACQ_heatmap_stade_C3"
    ),
    dpi = 200)

