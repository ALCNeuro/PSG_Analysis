#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Arthur_LC

ALC_PLOT_SS_threshold_trad.py

Currently HI x NT1 x CnS

"""
# %%% Paths & Packages

import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import numpy as np
from datetime import date
from glob import glob
import os
todaydate = date.today().strftime("%d%m%y")

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = os.path.join(root_dir, "Raw")
preproc_dir = os.path.join(root_dir, "Preproc")
fig_dir = os.path.join(root_dir, "Figs")

which = "all" # nt1_cns or all 


if which == "nt1_cns":
   palette = ["#8d99ae", "#d00000"]
   hue_order = ["CTL", "NT1"]
elif which == "all" :
   palette = ["#8d99ae", "#d00000", "#ffb703"]
   hue_order = ["CTL", "NT1", "HSI"]
 
order = ["N2", "N3"]
stages = ["N2", "N3"]


# %% SPINDLES NORMAL
# %% LOAD DF

# Which subjects ?

if which == "nt1_cns" :
    df = pd.read_csv(f'{fig_dir}/yasa/df_nt1_cns_ss_yasa.csv')
elif which == 'all':
    df = pd.read_csv(f'{fig_dir}/yasa/df_nt1_hi_cns_ss_yasa.csv')
del df['Unnamed: 0']

df = df.loc[df.sub_id != "N1_016"]

#%% density

fig, ax = plt.subplots(ncols = 1, nrows = 3, sharex=True, sharey=True, 
figsize = (4, 16), layout = 'tight')

for i_st, stage in enumerate(["N2", "N3"]) :
    for i_ch, channel in enumerate(["F3", "C3", "O1"]) :
        sns.violinplot(
            x = "stage", order = order, y = "density", 
            hue = "subtype", hue_order = hue_order,
            data = df.loc[df.channel == channel], 
            palette = palette, ax = ax[i_ch], cut = 2, inner = 'quartile'
            )
        
        # ax[i_ch].set_yticks(
        #     ticks = np.arange(0, 5, 1),
        #     labels = ["" for i in np.arange(0, 5, 1)]
        #     )
        # ax[i_ch].set_xticks(
        #     ticks = [0, 1],
        #     labels = ("", "")
        #     )
        ax[i_ch].legend_ = None
        # ax[i_ch].set_xlabel("")
        # ax[i_ch].set_ylabel("")

plt.savefig(
    f"{fig_dir}/{which}_ssdensity_subtype_chan_subplots.png", 
    dpi = 300
        )

# %% N2 - O1

x = "stage"
order = ['N2']
y = "density"
if which == "nt1_cns":
    hue_order = ["CTL", "NT1"]
    palette = ["#8d99ae", "#d00000"]
elif which == "all":
    hue_order = ["CTL", "NT1", "HSI"]
    palette = ["#8d99ae", "#d00000", "#ffb703"]
hue = "subtype"
box_width = .2
data = df

fig, ax = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (4, 8), layout = "tight")

for i_ch, chan in enumerate(["O1"]):
    sns.pointplot(
        x = x, 
        order = order, 
        y = y, 
        hue = hue, 
        hue_order = hue_order,
        data = data.loc[data.channel == chan], 
        errorbar = None,
        capsize = 0.05,
        dodge = .55,
        linestyle = 'none',
        alpha = .9,
        palette = palette, 
        ax = ax,
        markers = "D"
        )      
    sns.violinplot(
        x = x, 
        order = order, 
        y = y, 
        hue = hue, 
        hue_order = hue_order,
        data = data.loc[data.channel == chan], 
        palette = palette,
        # fill = False, 
        inner = None,
        alpha = .2,
        linecolor = "w",
        cut = 1, 
        ax = ax
        )
    sns.stripplot(
        x = x, 
        order = order, 
        y = y, 
        hue = hue, 
        hue_order = hue_order,
        data = data.loc[data.channel == chan], 
        alpha = 0.2,
        dodge = True,
        legend = None,
        ax = ax,
        palette = palette
        )
    
    ax.legend_ = None
    ax.set_ylabel("Sleep Spindle Density (nSS/nEpochs)", font = bold_font, fontsize = 24)
    ax.set_xlabel("N2", font = bold_font, fontsize = 24)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        )
    ax.set_yticks(
        np.linspace(0, 5, 6), 
        np.linspace(0, 5, 6), 
        font = font, 
        fontsize = 16
        )
    ax.set_ylim(0, 4)
    sns.despine(bottom=True)
    fig.tight_layout()

# ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/{which}_ss_density_O1_N2.png", 
    dpi = 300
    )

# %% STATS | SWA age + gender

feature = 'density'

if which == "nt1_cns":
    foi = ["Stage", "Channel", "ß (iHS_LTS vs HS)", "p_val"]
    dic = {f : [] for f in foi}
    
    for stage in df.stage.unique():
        for channel in df.channel.unique() :
            model_formula = f'{feature} ~ age + sexe + C(subtype, Treatment("CTL")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
            model = smf.mixedlm(model_formula, df, groups=df['sub_id'], missing='drop')
            model_result = model.fit()
            
            dic["Stage"].append(stage)
            dic["Channel"].append(channel)
            dic["ß (iHS_LTS vs HS)"].append(
                model_result.params['C(subtype, Treatment("CTL"))[T.NT1]'])
            dic["p_val"].append(
                model_result.pvalues['C(subtype, Treatment("CTL"))[T.NT1]'])
            
            
    corr_pval = multipletests(dic["p_val"], method='fdr_tsbh')
    print(np.asarray(dic["Channel"])[corr_pval[0]])
    print(np.asarray(dic["Stage"])[corr_pval[0]])
    print(np.asarray(corr_pval[1])[corr_pval[0]])
    dic['p_corr'] = list(corr_pval[1])
    stats_df = pd.DataFrame.from_dict(dic)
    print(stats_df)
elif which == "all" :
    
    group_comparison = [["CTL", "NT1"], ["CTL", "HSI"]]
    
    for duo in group_comparison :
        duo_df = df.loc[df.subtype.isin(duo)]
        foi = [
            "Stage", "Channel", "comparison", "ß", "p_val"
            ]
        dic = {f : [] for f in foi}
        for stage in df.stage.unique():
            for channel in df.channel.unique() :
                
                model_formula_ctl = f'{feature} ~ age + sexe + C(subtype, Treatment("CTL")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
                model_ctl = smf.mixedlm(model_formula_ctl, duo_df, groups=duo_df['sub_id'], missing='drop')
                model_result_ctl = model_ctl.fit()
                    
                dic["Stage"].append(stage)
                dic["Channel"].append(channel)
            
                dic["comparison"].append(f'{duo[0]} vs {duo[1]}')
                dic["ß"].append(
                    model_result_ctl.params[
                        f'C(subtype, Treatment("{duo[0]}"))[T.{duo[1]}]'
                        ])
                dic["p_val"].append(
                    model_result_ctl.pvalues[
                        f'C(subtype, Treatment("{duo[0]}"))[T.{duo[1]}]'
                        ])
                
        corr_pval = multipletests(dic["p_val"], method='fdr_tsbh')
        print(np.asarray(dic["Channel"])[corr_pval[0]])
        print(np.asarray(dic["Stage"])[corr_pval[0]])
        print(np.asarray(corr_pval[1])[corr_pval[0]])
        dic['p_corr'] = list(corr_pval[1])
        stats_df = pd.DataFrame.from_dict(dic)
        print(stats_df)

#%% feature

features = ['abs_power', 'rel_power','frequency', 
            'duration', 'amplitude', 'oscillations']

feature = "amplitude"
fig, ax = plt.subplots(ncols = 6, nrows = 3, sharex=True, #sharey=True, 
figsize = (16, 16), layout = 'tight')
for i_f, feature in enumerate(features) :
    for i_st, stage in enumerate([2, 3]) :
        for i_ch, channel in enumerate(["F3", "C3", "O1"]) :
            sns.violinplot(
                x = "stage", order = order, y = feature, 
                hue = "subtype", hue_order = hue_order,
                data = df.loc[df.channel == channel], 
                palette = palette, ax = ax[i_ch][i_f], cut = 2, 
                inner = 'quartile'
                )
            ax[i_ch][i_f].spines['right'].set_visible(False)
            ax[i_ch][i_f].spines['top'].set_visible(False)
            # ax[i_ch][i].spines['left'].set_visible(False)
            # ax[i_ch][i].spines['bottom'].set_visible(False)
            # ax[i_ch].set_yticks(
            #     ticks = np.arange(11.5, 16.5, 1),
            #     labels = ["" for i in np.arange(11, 16, 1)]
            #     )
            # ax[i_ch].set_xticks(
            #     ticks = [0, 1],
            #     labels = ("", "")
            #     )
            ax[i_ch][i_f].legend_ = None
            ax[i_ch][i_f].set_xlabel("")
            ax[i_ch][i_f].set_ylabel("")
            ax[0][i_f].set_title(
                feature.upper(), fontweight = "bold"
                )

plt.savefig(
    f"{fig_dir}/nt1_cns_features_SS_subtype_chan_subplots.png", 
    dpi = 300
        )
