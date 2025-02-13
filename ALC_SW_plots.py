#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Arthur_LC

ALC_SW_plots.py

"""
# %%% Paths & Packages

import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import numpy as np
from datetime import date
import os
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from glob import glob
from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
# font
personal_path = '/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/Others/Fonts/aptos-font'
font_path = personal_path + '/aptos-light.ttf'
font = FontProperties(fname=font_path)
bold_font = FontProperties(fname=personal_path + '/aptos-bold.ttf')

todaydate = date.today().strftime("%d%m%y")

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = os.path.join(root_dir, "Raw")
preproc_dir = os.path.join(root_dir, "Preproc")
fig_dir = os.path.join(root_dir, "Figs")

hi_cns_files = glob(os.path.join(
    preproc_dir, "hi_cns", "*PSG1*.fif")
    )
n1_files = glob(os.path.join(
    preproc_dir, "*N*PSG1*.fif")
    )

files = hi_cns_files+n1_files

sub_ids = np.array([file.split('/')[-1].split('_PSG1')[0] for file in files])
sub_ids = np.delete(sub_ids, np.where(sub_ids == "N1_016")[0])

which = "all" # nt1_cns or all

if which == "nt1_cns" :
    df = pd.read_csv(f"{fig_dir}/df_allsw_exgausscrit_nobin_nt1_cns.csv")
elif which == "all" :
    df = pd.read_csv(f"{fig_dir}/df_allsw_exgausscrit_nobin_nt1_hi_cns.csv")
del df['Unnamed: 0']

df = df.loc[df.sub_id.isin(sub_ids)]

#%% SWA | SLEEP | WAKEd, N1, N2, N3, REM

x = "stage"
order = ['WAKEd', 'N1', 'N2', 'N3', 'REM']
y = "density"
hue = "subtype"
if which == "nt1_cns":
    hue_order = ["CTL", "NT1"]
    palette = ["#8d99ae", "#d00000"]
elif which == "all":
    hue_order = ["CTL", "NT1", "HSI"]
    palette = ["#8d99ae", "#d00000", "#ffb703"]
box_width = .2
data = df


fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (8, 6), layout = "tight")

sns.violinplot(
    x = x, order = order, y = y, hue = hue, hue_order = hue_order,
    data = data, palette = palette,
    inner = 'quartile' , cut = 1, ax = ax1
    )

plt.title("SWA according to the stage\nAverage across all channels")
plt.xlabel("Sleep Stage")
plt.xticks(ticks = [0, 1, 2, 3, 4], labels = order)
plt.ylabel("SWA (nSW/nEpochs)")

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/{which}_swdensity.png", 
    dpi = 300
    )

#%% SWA | SLEEP | WAKEb, WAKEd, N1, N2, N3, REM

x = "stage"
order = ['WAKEd', 'N1', 'N2', 'N3', 'REM']
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
    nrows=1, ncols=3, sharex=True, sharey=True, 
    figsize = (24, 12), layout = "tight")

for i_ch, chan in enumerate(["F3", "C3", "O1"]):

    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = data.loc[data.channel == chan], palette = palette,
        inner = 'quartile' , cut = 1, ax = ax[i_ch]
        )
    ax[i_ch].legend_ = None
    ax[i_ch].set_title(chan)
    ax[0].set_ylabel("SWA (nSW/nEpochs)")
    ax[1].set_ylabel("")
    ax[2].set_ylabel("")
    ax[0].set_xlabel("")
    ax[1].set_xlabel("Sleep Stages")
    ax[2].set_xlabel("")

fig.suptitle("SWA according to the stage and channel")
plt.xticks(ticks = [0, 1, 2, 3, 4], labels = order)

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/{which}_swdensity_ch.png", 
    dpi = 300
    )

#%% SWA | SLEEP | N3, O1

x = "stage"
order = ['N3']
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
    # ax[i_ch].set_title(chan)
    ax.set_ylabel("Slow Waves density (nSW/nEpochs)", font = bold_font, fontsize = 24)
    ax.set_xlabel("N3", font = bold_font, fontsize = 24)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        )
    ax.set_yticks(
        np.linspace(0, 16, 5), 
        np.linspace(0, 16, 5), 
        font = font, 
        fontsize = 16
        )
    ax.set_ylim(0, 16)
    sns.despine(bottom=True)
    fig.tight_layout()

# ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/{which}_swdensity_O1_N3.png", 
    dpi = 300
    )

# %% STATS | SWD age + gender

this_df = df.loc[df.stage!="WAKEb"]
feature = 'density'

if which == "nt1_cns":

    foi = ["Stage", "Channel", "ß (iHS_LTS vs HS)", "p_val"]
    dic = {f : [] for f in foi}
    
    for stage in this_df.stage.unique():
        for channel in this_df.channel.unique() :
            model_formula = f'{feature} ~ age + C(genre) + C(subtype, Treatment("CTL")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
            model = smf.mixedlm(model_formula, this_df, groups=this_df['sub_id'])
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
    
elif which == "all":
    
    group_comparison = [["CTL", "NT1"], ["CTL", "HSI"]]
    
    for duo in group_comparison :
        duo_df = this_df.loc[this_df.subtype.isin(duo)]
        foi = [
            "Stage", "Channel", "comparison", "ß", "p_val"
            ]
        dic = {f : [] for f in foi}
        for stage in this_df.stage.unique():
            for channel in this_df.channel.unique() :
                
                model_formula_ctl = f'{feature} ~ age + genre + C(subtype, Treatment("CTL")) * C(channel, Treatment("{channel}")) * C(stage, Treatment("{stage}"))'
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

#%% SWA Wake vs SWA N3

x = "stage"
order = ['WAKEb', 'WAKEd', 'N3']
y1 = "density"
y2 = "density"
y3 = "downward_slope"
hue = "subtype"
hue_order = ["CTL", "HSI"]
box_width = .2
data = df

# palette = ["#5e6472", "#d00000", "#faa307"]
# palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (5, 8))

for i_ch, chan in enumerate(["F3", "C3", "O1"]):
    sns.violinplot(
        x = x, order = order, y = y1, hue = hue, hue_order = hue_order,
        data = data.loc[data.channel == chan], palette = palette,
        inner = 'quartile' , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_yticks(ticks = np.arange(2.5, 17.5, 2.5))
    ax[i_ch].set_ylim(-2, 20)

    ax[i_ch].legend_ = None
    # ax[i_ch].set_ylabel("")
    ax[i_ch].set_xlabel("")
    ax[i_ch].set_xticks(ticks = [], labels = "")
    ax[i_ch].set_yticks(
        ticks = np.arange(0, 22.5, 5),
            # labels = ["" for i in np.arange(2.5, 17.5, 2.5)]
            )
    ax[i_ch].set_title(chan)

# plt.xlabel("Sleep Stage")
plt.xticks(ticks = [0, 1, 2], labels = order)
# plt.ylabel("SWA (nSW/nEpochs)")

fig.suptitle("SW Density in Wake vs N3")
plt.tight_layout(pad = .7)
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/nolegend_density_waked_N3.png", 
    dpi = 300
    )

#%% SWA & PTP | WAKEb | FSC

x = "stage"
order = ['WAKEb']
y1 = "density"
y2 = "ptp"
y3 = "downward_slope"
hue = "subtype"
hue_order = ["CTL", "NT1"]
box_width = .2
data = df

# palette = ["#5e6472", "#d00000", "#faa307"]
# palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize = (6, 8))

for i_ch, chan in enumerate(["F3", "C3", "O1"]):
    sns.violinplot(
        x = x, order = order, y = y1, hue = hue, hue_order = hue_order,
        data = data.loc[data.channel == chan], palette = palette,
        inner = 'quartile' , cut = 1, ax = ax[i_ch][0]
        )
    
    sns.violinplot(
        x = x, order = order, y = y2, hue = hue, hue_order = hue_order,
        data = data.loc[data.channel == chan], palette = palette,
        inner = 'quartile' , cut = 1, ax = ax[i_ch][1]
        )
    sns.violinplot(
        x = x, order = order, y = y3, hue = hue, hue_order = hue_order,
        data = data.loc[data.channel == chan], palette = palette,
        inner = 'quartile' , cut = 1, ax = ax[i_ch][2]
        )
    
    ax[i_ch][0].set_yticks(ticks = np.arange(2.5, 17.5, 2.5))
    ax[i_ch][0].set_ylim(-2.5, 17.5)
    ax[i_ch][1].set_yticks(ticks = np.arange(40, 180, 20))
    ax[i_ch][1].set_ylim(0, 120)
    ax[i_ch][2].set_yticks(ticks = np.arange(200, 1400, 200))
    ax[i_ch][2].set_ylim(0, 1100)
    
    for i in range(3) :
        ax[i_ch][i].legend_ = None
        ax[i_ch][i].set_ylabel("")
        ax[i_ch][i].set_xlabel("")
        ax[i_ch][i].set_xticks(ticks = [], labels = "")
        ax[i_ch][0].set_yticks(
            ticks = np.arange(0, 17.5, 5),
            # labels = ["" for i in np.arange(0, 17.5, 2.5)]
            )
        ax[i_ch][1].set_yticks(
            ticks = np.arange(0, 140, 40),
            # labels = ["" for i in np.arange(0, 140, 20)]
            )
        ax[i_ch][2].set_yticks(
            ticks = np.arange(200, 1200, 200),
            # labels = ["" for i in np.arange(200, 1200, 200)]
            )
        # ax[i_ch][i].spines['right'].set_visible(False)
        # ax[i_ch][i].spines['top'].set_visible(False)
        # ax[i_ch][i].spines['left'].set_visible(False)
        # ax[i_ch][i].spines['bottom'].set_visible(False)
# plt.title("SWA according to the stage\nAverage across all channels")
# plt.xlabel("Sleep Stage")
# plt.xticks(ticks = [0, 1, 2, 3, 4, 5], labels = order)
# plt.ylabel("SWA (nSW/nEpochs)")

plt.tight_layout(pad = 1)
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/nolegend_density_ptp_dslope_WAKEb_FSC.png", 
    dpi = 300
    )

#%% PTP | STAGES | WAKEb, WAKEd, N1, N2, N3, REM

y = "ptp"

fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (8, 16), layout = "tight")

sns.violinplot(
    x = x, order = order, y = y, hue = hue, hue_order = hue_order,
    data = df, palette = palette,
    inner = "quartile" , cut = 1, ax = ax1
    )

plt.title("PTP according to the stage\nAverage across all channels")
plt.xlabel("Sleep Stage")
# plt.xticks(ticks = [0, 1, 2, 3, 4, 5], labels = order)
plt.ylabel("PTP (µV)")

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/RAPH_density_exgauss_{y}_subtype_SLEEP.png", 
    dpi = 300
    )

#%% feature | STAGES | WAKEb, WAKEd, N1, N2, N3, REM

y = "upward_slope"

fig, ax1 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, 
    figsize = (8, 16), layout = "tight")

sns.violinplot(
    x = x, order = order, y = y, hue = hue, hue_order = hue_order,
    data = df, palette = palette,
    inner = "quartile" , cut = 1, ax = ax1
    )

plt.title("Downward Slope according to the stage\nAverage across all channels")
plt.xlabel("Sleep Stage")
plt.xticks(ticks = [0, 1, 2, 3, 4, 5], labels = order)
plt.ylabel("Downward Slope (µV/s-1)")

ax1.legend_ = None
plt.show(block = False)
plt.savefig(
    f"{fig_dir}/RAPH_density_exgauss_{y}_subtype_SLEEP.png", 
    dpi = 300
    )

#%% SWA | STAGES | CHANNELS | SLEEP

channels = ['F3', 'C3', 'O1']
y = "density"

palette = ["#5e6472", "#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title("")
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 6, 1), 
                labels = ["" for i in np.arange(0, 6, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(0, 25, 5), 
            labels = ["" for i in np.arange(0, 25, 5)]
            )
        ax[i].set_ylabel("")
        ax[i].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/nolegend_swdensity_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
    
#%% PTP | STAGES | CHANNELS | SLEEP

channels = ['F3', 'C3', 'O1']
y = "ptp"

palette = ["#5e6472", "#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title("")
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 6, 1), 
                labels = ["" for i in np.arange(0, 6, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(0, 200, 50), 
            labels = ["" for i in np.arange(0, 200, 50)]
          )
        ax[i].set_ylabel("")
        ax[i].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/nolegend_ptp_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
    
#%% d_slope | STAGES | CHANNELS | SLEEP

channels = ['F3', 'C3', 'O1']
y = "downward_slope"

palette = ["#5e6472", "#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title("")
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 6, 1), 
                labels = ["" for i in np.arange(0, 6, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(0, 1400, 200), 
            labels = ["" for i in np.arange(0, 1400, 200)]
          )
        ax[i].set_ylabel("")
        ax[i].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/nolegend_downward_slope_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
    
#%% u_slope | STAGES | CHANNELS | SLEEP

channels = ['F3', 'C3', 'O1']
y = "upward_slope"

palette = ["#5e6472", "#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title("")
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 6, 1), 
                labels = ["" for i in np.arange(0, 6, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(0, 1600, 400), 
            labels = ["" for i in np.arange(0, 1600, 400)]
          )
        ax[i].set_ylabel("")
        ax[i].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/nolegend_upward_slope_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
    
#%% d_slope | STAGES | CHANNELS | SLEEP

channels = ['F3', 'C3', 'O1']
y = "downward_slope"

palette = ["#5e6472", "#d00000", "#faa307"]

fig, ax = plt.subplots(
    nrows=3, ncols=1, sharex=False, sharey=True, 
    figsize = (8, 16), layout = "tight")
for i_ch, chan in enumerate(channels):
    sns.violinplot(
        x = x, order = order, y = y, hue = hue, hue_order = hue_order,
        data = df.loc[df['channel'] == chan], palette = palette,
        inner = "quartile" , cut = 1, ax = ax[i_ch]
        )
    
    ax[i_ch].set_title("")
    for i in range(3) :
        ax[i].set_xlabel("")
        if i == 2 :
            ax[i].set_xticks(
                ticks = np.arange(0, 6, 1), 
                labels = ["" for i in np.arange(0, 6, 1)]
                )
        else : 
            ax[i].set_xticks(
                ticks = [], 
                labels = []
                )
        ax[i].set_yticks(
            ticks = np.arange(0, 1400, 200), 
            labels = ["" for i in np.arange(0, 1400, 200)]
          )
        ax[i].set_ylabel("")
        ax[i].legend_ = None
        
    plt.show(block = False)
    plt.savefig(
        f"{fig_dir}/nolegend_downward_slope_subplot_exgauss_subtype_stage.png", 
        dpi = 300
        )
    

