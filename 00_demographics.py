#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:48:22 2024

@author: arthurlecoz

00_demographics.py
"""
# %% Paths & Packages

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import mannwhitneyu, chi2_contingency
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

df_sw = pd.read_csv(
    os.path.join(fig_dir, "df_allsw_exgausscrit_nobin_nt1_cns.csv")
    )
del df_sw['Unnamed: 0']

# %% df manip

stages_min = ["NREM1", "NREM2", "NREM3", "REM"]
stages_per = ["N1", "N2", "N3", "R"]

df = df_demographics[[
    'N', 'code', 'diag', 'age', 'sexe', 'poids', 'taille',
    'IMC', 'ESS', 'ivresse', 'dur.inertie', 'sieste.reg',
    'duree.sieste', 'sieste.raf', 'atq.som', 'comp.auto',
    'dur.som.sem', 'dur.som.we', 'hyper.o', 'black.o',
    'hallu', 'para', 'ATCD.dep', 'cata', 'PL', 'HLA', 
    'lat.end.N1', 'lat.SP.N1', 'Lat.SP.15', 'eff.N1', 
    'NREM1.N1', 'N1.N1', 'NREM2.N1', 'N2.N1', 'NREM3.N1', 'N3.N1', 
    'REM.N1', 'R.N1', 'TST.N1', 'TST.C1', 'WASO.N1', 'PST.N1', 
    'ME.N1', 'IMPJ.N1','IAH.N1'
]]

columns_str = [
    'IMC', 'lat.end.N1', 'lat.SP.N1', 'eff.N1', 'NREM1.N1', 'N1.N1',
    'NREM2.N1', 'N2.N1', 'NREM3.N1', 'N3.N1', 'REM.N1', 'R.N1',
    'TST.N1', 'TST.C1', 'WASO.N1', 'PST.N1', 'ME.N1', 'IMPJ.N1',
    'IAH.N1'
]

for column in columns_str:
    df[column] = [float(val.replace(",", ".")) 
                  if type(val) == str 
                  else val 
                  for val in df[column]] 

df.to_csv("/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/lighter_demo.csv")

sub_df = df.loc[df.diag != "HI"]

catecol = [
    'sexe', 'diag', 'ivresse', 'sieste.reg', 'sieste.raf', 'hyper.o', 'black.o'
]
for col in catecol:
    sub_df[col] = pd.Categorical(sub_df[col])

good_ids = df_sw.sub_id.unique()

sub_df = sub_df.loc[sub_df.code.isin(good_ids)]

# %% Basis

col_res = ["Parameter", "NT1", "HS", "U / Chi²", "p_val"]
bigdic_basics = {col: [] for col in col_res}

#### n Subjects
bigdic_basics["Parameter"].append("Number")
bigdic_basics["NT1"].append(sub_df[sub_df.diag == "N1"].shape[0])
bigdic_basics["HS"].append(sub_df[sub_df.diag == "C1"].shape[0])
bigdic_basics["U / Chi²"].append("-")
bigdic_basics["p_val"].append("-")

#### Gender
bigdic_basics["Parameter"].append("M/F")
bigdic_basics["NT1"].append(f"{sub_df.loc[(sub_df.diag == 'N1') & (sub_df.sexe == 1)].shape[0]} / {sub_df.loc[(sub_df.diag == 'N1') & (sub_df.sexe == 0)].shape[0]}")
bigdic_basics["HS"].append(f"{sub_df.loc[(sub_df.diag == 'C1') & (sub_df.sexe == 1)].shape[0]} / {sub_df.loc[(sub_df.diag == 'C1') & (sub_df.sexe == 0)].shape[0]}")
contingency_table = pd.crosstab(sub_df['diag'], sub_df['sexe'])
chi2, p, _, _ = chi2_contingency(contingency_table)
bigdic_basics["U / Chi²"].append(f"{chi2:.2f}")
bigdic_basics["p_val"].append(f"{p:.4f}")

#### Age
bigdic_basics["Parameter"].append("Age")
bigdic_basics["NT1"].append(f"{sub_df[sub_df.diag == 'N1'].age.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'N1'].age.sem():.2f}")
bigdic_basics["HS"].append(f"{sub_df[sub_df.diag == 'C1'].age.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'C1'].age.sem():.2f}")
u_stat, p_val = mannwhitneyu(sub_df[sub_df.diag == 'N1'].age, sub_df[sub_df.diag == 'C1'].age)
bigdic_basics["U / Chi²"].append(f"{u_stat:.2f}")
bigdic_basics["p_val"].append(f"{p_val:.4f}")

#### BMI
bigdic_basics["Parameter"].append("BMI")
this_df = sub_df[["age", "sexe", "diag", "code", "IMC"]]
this_df = this_df.dropna(subset=['IMC'])
bigdic_basics["NT1"].append(
    f"""{sub_df[sub_df.diag == 'N1'].IMC.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'N1'].IMC.sem():.2f}
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
)
bigdic_basics["HS"].append(
    f"""{sub_df[sub_df.diag == 'C1'].IMC.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'C1'].IMC.sem():.2f}
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
u_stat, p_val = mannwhitneyu(this_df[this_df.diag == 'N1'].IMC, this_df[this_df.diag == 'C1'].IMC)
bigdic_basics["U / Chi²"].append(f"{u_stat:.2f}")
bigdic_basics["p_val"].append(f"{p_val:.4f}")

#### ESS
bigdic_basics["Parameter"].append("ESS")
this_df = sub_df[["sexe", "age", "diag", "code", "ESS"]]
this_df = this_df.dropna(subset=['ESS'])
bigdic_basics["NT1"].append(
    f"""{sub_df[sub_df.diag == 'N1'].ESS.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'N1'].ESS.sem():.2f}
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
)
bigdic_basics["HS"].append(
    f"""{sub_df[sub_df.diag == 'C1'].ESS.mean():.2f} \u00B1 {sub_df[sub_df.diag == 'C1'].ESS.sem():.2f}
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
u_stat, p_val = mannwhitneyu(this_df[this_df.diag == 'N1'].ESS, this_df[this_df.diag == 'C1'].ESS)
bigdic_basics["U / Chi²"].append(f"{u_stat:.2f}")
bigdic_basics["p_val"].append(f"{p_val:.4f}")

#### Sleep Drunkeness
bigdic_basics["Parameter"].append("Sleep drunkenness (n, %)")
this_df = sub_df[["sexe", "age", "diag", "code", "ivresse"]]
this_df = this_df.dropna(subset=['ivresse'])
bigdic_basics["NT1"].append(
    f"""{int(np.sum(np.asarray(this_df[this_df.diag == 'N1'].ivresse)))}, {np.sum(np.asarray(this_df[this_df.diag == 'N1'].ivresse))/this_df[this_df.diag == 'N1'].shape[0]*100:.2f}%
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
    )
    
bigdic_basics["HS"].append(
    f"""{int(np.sum(np.asarray(this_df[this_df.diag == 'C1'].ivresse)))}, {np.sum(np.asarray(this_df[this_df.diag == 'C1'].ivresse))/this_df[this_df.diag == 'C1'].shape[0]*100:.2f}%
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
contingency_table = pd.crosstab(this_df['diag'], this_df['ivresse'])
chi2, p, _, _ = chi2_contingency(contingency_table)
bigdic_basics["U / Chi²"].append(f"{chi2:.2f}")
bigdic_basics["p_val"].append(f"{p:.4f}")

#### Regular Nap
bigdic_basics["Parameter"].append("Regular Naps (n, %)")
this_df = sub_df[["sexe", "age", "diag", "code", "sieste.reg"]]
this_df = this_df.dropna(subset=['sieste.reg'])
bigdic_basics["NT1"].append(
    f"""{int(np.sum(np.asarray(this_df[this_df.diag == 'N1']['sieste.reg'])))}, {np.sum(np.asarray(this_df[this_df.diag == 'N1']['sieste.reg']))/this_df[this_df.diag == 'N1'].shape[0]*100:.2f}%
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
)
bigdic_basics["HS"].append(
    f"""{int(np.sum(np.asarray(this_df[this_df.diag == 'C1']['sieste.reg'])))}, {np.sum(np.asarray(this_df[this_df.diag == 'C1']['sieste.reg']))/this_df[this_df.diag == 'C1'].shape[0]*100:.2f}%
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
contingency_table = pd.crosstab(this_df['diag'], this_df['sieste.reg'])
chi2, p, _, _ = chi2_contingency(contingency_table)
bigdic_basics["U / Chi²"].append(f"{chi2:.2f}")
bigdic_basics["p_val"].append(f"{p:.4f}")

#### Dur Som Sem
bigdic_basics["Parameter"].append("Sleep Duration (week) (min)")
this_df = sub_df[["sexe", "age", "diag", "code", "dur.som.sem"]]
this_df = this_df.dropna(subset=['dur.som.sem'])
bigdic_basics["NT1"].append(
    f"""{sub_df[sub_df.diag == 'N1']['dur.som.sem'].mean():.2f} \u00B1 {sub_df[sub_df.diag == 'N1']['dur.som.sem'].sem():.2f}
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
)
bigdic_basics["HS"].append(
    f"""{sub_df[sub_df.diag == 'C1']['dur.som.sem'].mean():.2f} \u00B1 {sub_df[sub_df.diag == 'C1']['dur.som.sem'].sem():.2f}
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
u_stat, p_val = mannwhitneyu(
    this_df[this_df.diag == 'N1']['dur.som.sem'], 
    this_df[this_df.diag == 'C1']['dur.som.sem']
    )
bigdic_basics["U / Chi²"].append(f"{u_stat:.2f}")
bigdic_basics["p_val"].append(f"{p_val:.4f}")

#### Dur Som we
bigdic_basics["Parameter"].append("Sleep Duration (week-end) (min)")
this_df = sub_df[["sexe", "age", "diag", "code", "dur.som.we"]]
this_df = this_df.dropna(subset=['dur.som.we'])
bigdic_basics["NT1"].append(
    f"""{sub_df[sub_df.diag == 'N1']['dur.som.we'].mean():.2f} \u00B1 {sub_df[sub_df.diag == 'N1']['dur.som.we'].sem():.2f}
    value available for {this_df[this_df.diag == 'N1'].shape[0]} participants"""
)
bigdic_basics["HS"].append(
    f"""{sub_df[sub_df.diag == 'C1']['dur.som.we'].mean():.2f} \u00B1 {sub_df[sub_df.diag == 'C1']['dur.som.we'].sem():.2f}
    value available for {this_df[this_df.diag == 'C1'].shape[0]} participants"""
)
u_stat, p_val = mannwhitneyu(
    this_df[this_df.diag == 'N1']['dur.som.we'], 
    this_df[this_df.diag == 'C1']['dur.som.we']
    )
bigdic_basics["U / Chi²"].append(f"{u_stat:.2f}")
bigdic_basics["p_val"].append(f"{p_val:.4f}")

df = pd.DataFrame.from_dict(bigdic_basics)
print(df)

col_signif = []
for value in df.p_val:
    if value == '-' or float(value) > .05:
        col_signif.append('ns')
    elif float(value) < .05 and float(value) > 0.01:
        col_signif.append('*')
    elif float(value) < .01 and float(value) > 0.001:
        col_signif.append('**')
    elif float(value) < .001:
        col_signif.append('***')

df['* or ns'] = col_signif

coi = ['lat.end.N1', 'lat.SP.N1', 'eff.N1', 'TST.N1', 'WASO.N1',
       'N1.N1', 'N2.N1', 'N3.N1', 'R.N1', 'ME.N1', 'IMPJ.N1', 'IAH.N1']

col_res = ["Parameter", "NT1", "HS", "U / Chi²", "p_val"]
bigdic = {col: [] for col in col_res}

for c in coi:
    bigdic['Parameter'].append(str(c)[:-3])
    mean_c = sub_df[['diag', c]].groupby(
        'diag', as_index=False, observed=False).mean()
    sem_c = sub_df[['diag', c]].groupby(
        'diag', as_index=False, observed=False).sem()
    
    bigdic['NT1'].append(
        f"{mean_c[c].loc[mean_c.diag == 'N1'].iloc[0]:.2f} \u00B1 {sem_c[c].loc[sem_c.diag == 'N1'].iloc[0]:.2f}"
    )
    bigdic['HS'].append(
        f"{mean_c[c].loc[mean_c.diag == 'C1'].iloc[0]:.2f} \u00B1 {sem_c[c].loc[sem_c.diag == 'C1'].iloc[0]:.2f}"
    )
    
    u_stat, p_val = mannwhitneyu(
        sub_df[sub_df.diag == 'N1'][c], sub_df[sub_df.diag == 'C1'][c]
        )
    bigdic['U / Chi²'].append(f"{u_stat:.2f}")
    bigdic['p_val'].append(f"{p_val:.5f}")

stats_df = pd.DataFrame.from_dict(bigdic)

col_signif = []
for value in stats_df.p_val:
    if float(value) > .05:
        col_signif.append('ns')
    elif float(value) < .05 and float(value) > 0.01:
        col_signif.append('*')
    elif float(value) < .01 and float(value) > 0.001:
        col_signif.append('**')
    elif float(value) < .001:
        col_signif.append('***')

stats_df['* or ns'] = col_signif

df_stats = pd.concat([df, stats_df])

# Create a table object
x = PrettyTable()

# Add columns
x.field_names = df_stats.columns.tolist()
for row in df_stats.values:
    x.add_row(row)

print(x)

    
    
    
    