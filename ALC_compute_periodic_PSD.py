#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:44:41 2024

@author: arthurlecoz

ALC_periodic_PSD.py

"""
# %% Paths
import mne, os, numpy as np, pandas as pd
import multiprocessing
from glob import glob
from fooof import FOOOF
from fooof.bands import Bands
import pickle
from fooof.sim.gen import gen_aperiodic
from statsmodels.nonparametric.smoothers_lowess import lowess

root_dir = "/Volumes/DDE_ALC/PhD/NT1_HI/PSG"
raw_dir = f"{root_dir}/Raw"
preproc_dir = f"{root_dir}/Preproc/"
fig_dir = f"{root_dir}/Figs/"

hi_cns_files = glob(os.path.join(
    preproc_dir, "hi_cns", "*PSG1*.fif")
    )
n1_files = glob(os.path.join(
    preproc_dir, "*N*PSG1*.fif")
    )

epochs_files = hi_cns_files+n1_files

df_demographics = pd.read_csv(
    "/Volumes/DDE_ALC/PhD/NT1_HI/Demographics/Sujets_SLHIP_clin.csv",
    sep = ";"
    )

subtypes = ["C1", "HI", "N1"]
channels = ["F3", "C3", "O1"]
# psd_palette = ["#5e6472", "#faa307"]
# sem_palette = ['#dee0e2', '#FEECCD']
psd_palette = ["#5e6472", "#faa307"]
sem_palette = ['#dee0e2', '#FEECCD']

freqs = np.linspace(0.5, 40, 159)

bands = Bands({
    "delta" : (.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 12),
    "sigma" : (12, 16),
    "iota" : (25, 35)
    })

method = "welch"
fmin = 0.5
fmax = 40
n_fft = 1024
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

# %% Loop

big_dic_psd_savepath = os.path.join(
    fig_dir, "fooof_psd_flat_spectra_2.pickle"
    )

stages = ["WAKE", "N1", "N2", "N3", "REM"]

# big_dic = {subtype : {stage : {chan : [] for chan in channels}
#                       for stage in stages} for subtype in subtypes}

    
def compute_periodic_psd(file) :
    
    # if subtype.startswith('N') : continue
    sub_id = file.split('/')[-1].split('_PSG1')[0]
    
    this_subject_savepath = os.path.join(
        fig_dir, "fooof", f"{sub_id}_periodic_psd.pickle"
        )
    
    if not os.path.exists(this_subject_savepath) : 
    
        temp_dic = {stage : {chan : [] for chan in channels}
                              for stage in stages}
        
        # thisDemo = df_demographics.loc[df_demographics.code == sub_id]
        # age = thisDemo.age.iloc[0]
        # genre = thisDemo.sexe.iloc[0]
        print(f"...processing {sub_id}")
        
        epochs = mne.read_epochs(file, preload = True)
        epochs = epochs[epochs.metadata.divBin2 != "noBin"]
        # sf = epochs.info['sfreq']
        
        # metadata = epochs.metadata.reset_index()
            
        if len(stages) < 5 :
            print("not adapted yet...")
            # for stage in ["WAKE", "N1", "REM"] :
            #     if stage not in epochs.metadata.scoring.unique() : continue
            #     else :
            #         temp_list = []
            #         temp_power = epochs[stage].compute_psd(
            #                 method = method,
            #                 fmin = fmin, 
            #                 fmax = fmax,
            #                 n_fft = n_fft,
            #                 n_overlap = n_overlap,
            #                 n_per_seg = n_per_seg,
            #                 window = window,
            #                 picks = channels
            #                 )
            #         for i_epoch in range(len(epochs[stage])) :
            #             this_power = temp_power[i_epoch]
            #             for i_ch, channel in enumerate(channels) :
            #                 psd = lowess(np.squeeze(this_power.copy().pick(channel).get_data()), freqs, 0.075)[:, 1]
            #                 fm = FOOOF(peak_width_limits = [1, 15], aperiodic_mode="fixed")
            #                 fm.add_data(freqs, psd)
            #                 fm.fit()
                            
            #                 init_ap_fit = gen_aperiodic(
            #                     fm.freqs, 
            #                     fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
            #                     )
            #                 # Recompute the flattened spectrum using the initial aperiodic fit
            #                 init_flat_spec = fm.power_spectrum - init_ap_fit
            #                 temp_list.append(init_flat_spec)
            #             big_dic[subtype][stage][channel].append(np.nanmean(temp_list, axis = 0))
        
            # temp_list = []
            # temp_power = epochs["N2", "N3"].compute_psd(
            #             method = method,
            #             fmin = fmin, 
            #             fmax = fmax,
            #             n_fft = n_fft,
            #             n_overlap = n_overlap,
            #             n_per_seg = n_per_seg,
            #             window = window,
            #             picks = channels
            #             )
            # for i_epoch in range(len(epochs["N2", "N3"])) :
            #     this_power = temp_power[i_epoch]
            #     for i_ch, channel in enumerate(channels) :
            #         psd = lowess(np.squeeze(
            #             this_power.copy().pick(channel).get_data()), 
            #             freqs, 0.075)[:, 1]
            #         fm = FOOOF(peak_width_limits = [.5, 15], aperiodic_mode="fixed")
            #         fm.add_data(freqs, psd)
            #         fm.fit()
                    
            #         init_ap_fit = gen_aperiodic(
            #             fm.freqs, 
            #             fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
            #             )
            #         # Recompute the flattened spectrum using the initial aperiodic fit
            #         init_flat_spec = fm.power_spectrum - init_ap_fit
            #         temp_list.append(init_flat_spec)
                    
            #     big_dic[subtype]["N2_N3"][channel].append(np.nanmean(temp_list, axis = 0))
        else : 
            for stage in stages :
                print(f'processing stage {stage}')
                if stage not in epochs.metadata.scoring.unique() : continue
                else : 
                    temp_list = []
                    temp_power = epochs[stage].compute_psd(
                            method = method,
                            fmin = fmin, 
                            fmax = fmax,
                            n_fft = n_fft,
                            n_overlap = n_overlap,
                            n_per_seg = n_per_seg,
                            window = window,
                            picks = channels
                            )
                    for i_ch, channel in enumerate(channels) :
                        print(f'processing stage {channel}')
                        for i_epoch in range(len(epochs[stage])) :
                            this_power = temp_power[i_epoch]                        
                            
                            psd = lowess(np.squeeze(
                                this_power.copy().pick(channel).get_data()), 
                                freqs, 0.075)[:, 1]
                            
                            if np.any(psd < 0) :
                                for id_0 in np.where(psd<0)[0] :
                                    psd[id_0] = abs(psd).min()
                                    
                            fm = FOOOF(peak_width_limits = [.5, 4], aperiodic_mode="fixed")
                            fm.add_data(freqs, psd)
                            fm.fit()
                            
                            init_ap_fit = gen_aperiodic(
                                fm.freqs, 
                                fm._robust_ap_fit(fm.freqs, fm.power_spectrum)
                                )
                            
                            init_flat_spec = fm.power_spectrum - init_ap_fit
                            temp_list.append(init_flat_spec)
                        temp_dic[stage][channel].append(
                            np.nanmean(temp_list, axis = 0)
                            )
        with open (this_subject_savepath, 'wb') as handle:
            pickle.dump(temp_dic, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    eeg_files = epochs_files
    
    # Set up a pool of worker processes
    pool = multiprocessing.Pool(processes = 4)
    
    # Process the EEG files in parallel
    pool.map(compute_periodic_psd, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()
    

