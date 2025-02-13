#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:11:28 2022

@author: arthurlecoz

=============================================================================
01_Preproc.py

Preprocessing .edf files from the sleep pathology unit

Select first the DataType 
                    
=============================================================================
"""
# %% Paths
# import mne, os, os.path, numpy as np, glob
# from DT_tools_ALC import mne_boundaries
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
fig_dir = root_dir+'/YASA' 

# %% Script
overwrite = False

def preproc_SPU(DataType, raw_dir, preproc_dir, overwrite) :
    import mne, glob, numpy as np, os, pandas as pd
    from DT_tools_ALC import mne_boundaries
    from datetime import date 
    todaydate = date.today().strftime("%d%m%y")

    mapping_type = {'EMG_1':'emg','EMG_2':'emg'}
    
    if DataType == 'PSG' :
        for filename in glob.glob(raw_dir + "/*.edf"):
            sub_id = filename[len(raw_dir)+1:][:-4]
            if not overwrite :
                checkname = (preproc_dir + "/"
                             + sub_id + "_vALC.edf")
                if len(glob.glob(checkname)) != 0 :
                    continue
            # if filename.startswith("HI") or filename.startswith("N2") :
            #     continue
            print("\n...Processing... : ", filename)
            
            raw =  mne.io.read_raw(filename, preload=False, verbose=None)
            
            # Pick channel of interest
            if "Fp1" in raw.ch_names :
                raw.pick([
                'Fp1','C3','O1','A2','EOG D','EOG G','EMG 1','EMG 2'
                ])
                mapping_name = {
                    'EMG 1': 'EMG_1' , 'EMG 2': 'EMG_2', 
                    'EOG D':'EOG_D', 'EOG G':'EOG_G', 
                    'Fp1' : 'F3'
                    }
            else :
                raw.pick([
                'F3','C3','O1','A2','EOG D','EOG G','EMG 1','EMG 2'
                ])
                mapping_name = {
                    'EMG 1': 'EMG_1' , 'EMG 2': 'EMG_2', 
                    'EOG D':'EOG_D', 'EOG G':'EOG_G'
                    }
            # Change names
            raw.rename_channels(mapping_name, verbose=None)
            # Change Channel Type
            raw.set_channel_types(mapping_type, verbose=None)
            # Load data before resampling & filtering
            raw.load_data()
            raw.resample(256, npad="auto")
            
            raw.notch_filter((50, 100))
            raw.filter(0.1, 40, n_jobs = -1)
                    
                    # Rereferencing A2 & EMG
        # / ! \ If you don't specify "copy = False",                            / ! \
        # / ! \ You have to encode the result in a new variable                 / ! \ 
        # / ! \ Otherwise, the result is not taken into acount in the raw file  / ! \
        
            mne.set_eeg_reference(
                raw, ref_channels=['A2'],copy=False, projection=False, 
                ch_type='eeg',forward=None, verbose=None
                ) 
            mne.set_bipolar_reference(
                raw, ['EMG_1'], ['EMG_2'], ch_name='EMG', ch_info=None, 
                drop_refs=True, copy=False, verbose=None
                ) 
            
            # Saving to Preproc
            savename = preproc_dir + "/" + filename[len(raw_dir)+1:][:-4] + "_vALC.edf"
            mne.export.export_raw(
                savename, raw, fmt='edf', overwrite=True
                )
            
            print("\n ...Finished processing...", filename)          
    
    elif DataType == 'TILE' :
        inspect = False
        subid_list = []; boundalength_list = []; boundaries_list = []

        for file in glob.glob(raw_dir + '/*.edf') :
            if not overwrite :
                checkname = preproc_dir + "/" + file.split("/")[-1][:-4] + '_TILE4_vALC.edf'
                if os.path.isfile(checkname):
                    continue
            raw = mne.io.read_raw_edf(file, preload = True)
            sub_id = file[len(raw_dir)+1:][:-4]
            if sub_id.startswith("N1TILE003") :
                continue
            
            # Pick channel of interest
            if "Fp1" in raw.ch_names :
                raw.pick([
                'Fp1','C3','O1','A2','EOG D','EOG G','EMG 1','EMG 2'
                ])
                mapping_name = {
                    'EMG 1': 'EMG_1' , 'EMG 2': 'EMG_2', 
                    'EOG D':'EOG_D', 'EOG G':'EOG_G', 
                    'Fp1' : 'F3'
                    }
            else :
                raw.pick([
                'F3','C3','O1','A2','EOG D','EOG G','EMG 1','EMG 2'
                ])
                mapping_name = {
                    'EMG 1': 'EMG_1' , 'EMG 2': 'EMG_2', 
                    'EOG D':'EOG_D', 'EOG G':'EOG_G'
                    }
            
            # Change names
            raw.rename_channels(mapping_name, verbose=None)
            
            # Change Channel Type
            raw.set_channel_types(mapping_type, verbose=None)
            
            # Resampling & Filtering
            raw.resample(256, npad="auto", n_jobs = -1)
            raw.filter(0.1, 40, n_jobs = -1)
            raw.notch_filter((50, 100), n_jobs = -1)
            
            # Rereferencing A2 & EMG
            mne.set_eeg_reference(
                raw, ref_channels=['A2'],copy=False, projection=False, 
                ch_type='eeg', forward=None, verbose=None)
            mne.set_bipolar_reference(
                raw, ['EMG_1'], ['EMG_2'], ch_name='EMG', ch_info=None, 
                drop_refs=True, copy=False, verbose=None) 
            
            # Hypnogram loading
            textfile = file[:-4] + '.txt'
            hypno = np.loadtxt(textfile, dtype=str)
            d = {'?':'9', 'M':'9', 'U' : '9', 'W':'0', '4':'3', 'R':'4'}
            for keys, values in d.items():
                hypno = np.char.replace(hypno, keys, values)
            if hypno[0]=='Score':
                hypno=hypno[1:]
            
            # TILE segmentation
                # From the np.array to the raw.Annotations
            hypno_event = mne.make_fixed_length_events(
                raw,
                0,
                start = 0,
                stop = None,
                duration = 30
                ) 
            hypno_event[:,2] = hypno[:len(hypno_event)]
            hypno_annot = mne.annotations_from_events(hypno_event, raw.info['sfreq'])
            raw.annotations.append(
                onset = hypno_annot.onset,
                duration = hypno_annot.duration,
                description = hypno_annot.description
                )
            
            boundaries = mne_boundaries(
                raw, 
                hypno_event, 
                fig_dir, 
                sub_id, 
                inspect
                )
                
            subid_list.append(sub_id)
            boundalength_list.append(len(boundaries))
            boundaries_list.append(boundaries)
            
            raw.annotations.delete(np.where(raw.annotations.description))
            
                # Actuel cropping and saving
            for i,limits in enumerate(boundaries) :
                if limits[0] == boundaries[-1][0] :
                    mini_raw = raw.copy().crop(
                        tmin = (limits[0] * 30),
                        tmax = (len(raw)/raw.info['sfreq'])-1
                        )
                else :
                    mini_raw = raw.copy().crop(
                        tmin = (limits[0] * 30),
                        tmax = (limits[1] * 30)
                        )
                mini_hypno = hypno[limits[0] : limits[1]]
                while mini_hypno[0] == '9' :
                    mini_hypno = mini_hypno[1:]
                    mini_raw = mini_raw.crop(
                        tmin = 30,
                        tmax = None
                        )
                while mini_hypno[-1] == '9' :
                    mini_hypno = mini_hypno[:-1]
                    mini_raw = mini_raw.crop(
                        tmin = 0,
                        tmax = len(mini_raw)/ mini_raw.info['sfreq'] - 30
                        )
                
                raw_savename = preproc_dir + "/" + sub_id + "_TILE" + str(i) + "_vALC.edf"
                hyp_savename = raw_savename[:-4] + ".txt"
               
                mne.export.export_raw(
                    raw_savename, mini_raw, fmt='edf', 
                    overwrite = True
                    )
                np.savetxt(
                    hyp_savename, mini_hypno, fmt='%c'
                    )
                print(raw_savename, "... Processed !")
                
                # Dataframe for feedback of preprocessing
        df = pd.DataFrame(
            {
              "SubID" : subid_list,
              "nBoundaries" : boundalength_list,
              "Boundaries" : boundaries_list
              }
            )
        df_savename = (fig_dir + "/TILEs_boundaries_subjects_" 
                       + todaydate + "_.csv")
        df.to_csv(df_savename)
    
    return "\n...ALL FILES WERE CORRECTLY PREPROCESSED ! :)\n"  
       
# %% 

preproc_SPU(DataType, raw_dir, preproc_dir, overwrite) 