import sys
from os import listdir
from os.path import isfile, join
import time
import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io
import iisignature

# set flag for file locations
local = False
# debug flag
debug = True

if local:
    dir_flacs = '/home/paul/Desktop/cache/dale/local_data'
    fnstem_appliance_data = "/home/paul/Desktop/cache/dale/local_data/channel_"

    # for running in gedit
    # sys.path.append('/home/paul/Desktop/cache/dale/code')  
else:
    # dir_flacs = '/scratch/dale_data/UK-DALE-2017/UK-DALE-2017-16kHz/house_1/2016/wk01'
    fnstem_appliance_data = "/scratch/moorep/dale/house_1/channel_"

import read_flac_file as rff



# returns sorted file list from dir_flacs - currently not used
def get_flac_files(dir_flacs):

    onlyfiles = [f for f in listdir(dir_flacs) if isfile(join(dir_flacs, f))]
    return(sorted(onlyfiles))
      


# gets appliance data and returns as a data frame
def get_appliance_data(channel, ts_start, number_of_hours=2):

    if channel == -1:
        fn_dat = "/scratch/moorep/dale/house_1/mains.dat"
    else:
        fn_dat = fnstem_appliance_data + str(channel) + ".dat"

    # set time interval
    ### ts_start = time.mktime(dt.date(date_tuple[0],date_tuple[1],date_tuple[2]).timetuple()) + 3600*start_hour
    ts_end = ts_start + number_of_hours*3600 
    
    string = datetime.utcfromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S and %f microseconds')
    print("Starting time of csv day file is " + string)

    # read appliance data
    
    df = pd.read_csv(fn_dat, ' ')
    df.columns = ['ts','watts']
    df = df.loc[(df['ts']>=ts_start) & (df['ts']<ts_end),:]
    
    return(df)




# finds signature for each cycle in the input voltage
def cycle_sigs(voltage, current, deg, logsig):
    
    # number of channels (dimensions) in path - time, voltage, current
    m = 3

    # find the signature length using an arbitrary path
    sample_path = np.ones([10,m])
    if logsig:
        s = iisignature.prepare(m,deg)
        len_sig = len(iisignature.logsig(sample_path,s))
    else:
        len_sig = len(iisignature.sig(sample_path,deg))
        
        
    # find zero up-crossing indexes for voltage
    zci = np.where(np.diff(np.sign(voltage))>0)[0]
    
#debug
    #print('Limiting zci')
    #zci = zci[0:100000]

    sigs = np.empty([len(zci)-1,len_sig]);

    # create a path for each cycle 
    for k,z in enumerate(zci[0:-1]):

        ch1 = voltage[zci[k]:zci[k+1]]
        ch2 = current[zci[k]:zci[k+1]]
        ch0 = np.linspace(0,1,ch1.shape[0]) # time dimension

        path = np.column_stack([ch0, ch1, ch2])
        #pt.plot(path)

        # create the signature using iisignature
        if logsig:
            sigs[k,:] = iisignature.logsig(path,s)
        else:
            sigs[k,:]  = iisignature.sig(path,deg)  
        
    # drop the final zci because it is at the cycle end
    zci = zci[0:-1]
    return(zci, sigs)


# creates the training set given appliance dataframe, cycle signatures and their timestamps
def create_labelled_set(df_appliance, sigs, ts_sigs):
    if debug:
        sigs = np.column_stack([sigs, ts_sigs])
        
    # margin in seconds from the appliance's switch on and off times
    margin = 10

    # preallocate array and truncate later
    labelled_set = np.zeros((sigs.shape[0],sigs.shape[1]+1));

    # create the on/off labels by detecting when the fridge uses more than 20 watts
    df_appliance['on'] = df_appliance['watts'] > 20

    # find the indexes for when the appliance turns on and off
    on_off_indexes = np.where(np.diff(df_appliance['on']))[0]
    print("State switches %d times."%len(on_off_indexes))
    
    c = 0
    
    # set the first label - subsequent labels are obtained by toggling
    label = int(df_appliance['on'].iloc[0])
    
    sig_len = sigs.shape[1]
    ts_appliance = df_appliance['ts']

    # iterate over the appliance on-off periods
    for idx, ooi in np.ndenumerate(on_off_indexes[0:-1]):

        # find the on or off time and select a clean on or off period
        ooi_next = on_off_indexes[idx[0]+1]
        ts1 = ts_appliance.iloc[ooi]+margin
        ts2 = ts_appliance.iloc[ooi_next]-margin

        # find the indexes into the list of cycle timestamps for the period
        s_indexes = np.where((ts_sigs>=ts1) & (ts_sigs<=ts2))[0]

        if len(s_indexes) > 0:
        # create labelled set, appending labels to the end of the signatures
            labelled_set[c:c+len(s_indexes),0:sig_len] = sigs[s_indexes]
            labelled_set[c:c+len(s_indexes),sig_len] = label
            c = c+len(s_indexes)
            
        label = 1-label ## I think this should be toggled every time, might check
            
    # trucate labelled_set
    labelled_set = labelled_set[:c,:]

    return(labelled_set)

        
# main function
def get_labelled_flac_file(fn_flac, week, signature_degree=2, logsig=True, convert_va=True):
    dir_flacs = '/scratch/dale_data/UK-DALE-2017/UK-DALE-2017-16kHz/house_1' + week

    # choose appliance - number 12 is the fridge
    appliance_number = 12

    # display flac date
    ts_start = float(fn_flac[3:13])+float(fn_flac[14:20])/1e6
    date = datetime.utcfromtimestamp(ts_start)
    string = datetime.utcfromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S and %f microseconds')
    print("Starting time of flac file is " + string)

    # get appliance data for the chosen day - 4 Jan 2016 which is the first recording in 2016
    # note that the appliance data is irregularly sampled, with increments of typically 6,7,8 seconds
    df_appliance = get_appliance_data(appliance_number, ts_start)
    
    # read the flac file (one hour of measurement), find signatures, find the timestamps  
    (scaled_voltage, scaled_current) = rff.read_flac(dir_flacs + "/" + fn_flac, convert_va)
    (zci, sigs) = cycle_sigs(scaled_voltage, scaled_current, signature_degree, logsig)
    ts_sigs = ts_start + zci/16000

    
    # create set of signatures and labels
    labelled_sigs = create_labelled_set(df_appliance, sigs, ts_sigs)
    print(labelled_sigs.shape)
    return labelled_sigs

def version2(fn_flac, week, signature_degree=2, logsig=True, convert_va=True):
    dir_flacs = '/scratch/dale_data/UK-DALE-2017/UK-DALE-2017-16kHz/house_1' + week
    #print("Reading flac file..")
    (voltage,current) = rff.read_flac(dir_flacs + "/" + fn_flac, convert_va)
    
    ts_start = float(fn_flac[3:13])+float(fn_flac[14:20])/1e6
    date = datetime.utcfromtimestamp(ts_start)
    string = datetime.utcfromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S and %f microseconds')
    print("Starting time of flac file is " + string)
    
    print("Constructing signatures..")
    (zci, sigs) = cycle_sigs(voltage,current, signature_degree, logsig)

    # find the timestamps for the start of each cycle - each sample takes 1/16000 seconds
    ts_sigs = ts_start + zci/16000

    fn_dat = "/scratch/moorep/dale/house_1/channel_12.dat"
    #start_hour = 0
    number_of_hours = 1
    ts_end = ts_start + number_of_hours*3600 + 1800
    print("Starting time of csv file is " + string)

    # read appliance data
    df = pd.read_csv(fn_dat, ' ')    
    df.columns = ['ts','watts']
    df = df.loc[(df['ts']>=ts_start) & (df['ts']<ts_end),:]
        
    #labelling the dataframe with watts
    lst = []
    for idx, ts in np.ndenumerate(df['ts']):
        s_len = sigs.shape[1]
        power = (df.iloc[idx]['watts'])
        
        #get signatures at ts
        s_indexes = np.where((ts_sigs>=ts-1) & (ts_sigs<=ts+1))[0]
        
        if len(s_indexes) > 0:
            if debug:
                s = np.zeros(s_len+2)
                s[s_len+1] = ts # for debug
            else:
                s = np.zeros(s_len+1)
            s[:s_len] = sigs[s_indexes[0]]
            s[s_len] = 1 if power > 20 else 0
            lst.append(s)
   
    try:
        filtered_set = np.row_stack(lst)
        print(filtered_set.shape)
        return filtered_set
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")



# function
if __name__ == "__main__":
    sys.path.append("..")
    copy = label_flac_file()
    print(copy.shape,"Done")