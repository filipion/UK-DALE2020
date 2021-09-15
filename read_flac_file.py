#
# Purpose: Read voltage and current from a Dale flac file.
#
# Input          
#           
# Effects: 
#
# Usage examples
#
# (c) 2020 Paul Moore - moorep@maths.ox.ac.uk 
#
# This software is provided 'as is' with no warranty or other guarantee of
# fitness for the user's purpose.  Please let the author know of any bugs
# or potential improvements.

import numpy as np
import soundfile as sf

# House 1 calibration - https://jack-kelly.com/data/ for conversion method
volts_per_adc_step = 1.90101491444e-07
amps_per_adc_step = 4.9224284384e-08
number_of_ADC_steps = 2**31 


# Reads a flac file and returns voltage and current.  If return_vi is true, converts to volts and amps.
def read_flac(fn_flac, return_vi):

    # read the flac file using soundfile
    data, samplerate = sf.read(fn_flac)    

    # derive voltage and current
    voltage = data[:,0]     
    current = data[:,1]

    if return_vi:
        voltage = voltage * volts_per_adc_step * number_of_ADC_steps
        current = current * amps_per_adc_step * number_of_ADC_steps

    return((voltage,current))


# Reads voltage and current from a flac file starting from signal_start with length signal_length, both in seconds
def read_flac_segment(fn_flac, signal_start, signal_length):

    # find frame indexes from start and length
    samplerate = 16000
    sig_start = round(signal_start*samplerate)
    sig_end = sig_start + round(signal_length*samplerate)

    # read the flac file using soundfile
    data, samplerate = sf.read(fn_flac, start=sig_start, stop=sig_end)    

    # derive voltage and current
    voltage = data[:,0] * volts_per_adc_step * number_of_ADC_steps
    current = data[:,1] * amps_per_adc_step * number_of_ADC_steps

    return((voltage,current))




