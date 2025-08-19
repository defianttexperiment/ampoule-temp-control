import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter, morlet

# ---------------- MANUAL CONFIGURATION ----------------
# Ways to use this function
file_name = '0703_swing_bottom'
temp_input_is_raw_data = True
show_peak_lines = True
use_channel_b = False
interval = 10 # Factor of data compression, change to remove more or less data
half_life = 10 # Factor of drop-off for smoothing raw data
peak_prominence = 4 # Height in mV required to be labeled as a peak
plus_40_correction = True # Recent data has -40 mV as a baseline for some reason

# Change window of data used.
start_time = 1800
end_time = 3100

# ---------------- IMPORT CHANNEL DATA ----------------
# Load all finals in waveforms directory
directory_path = os.getcwd() + "/" + file_name # + '/data_archive/'
try:
    test_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
    if os.path.basename(test_files[0])[9:11] == "00":
        all_files = sorted(
            (f for f in os.listdir(directory_path) if f.lower().endswith('.csv')),
            key=lambda f: int(os.path.basename(f)[14:16])  # digits 15 & 16 (1-based)
        )
    else:
        all_files = sorted(
            (f for f in os.listdir(directory_path) if f.lower().endswith('.csv')),
            key=lambda f: int(os.path.basename(f)[9:11])  # digits 10 & 11 (1-based)
        )
    print(all_files)
    if not all_files:
        print("No CSV files found in the directory.")
except Exception as e:
    print(f"Exception {e}: Files already loaded.")

# Initializations
fulltimedata = []
fullchanneldata = []
fullchannelbdata = []
smchanneldata = []
smchannelbdata = []
smtimedata = []
df_list = []
csv_name = file_name + 'data.csv'
sm_csv_name = file_name + 'smdata.csv'

try:
    sm_data = pd.read_csv(sm_csv_name)
    smtimedata = sm_data['Time'].tolist()
    smtimedata.pop(0)
    smchanneldata = sm_data['Channel A'].tolist()
    smchanneldata.pop(0)
    smchannelbdata = sm_data['Channel B'].tolist()
    smchannelbdata.pop(0)
except:
    try:
        initial_data = pd.read_csv(csv_name)
        fulltimedata = initial_data['Time'].tolist()
        fulltimedata.pop(0)
        fullchanneldata = initial_data['Channel A'].tolist()
        fullchanneldata.pop(0)
        if use_channel_b:
            try:
                fullchannelbdata = initial_data['Channel B'].tolist()
                fullchannelbdata.pop(0)
            except:
                print('No Channel B data found.')
    except:
        print("Loading data from files...")
        # Add time & data from every file
        for file in all_files:
            file_path = os.path.join(directory_path, file)
            initial_data = pd.read_csv(file_path)

            strtimedata = initial_data['Time'].tolist()
            strtimedata = strtimedata[2:]
            timedata = [float(s) for s in strtimedata]
            if len(fulltimedata) > 0: # might be buggy
                for i in range(len(timedata)):
                    timedata[i] = timedata[i] + fulltimedata[-1]
            fulltimedata.extend(timedata)

            strchanneldata = initial_data['Channel A'].tolist()
            strchanneldata = strchanneldata[2:]
            channeldata = []
            for s in strchanneldata:
                try:
                    channeldata.append(float(s))
                except ValueError:
                    channeldata.append(channeldata[-1])
            fullchanneldata.extend(channeldata)

            if use_channel_b:
                try:
                    strchannelbdata = initial_data['Channel B'].tolist()
                    strchannelbdata = strchannelbdata[2:]
                    channelbdata = []
                    for s in strchannelbdata:
                        try:
                            channelbdata.append(float(s))
                        except ValueError:
                            channelbdata.append(channelbdata[-1])
                    fullchannelbdata.extend(channelbdata)
                except:
                    print("No Channel B data found.")

        finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata, fullchannelbdata)), columns = ['Time', 'Channel A', 'Channel B'])
        finaldf.to_csv(csv_name, index=False)
    
    print("Condensing data...")
    for i in range(int(len(fullchanneldata)/interval)):
        start = interval*i
        end = interval*i+(interval-1)
        smchanneldata.append(mean(fullchanneldata[start:end]))
        if use_channel_b:
            smchannelbdata.append(mean(fullchannelbdata[start:end]))
        smtimedata.append(fulltimedata[start])
    smdf = pd.DataFrame(list(zip(smtimedata, smchanneldata, smchannelbdata)), columns = ['Time', 'Channel A', 'Channel B'])
    smdf.to_csv(sm_csv_name, index=False)

# ---------------- EDITING DATA ----------------
print("Cutting data to given start/end times...")
# Cut channel & time data to time constraints
if smtimedata[0] < start_time:
    print(len(smtimedata))
    for i in range(len(smtimedata)):
        if smtimedata[i] < start_time:
            continue
        else:
            smtimedata = smtimedata[i:]
            smchanneldata = smchanneldata[i:]
            if use_channel_b:
                smchannelbdata = smchannelbdata[i:]
            print("Cut early at index %s" % i)
            break
if smtimedata[-1] > end_time:
    for i in range(len(smtimedata)):
        if smtimedata[len(smtimedata)-1-i] > end_time:
            continue
        else:
            smtimedata = smtimedata[:len(smtimedata)-1-i]
            smchanneldata = smchanneldata[:len(smchanneldata)-1-i]
            if use_channel_b:
                smchannelbdata = smchannelbdata[:len(smchannelbdata)-1-i]
            print("Cut late at index %s" % i)
            break

# Apply smoothing to channel data
print("Smoothing data...")
smoothed_data = savgol_filter(smchanneldata, window_length=101, polyorder=3)
if use_channel_b:
    smoothed_b_data = savgol_filter(smchannelbdata, window_length=101, polyorder=3)

# ---------------- IMPORT TEMPERATURE DATA ----------------
# Import temperature data
print("Importing temperature data...")
raw_temp_data = pd.read_csv(file_name + 'Tdata.csv')
try:
    temp_data = raw_temp_data['TSic AIN0 (°C)'].tolist()
except:
    temp_data = raw_temp_data['TSic AIN2 (°C)'].tolist()
original_temp_data = temp_data
temp_time_data = raw_temp_data['Time (s)'].tolist()

# Smooth out raw temperature data.
print("Smoothing temperature data...")
sum_numerator = 0
sum_denominator = 0
if temp_input_is_raw_data:
    print("Smoothing raw temperature data.")
    smoothing_range = 200 # averages over this range *before* the given datum
    placeholder_temp_data = []
    for i in range(len(temp_data)):
        sum_numerator = 0
        sum_denominator = 0
        start = max(0, i-smoothing_range)  # Cleaner way to handle negative start
        
        try: 
            # Theory: Do a weighted average of the most recent smoothing_range data points with exponential 0.5^(t/20).
            for j in range(start, i+1):  # Simplified - just iterate from start to current index
                time_diff = i - j  # How far back this point is
                weight = 0.5**(time_diff/half_life)
                sum_numerator += temp_data[j] * weight
                sum_denominator += weight
            
            # Add safety check for division by zero
            if sum_denominator > 0:
                placeholder_temp_data.append(sum_numerator/sum_denominator)
            else:
                placeholder_temp_data.append(temp_data[i])  # Fallback to original value
                
        except Exception as e: 
            print(e)
            placeholder_temp_data.append(temp_data[i])
    temp_data = placeholder_temp_data
# Remove data outside the previously defined window
print("Cutting temperature data to given start/end times...")
if temp_time_data[0] < start_time:
    for i in range(len(temp_time_data)):
        if temp_time_data[i] < start_time:
            continue
        else:
            temp_time_data = temp_time_data[i:]
            temp_data = temp_data[i:]
            original_temp_data = original_temp_data[i:]
            break
if temp_time_data[-1] > end_time:
    for i in range(len(temp_time_data)):
        if temp_time_data[len(temp_time_data)-1-i] > end_time:
            continue
        else:
            temp_time_data = temp_time_data[:len(temp_time_data)-1-i]
            temp_data = temp_data[:len(temp_data)-1-i]
            original_temp_data = original_temp_data[:len(original_temp_data)-1-i]
            break

smooth_temp_data = savgol_filter(temp_data, window_length=120, polyorder=3) # originally 120 with non-raw data

# Create temperature data according to time
print("Extending temperature data...")
temp_index = 0
timed_temp_data = []
for t in smtimedata:
    # Find correct temperature index with bounds checking
    while (temp_index < len(temp_time_data) - 2 and 
           temp_time_data[temp_index + 1] < t):
        temp_index += 1
    
    # Ensure we stay within bounds
    if temp_index < len(temp_time_data) - 1:
        t_diff_below = t - temp_time_data[temp_index]
        t_diff_above = temp_time_data[temp_index + 1] - t
        interpolated_temp = (smooth_temp_data[temp_index] * t_diff_above + 
                           smooth_temp_data[temp_index + 1] * t_diff_below) / (t_diff_above + t_diff_below)
        timed_temp_data.append(interpolated_temp)
    else:
        # Use last available temperature if out of bounds
        timed_temp_data.append(smooth_temp_data[-1])

# ---------------- FIND PEAKS ----------------
print("Finding peaks...")
all_peaks = []
peak_file_path = os.getcwd() + "/" + file_name + "peaks.csv"

# Defining wavelet for find_peaks_cwt
morlet_wavelet = lambda M, s: morlet(M, w=5, s=s, complete=False)

all_peaks, _ = find_peaks(smoothed_data, prominence=peak_prominence, distance=200)
# peaks = find_peaks_cwt(smoothed_data, np.arange(30,500), min_snr = 0.1) # Uses wavelet matching

all_peaks_times = []
for peak in all_peaks:
    all_peaks_times.append(smtimedata[peak])
    
# Convert times to temps
all_peaks_temps = []
for peak_time in all_peaks_times:
    peak_index = temp_time_data.index(min(temp_time_data, key=lambda x: abs(x-peak_time)))
    rough_peak_temp = smooth_temp_data[peak_index]
    all_peaks_temps.append(rough_peak_temp)
    
# Find peak rate over a sliding window
def sliding_window_peak_rate(all_peaks_times, time_array, window_size=1):
        # Convert to numpy array if it's a list
        peak_times = np.array(all_peaks_times)
        
        peak_rate = np.zeros_like(time_array)
        
        for i, t in enumerate(time_array):
            # Count peaks within window centered at current time
            window_start = t - window_size/2
            window_end = t + window_size/2
            
            peaks_in_window = np.sum((peak_times >= window_start) & 
                                    (peak_times <= window_end))
            
            # Convert to rate (peaks per second)
            peak_rate[i] = peaks_in_window / window_size
        
        return peak_rate

peak_rate = sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.1)

# ---------------- PLOT FIGURES ----------------
print("Plotting figures...")
fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(16, 6))
ax3 = ax2.twinx() 

# Left plot: actual & smoothed data

ax1.plot(smtimedata, smchanneldata, label='Raw data', alpha=0.2)
ax1.plot(smtimedata, smoothed_data, label='Smoothed data', linewidth=2)
if use_channel_b:
    ax1.plot(smtimedata, smoothed_b_data, label='Scattering data', linewidth=2)
if show_peak_lines:
    if plus_40_correction:
        ax1.vlines(all_peaks_times, -40, 20, colors="grey", alpha=0.4, label='Peaks')
    else:
        ax1.vlines(all_peaks_times, 0, 20, colors="grey", alpha=0.4, label='Peaks')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal Value')
ax1.set_title('Signal Data with Detected Peaks')
ax1.legend()

ax5.plot(timed_temp_data, smchanneldata, label='Raw data', alpha=0.2)
ax5.plot(timed_temp_data, smoothed_data, label='Smoothed data', linewidth=2)
if use_channel_b:
    ax5.plot(timed_temp_data, smoothed_b_data, label='Scattering data', linewidth=2)
if show_peak_lines:
    if plus_40_correction:
        ax5.vlines(all_peaks_temps, -40, 20, colors="grey", alpha=0.4, label='Peaks')
    else:
        ax5.vlines(all_peaks_temps, 0, 20, colors="grey", alpha=0.4, label='Peaks')
ax5.set_xlabel('Temperature (°C)')
ax5.set_ylabel('Signal Value')
ax5.set_title('Signal Data with Detected Peaks')

# Calculate derivative for second plot
if use_channel_b:
    b_rate_of_change = []
    for i in range(len(smoothed_b_data)):
        try:
            b_rate_of_change.append((smoothed_b_data[i+250]-smoothed_b_data[i])*100)
        except:
            continue

# Right plot: peak density
smooth_peak_rate = savgol_filter(peak_rate, window_length=5000, polyorder=3)
ax2.plot(timed_temp_data, peak_rate, 'b-', linewidth=2)
if use_channel_b:
    ax3.plot(timed_temp_data[250:], b_rate_of_change, label='Scattering data', linewidth=2, alpha=0.4)
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Peaks per degree')
ax3.set_ylabel('Differential of scattering data')
ax2.set_title('Peak Density (0.1°C window)')

ax4.plot(temp_time_data, original_temp_data, temp_time_data, temp_data, temp_time_data, smooth_temp_data)
ax4.set_ylabel('Temperature (°C)')
ax4.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()