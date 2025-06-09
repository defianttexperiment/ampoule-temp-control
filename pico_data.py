import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.signal import find_peaks, savgol_filter

"""
def combine_csv_files(, output_file):
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not all_files:
        print("No CSV files found in the directory.")
        return

    df_list = []
    for file in all_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        df.drop(axis=0,index=[0,1])
        print(df)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
"""

# Load all finals in waveforms directory
directory_path = os.getcwd() + '/20250528'
all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
if not all_files:
    print("No CSV files found in the directory.")

# Initializations
fulltimedata = []
fullchanneldata = []
df_list = []

# Add time & data from every file
for file in all_files:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)

    strtimedata = data['Time'].tolist()
    strtimedata = strtimedata[2:]
    timedata = [float(s) for s in strtimedata]
    if len(fulltimedata) > 0: # might be buggy
        for i in range(len(timedata)):
            timedata[i] = timedata[i] + fulltimedata[-1]
    fulltimedata.extend(timedata)

    strchanneldata = data['Channel A'].tolist()
    strchanneldata = strchanneldata[2:]
    channeldata = []
    for s in strchanneldata:
        try:
            channeldata.append(float(s))
        except ValueError:
            channeldata.append(channeldata[-1])
    fullchanneldata.extend(channeldata)


interval = 10 # Factor of reduction, change to remove more or less data

# Condense channel & time data
smchanneldata = []
smtimedata = []
for i in range(int(len(fullchanneldata)/interval)):
    start = interval*i
    end = interval*i+(interval-1)
    smchanneldata.append(mean(fullchanneldata[start:end]))
    smtimedata.append(fulltimedata[start])

# Apply smoothing to channel data
smoothed_data = savgol_filter(smchanneldata, window_length=101, polyorder=3)



# Find peaks! Divide into segments to change prominences
segment1 = slice(0, len(data)//3)  # First third with clear oscillations
segment2 = slice(len(data)//3, 2*len(data)//3)  # Middle section
segment3 = slice(2*len(data)//3, None)  # Last third with smaller variations

all_peaks = []
all_troughs = []

# Different parameters for different segments
segments = [segment1, segment2, segment3]
prominences = [1, 1, 1]

for seg, prom in zip(segments, prominences):
    # Find peaks in segment
    peaks, _ = find_peaks(smoothed_data[seg], 
                         prominence=prom,
                         distance=200)
    
    # Adjust indices to full data
    if seg.start:
        peaks += seg.start
    all_peaks.extend(peaks)
    
    # Find troughs
    troughs, _ = find_peaks(-smoothed_data[seg],
                           prominence=prom,
                           distance=200)
    if seg.start:
        troughs += seg.start
    all_troughs.extend(troughs)

all_peaks_times = []
all_troughs_times = []
for peak in all_peaks:
    all_peaks_times.append(smtimedata[peak])
for trough in all_troughs:
    all_troughs_times.append(smtimedata[trough])

print(all_peaks_times)

def sliding_window_peak_rate(all_peaks_times, time_array, window_size=1):
    """
    Calculate peak rate using a true sliding window approach.
    
    Parameters:
    -----------
    all_peaks_times : list or array
        Timestamps of all detected peaks
    time_array : array
        Full time array for output
    window_seconds : float
        Window size in seconds (default: 30)
        
    Returns:
    --------
    peak_rate : array
        Peak rate (peaks per second) at each time point
    """
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



# Import temperature data
data = pd.read_csv("20250528Tdata.csv")
temp_data = data['TSic AIN0 (°C)'].tolist()
smooth_temp_data = savgol_filter(temp_data, window_length=120, polyorder=3)
temp_time_data = data['Time (s)'].tolist()

for n in range(len(temp_time_data)):
    temp_time_data[n] = temp_time_data[n]-40

# Create temperature data according to time
temp_index = 0
timed_temp_data = []
for t in smtimedata:
    if temp_time_data[temp_index+1] < t:
        temp_index = temp_index+1
    t_diff_below = t - temp_time_data[temp_index]
    t_diff_above = temp_time_data[temp_index+1] - t
    timed_temp_data.append((smooth_temp_data[temp_index]*t_diff_above + smooth_temp_data[temp_index+1]*t_diff_below)/(t_diff_above + t_diff_below))
print(len(timed_temp_data))

# Convert times to temps
all_peaks_temps = []
all_troughs_temps = []
for peak_time in all_peaks_times:
    peak_index = temp_time_data.index(min(temp_time_data, key=lambda x: abs(x-peak_time)))
    rough_peak_temp = smooth_temp_data[peak_index]
    all_peaks_temps.append(rough_peak_temp)
for trough_time in all_troughs_times:
    trough_index = temp_time_data.index(min(temp_time_data, key=lambda x: abs(x-trough_time)))
    rough_trough_temp = smooth_temp_data[trough_index]
    all_troughs_temps.append(rough_trough_temp)

# Plot figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: actual & smoothed data
ax1.plot(timed_temp_data, smchanneldata, label='Raw data', alpha=0.7)
ax1.plot(timed_temp_data, smoothed_data, label='Smoothed data', linewidth=2)
ax1.vlines(all_peaks_temps, 0, 10, colors="red", alpha=0.5, label='Peaks')
ax1.vlines(all_troughs_temps, 0, 10, colors="green", alpha=0.5, label='Troughs')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal Value')
ax1.set_title('Signal Data with Detected Peaks and Troughs')
ax1.legend()

# Right plot: peak density
peak_rate = sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.5)
smooth_peak_rate = savgol_filter(peak_rate, window_length=3000, polyorder=3)
ax2.plot(timed_temp_data, smooth_peak_rate, 'b-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Peaks per second')
ax2.set_title('Peak Density (40s window)')

plt.tight_layout()
plt.show()