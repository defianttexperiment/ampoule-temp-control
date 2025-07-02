import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
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
# Ways to use this function
show_temp_data = False # Displays temp data instead of peaks data

# Load all finals in waveforms directory
directory_path = os.getcwd() + '/20250625'
try:
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found in the directory.")
except:
    print("No directory found.")

# Initializations
fulltimedata = []
fullchanneldata = []
df_list = []
csv_name = '20250625data.csv'

# Change window of data used. 
start_time = 600
end_time = 100000

try:
    initial_data = pd.read_csv(csv_name)
    fulltimedata = initial_data['Time'].tolist()
    fulltimedata.pop(0)
    fullchanneldata = initial_data['Channel A'].tolist()
    fullchanneldata.pop(0)
except:
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

    finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata)), columns = ['Time', 'Channel A'])
    finaldf.to_csv(csv_name, index=False)


interval = 10 # Factor of reduction, change to remove more or less data

# Condense channel & time data
smchanneldata = []
smtimedata = []
for i in range(int(len(fullchanneldata)/interval)):
    start = interval*i
    end = interval*i+(interval-1)
    smchanneldata.append(mean(fullchanneldata[start:end]))
    smtimedata.append(fulltimedata[start])

if smtimedata[0] < start_time:
    print(len(smtimedata))
    for i in range(len(smtimedata)):
        if smtimedata[i] < start_time:
            continue
        else:
            smtimedata = smtimedata[i:]
            smchanneldata = smchanneldata[i:]
            print("Cut early at index %s" % i)
            break
if smtimedata[-1] > end_time:
    for i in range(len(smtimedata)):
        if smtimedata[len(smtimedata)-1-i] > end_time:
            continue
        else:
            smtimedata = smtimedata[:len(smtimedata)-1-i]
            smchanneldata = smchanneldata[:len(smchanneldata)-1-i]
            print("Cut late at index %s" % i)
            break

# Apply smoothing to channel data
smoothed_data = savgol_filter(smchanneldata, window_length=201, polyorder=3)



# Find peaks! Divide into segments to change prominences
segment1 = slice(0, len(smoothed_data)//3)  # First third with clear oscillations
segment2 = slice(len(smoothed_data)//3, 2*len(smoothed_data)//3)  # Middle section
segment3 = slice(2*len(smoothed_data)//3, None)  # Last third with smaller variations

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
raw_temp_data = pd.read_csv("20250625Tdata.csv")
temp_data = raw_temp_data['TSic AIN0 (°C)'].tolist()
temp_time_data = raw_temp_data['Time (s)'].tolist()

# Correcting for 0.2C bias in TSic placement
for i in range(len(temp_data)):
    temp_data[i] = temp_data[i]-0.2

# Remove data outside the window we want
if temp_time_data[0] < start_time:
    for i in range(len(temp_time_data)):
        if temp_time_data[i] < start_time:
            continue
        else:
            temp_time_data = temp_time_data[i:]
            temp_data = temp_data[i:]
            break
if temp_time_data[-1] > end_time:
    for i in range(len(temp_time_data)):
        if temp_time_data[len(temp_time_data)-1-i] > end_time:
            continue
        else:
            temp_time_data = temp_time_data[:len(temp_time_data)-1-i]
            temp_data = temp_data[:len(temp_data)-1-i]
            break

smooth_temp_data = savgol_filter(temp_data, window_length=400, polyorder=3)

if show_temp_data:
    plt.plot(temp_time_data, temp_data, temp_time_data, smooth_temp_data)
    plt.show()

# Create temperature data according to time
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
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Signal Value')
ax1.set_title('Signal Data with Detected Peaks and Troughs')
ax1.legend()

# Right plot: peak density
peak_rate = sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.2)
smooth_peak_rate = savgol_filter(peak_rate, window_length=5000, polyorder=3)
ax2.plot(timed_temp_data, smooth_peak_rate, 'b-', linewidth=2)
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Peaks per degree')
ax2.set_title('Peak Density (0.2°C window)')

# Add debug prints to understand what's happening
print(f"Signal data range: {smtimedata[0]:.1f} to {smtimedata[-1]:.1f}")
print(f"Temperature data range: {temp_time_data[0]:.1f} to {temp_time_data[-1]:.1f}")
print(f"Target range: {start_time} to {end_time}")
print(f"Number of peaks found: {len(all_peaks_times)}")
print(f"Peak times: {all_peaks_times[:5]}...")  # First 5 peaks
print(f"Signal data length: {len(smtimedata)}")
print(f"Temperature data length: {len(timed_temp_data)}")

# Check if peak indices are valid
print(f"Peak indices range: {min(all_peaks) if all_peaks else 'No peaks'} to {max(all_peaks) if all_peaks else 'No peaks'}")
print(f"Max valid index: {len(smtimedata)-1}")

plt.tight_layout()
plt.show()