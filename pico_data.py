"""
PicoScope Data Analysis Script
============================

This script analyzes temperature-dependent optical data collected from a PicoScope oscilloscope.
It processes light intensity measurements (Channel A) and optional scattering data (Channel B)
along with temperature measurements to identify peaks and analyze temperature-dependent behavior.

TYPICAL USE CASE:
- Analyzing optical properties of materials as they are heated/cooled
- Identifying phase transitions or other temperature-dependent phenomena
- Peak detection in noisy optical signals

INPUT DATA EXPECTED:
1. PicoScope CSV files in a subdirectory (raw waveforms)
   OR pre-processed data files: {filename}data.csv, {filename}smdata.csv
2. Temperature data file: {filename}Tdata.csv

OUTPUT:
- 4-panel plot showing:
  * Top left: Light intensity vs time with detected peaks
  * Top right: Light intensity vs temperature with peaks
  * Bottom left: Temperature vs time (raw and smoothed)
  * Bottom right: Peak density analysis vs temperature
"""

import csv
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter, morlet

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS FOR YOUR DATA
# ============================================================================

# ---------------- BASIC DATA SETTINGS ----------------
# What data are you analyzing?
file_name = 'tempsweep092525'  # CHANGE THIS: Name of your data directory/file prefix
                                 # Script expects: {file_name}/waveform_files.csv OR {file_name}data.csv
                                 # And: {file_name}Tdata.csv (temperature data)

# What type of data format do you have?
files_are_txt = True            # True: Files are .txt and tab-separated rather than .csv
temp_input_is_raw_data = True    # True: Temperature data needs smoothing (typical for recent data)
                                # False: Temperature data is already processed
use_channel_b = True            # True: Analyze both Channel A and B from PicoScope
                               # False: Only analyze Channel A (light intensity)
invert_correction = True       # True: Flip the sign of Channel A data (if inverted)
plus_40_correction = True      # True: Add 40mV to Channel A data (baseline correction)

# ---------------- DISPLAY SETTINGS ----------------
# What portion of data would you like to analyze and display?
show_peak_lines = True          # True: Show vertical lines at detected peaks in plots
start_time = 3600               # Start time (seconds) for analysis window
end_time = 5800                # End time (seconds) for analysis window

# ---------------- ADVANCED PROCESSING PARAMETERS ----------------
# Fine-tune these if you understand the signal processing involved
interval = 10                   # Data compression factor (higher = more compression, faster processing)
                               # Recommended: 10 (takes every 10th point after averaging)

half_life = 30                  # Time constant (seconds) for exponential smoothing of temperature
                               # Smaller = less smoothing, larger = more smoothing
                               # Recommended: 30 (true value likely between 10-60)

peak_prominence = 0.5            # Minimum peak height (mV) to be counted as a real peak
                               # Increase if too many false peaks, decrease if missing real peaks

# ============================================================================
# DATA LOADING SECTION - HANDLES MULTIPLE DATA SOURCES
# ============================================================================

print("=== PICOSCOPE DATA ANALYSIS STARTING ===")
print(f"Analyzing data from: {file_name}")

# Initialize data storage arrays
fulltimedata = []               # Complete time series from raw files
fullchanneldata = []            # Complete Channel A data from raw files  
fullchannelbdata = []           # Complete Channel B data from raw files
smchanneldata = []              # Smoothed/compressed Channel A data
smchannelbdata = []             # Smoothed/compressed Channel B data
smtimedata = []                 # Time points for smoothed data
df_list = []                    # List for DataFrame operations

# Define expected file names
csv_name = file_name + 'data.csv'           # Processed data file
sm_csv_name = file_name + 'smdata.csv'      # Smoothed data file
temp_csv_name = file_name + 'Tdata.csv'     # Temperature data file

print(f"Looking for data files: {csv_name}, {sm_csv_name}, {temp_csv_name}")

# DATA LOADING PRIORITY:
# 1. Load pre-smoothed data if available (fastest)
# 2. Load pre-processed data if available  
# 3. Process raw PicoScope files (slowest)

if os.path.exists(sm_csv_name):
    # OPTION 1: Load already-smoothed data (fastest option)
    print("Found smoothed data file. Loading pre-processed data...")
    sm_data = pd.read_csv(sm_csv_name)
    
    # Remove header row if present and convert to lists
    smtimedata = sm_data['Time'].tolist()
    smtimedata.pop(0)  # Remove first element (likely a header)
    smchanneldata = sm_data['Channel A'].tolist()
    smchanneldata.pop(0)
    
    if use_channel_b:
        smchannelbdata = sm_data['Channel B'].tolist()
        smchannelbdata.pop(0)

elif os.path.exists(csv_name):
    # OPTION 2: Load processed data but need to smooth it
    print("Found processed data file. Loading and smoothing...")
    initial_data = pd.read_csv(csv_name)
    
    # Load full data arrays
    fulltimedata = initial_data['Time'].tolist()
    fulltimedata.pop(0)  # Remove header
    fullchanneldata = initial_data['Channel A'].tolist()
    fullchanneldata.pop(0)
    
    if use_channel_b:
        try:
            fullchannelbdata = initial_data['Channel B'].tolist()
            fullchannelbdata.pop(0)
        except KeyError:
            print('Warning: No Channel B data found in processed file.')
            use_channel_b = False

    # Re-save the data (cleaning step)
    finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata, fullchannelbdata)), 
                          columns=['Time', 'Channel A', 'Channel B'])
    finaldf.to_csv(csv_name, index=False)
    
    # Compress data by averaging every 'interval' points
    print(f"Condensing data by factor of {interval}...")
    num_points = int(len(fullchanneldata) / interval)
    for i in range(num_points):
        start_idx = interval * i
        end_idx = interval * i + (interval - 1)
        
        # Average over the interval
        smchanneldata.append(mean(fullchanneldata[start_idx:end_idx]))
        if use_channel_b:
            smchannelbdata.append(mean(fullchannelbdata[start_idx:end_idx]))
        smtimedata.append(fulltimedata[start_idx])
    
    # Save smoothed data for future use
    smdf = pd.DataFrame(list(zip(smtimedata, smchanneldata, smchannelbdata)), 
                       columns=['Time', 'Channel A', 'Channel B'])
    smdf.to_csv(sm_csv_name, index=False)

else:
    # OPTION 3: Process raw PicoScope CSV files (slowest but most complete)
    print("No processed data found. Loading raw PicoScope files...")
    directory_path = os.path.join(os.getcwd(), file_name)
    
    try:
        # Find all files in the data directory
        if files_are_txt:
            test_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.txt')]
        else:
            test_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
        
        if not test_files:
            raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")
        
        # Sort files by sequence number (handles different naming conventions)
        # This assumes PicoScope files have sequence numbers in positions 10-11 or 15-16
        if os.path.basename(test_files[0])[9:11] == "00":
            # Naming format: prefix_00_XX.csv (sequence in positions 15-16)
            all_files = sorted(test_files, key=lambda f: int(os.path.basename(f)[14:16]))
        else:
            # Naming format: prefix_XX.csv (sequence in positions 10-11) 
            all_files = sorted(test_files, key=lambda f: int(os.path.basename(f)[9:11]))
            
        print(f"Found {len(all_files)} CSV files: {all_files[:3]}{'...' if len(all_files) > 3 else ''}")
        
    except Exception as e:
        print(f"Error loading file list: {e}")
        raise

    # Process each raw PicoScope file
    print("Processing raw PicoScope files...")
    for file_idx, file in enumerate(all_files):
        if file_idx % 10 == 0:  # Progress indicator
            print(f"Processing file {file_idx + 1}/{len(all_files)}")
            
        file_path = os.path.join(directory_path, file)
        if files_are_txt:
            initial_data = pd.read_csv(file_path, sep="\t")
        else:
            initial_data = pd.read_csv(file_path)

        # Extract and clean time data
        strtimedata = initial_data['Time'].tolist()[2:]  # Skip header rows
        timedata = [float(s) for s in strtimedata]
        
        # Make time continuous across files
        if len(fulltimedata) > 0:
            time_offset = fulltimedata[-1]  # Last time point from previous file
            timedata = [t + time_offset for t in timedata]
        fulltimedata.extend(timedata)

        # Extract and clean Channel A data (main optical signal)
        strchanneldata = initial_data['Channel A'].tolist()[2:]  # Skip header rows
        channeldata = []
        for s in strchanneldata:
            try:
                channeldata.append(float(s))
            except ValueError:
                # Handle non-numeric values by using previous valid value
                channeldata.append(channeldata[-1] if channeldata else 0)
        fullchanneldata.extend(channeldata)

        # Extract and clean Channel B data if requested
        if use_channel_b:
            try:
                strchannelbdata = initial_data['Channel B'].tolist()[2:]
                channelbdata = []
                for s in strchannelbdata:
                    try:
                        channelbdata.append(float(s))
                    except ValueError:
                        channelbdata.append(channelbdata[-1] if channelbdata else 0)
                fullchannelbdata.extend(channelbdata)
            except KeyError:
                print("Warning: Channel B not found in raw files.")
                use_channel_b = False

    # Save processed raw data
    print("Saving processed data...")
    finaldf = pd.DataFrame(list(zip(fulltimedata, fullchanneldata, fullchannelbdata)), 
                          columns=['Time', 'Channel A', 'Channel B'])
    finaldf.to_csv(csv_name, index=False)
    
    # Compress data by averaging (reduces noise and computation time)
    print(f"Condensing data by factor of {interval}...")
    num_points = int(len(fullchanneldata) / interval)
    for i in range(num_points):
        start_idx = interval * i
        end_idx = interval * i + (interval - 1)
        
        smchanneldata.append(mean(fullchanneldata[start_idx:end_idx]))
        if use_channel_b:
            smchannelbdata.append(mean(fullchannelbdata[start_idx:end_idx]))
        smtimedata.append(fulltimedata[start_idx])
    
    # Save smoothed data for future use
    smdf = pd.DataFrame(list(zip(smtimedata, smchanneldata, smchannelbdata)), 
                       columns=['Time', 'Channel A', 'Channel B'])
    smdf.to_csv(sm_csv_name, index=False)

print(f"Data loading complete. {len(smchanneldata)} data points loaded.")

# ============================================================================
# DATA PREPROCESSING SECTION
# ============================================================================

# ---------------- TIME WINDOW SELECTION ----------------
# Trim data to the specified time window for analysis
print(f"Trimming data to time window: {start_time} - {end_time} seconds...")

# Find start index
if smtimedata[0] < start_time:
    for i, time_val in enumerate(smtimedata):
        if time_val >= start_time:
            smtimedata = smtimedata[i:]
            smchanneldata = smchanneldata[i:]
            if use_channel_b:
                smchannelbdata = smchannelbdata[i:]
            print(f"Trimmed {i} points from start of data")
            break

# Find end index  
if smtimedata[-1] > end_time:
    for i in range(len(smtimedata)):
        reverse_idx = len(smtimedata) - 1 - i
        if smtimedata[reverse_idx] <= end_time:
            trim_idx = reverse_idx + 1
            smtimedata = smtimedata[:trim_idx]
            smchanneldata = smchanneldata[:trim_idx]
            if use_channel_b:
                smchannelbdata = smchannelbdata[:trim_idx]
            print(f"Trimmed {i} points from end of data")
            break

# ---------------- DATA CORRECTIONS ----------------
# Apply corrections for known data issues
if invert_correction:
    print("Applying inversion correction to Channel A...")
    smchanneldata = [-x for x in smchanneldata]

if plus_40_correction:
    print("Applying +40mV baseline correction to Channel A...")
    smchanneldata = [x + 40 for x in smchanneldata]

# ---------------- SIGNAL SMOOTHING ----------------
# Apply Savitzky-Golay filter to reduce noise while preserving peaks
print("Applying smoothing filter to optical data...")
# Savitzky-Golay parameters: window=101 points, polynomial order=3
# This preserves peak shapes while reducing high-frequency noise
smoothed_data = savgol_filter(smchanneldata, window_length=101, polyorder=3)

if use_channel_b:
    smoothed_b_data = savgol_filter(smchannelbdata, window_length=101, polyorder=3)

print(f"Data preprocessing complete. {len(smoothed_data)} points ready for analysis.")

# ============================================================================
# TEMPERATURE DATA PROCESSING SECTION
# ============================================================================

print("Loading and processing temperature data...")

# Load temperature data file
try:
    raw_temp_data = pd.read_csv(temp_csv_name)
    
    # Handle different temperature column names
    temp_column_options = ['TSic AIN0 (°C)', 'TSic AIN2 (°C)', 'Temperature (°C)', 'Temp (°C)']
    temp_data = None
    
    for col in temp_column_options:
        if col in raw_temp_data.columns:
            temp_data = raw_temp_data[col].tolist()
            print(f"Using temperature column: {col}")
            break
    
    if temp_data is None:
        raise KeyError(f"No recognized temperature column found. Available columns: {raw_temp_data.columns.tolist()}")
    
    temp_time_data = raw_temp_data['Time (s)'].tolist()
    original_temp_data = temp_data.copy()  # Keep unprocessed version for plotting
    
except Exception as e:
    print(f"Error loading temperature data: {e}")
    raise

# ---------------- TEMPERATURE SMOOTHING ----------------
# Raw temperature data often has noise and needs smoothing
print("Smoothing temperature data...")

if temp_input_is_raw_data:
    print("Applying exponential smoothing to raw temperature data...")
    smoothing_range = 200  # Number of previous points to include in average
    
    smoothed_temp_data = []
    for i in range(len(temp_data)):
        # Exponentially weighted moving average
        # More recent points get higher weights
        sum_numerator = 0    # Weighted sum of temperatures
        sum_denominator = 0  # Sum of weights
        start_idx = max(0, i - smoothing_range)
        
        try:
            for j in range(start_idx, i + 1):
                time_diff = i - j  # How many points back
                weight = 0.5 ** (time_diff / half_life)  # Exponential decay
                sum_numerator += temp_data[j] * weight
                sum_denominator += weight
            
            if sum_denominator > 0:
                smoothed_temp_data.append(sum_numerator / sum_denominator)
            else:
                smoothed_temp_data.append(temp_data[i])  # Fallback
                
        except Exception as e:
            print(f"Error at index {i}: {e}")
            smoothed_temp_data.append(temp_data[i])
    
    temp_data = smoothed_temp_data
else:
    print("Temperature data assumed to be pre-processed.")

# ---------------- TEMPERATURE TIME WINDOW ----------------
# Trim temperature data to match the optical data time window
print("Trimming temperature data to analysis window...")

# Find start index for temperature
if temp_time_data[0] < start_time:
    for i, time_val in enumerate(temp_time_data):
        if time_val >= start_time:
            temp_time_data = temp_time_data[i:]
            temp_data = temp_data[i:]
            original_temp_data = original_temp_data[i:]
            break

# Find end index for temperature
if temp_time_data[-1] > end_time:
    for i in range(len(temp_time_data)):
        reverse_idx = len(temp_time_data) - 1 - i
        if temp_time_data[reverse_idx] <= end_time:
            trim_idx = reverse_idx + 1
            temp_time_data = temp_time_data[:trim_idx]
            temp_data = temp_data[:trim_idx]
            original_temp_data = original_temp_data[:trim_idx]
            break

# Apply additional smoothing with Savitzky-Golay filter
smooth_temp_data = savgol_filter(temp_data, window_length=120, polyorder=3)

# ---------------- TEMPERATURE INTERPOLATION ----------------
# Create temperature values that correspond to each optical data point
# This is necessary because temperature and optical data may have different sampling rates
print("Interpolating temperature data to match optical data timing...")

temp_index = 0
timed_temp_data = []  # Temperature at each optical data point

for optical_time in smtimedata:
    # Find the temperature data points that bracket this optical time point
    while (temp_index < len(temp_time_data) - 2 and 
           temp_time_data[temp_index + 1] < optical_time):
        temp_index += 1
    
    # Linear interpolation between bracketing temperature points
    if temp_index < len(temp_time_data) - 1:
        t_diff_below = optical_time - temp_time_data[temp_index]
        t_diff_above = temp_time_data[temp_index + 1] - optical_time
        total_diff = t_diff_above + t_diff_below
        
        if total_diff > 0:
            # Weighted average based on time distance
            interpolated_temp = ((smooth_temp_data[temp_index] * t_diff_above + 
                                smooth_temp_data[temp_index + 1] * t_diff_below) / total_diff)
        else:
            interpolated_temp = smooth_temp_data[temp_index]
            
        timed_temp_data.append(interpolated_temp)
    else:
        # Use last available temperature if out of bounds
        timed_temp_data.append(smooth_temp_data[-1])

print(f"Temperature processing complete. Range: {min(timed_temp_data):.1f}°C to {max(timed_temp_data):.1f}°C")

# ============================================================================
# PEAK DETECTION SECTION
# ============================================================================

print("Detecting peaks in optical signal...")

# Initialize peak storage
all_peaks = []  # Indices of peaks in the smoothed data

# Peak detection using scipy.signal.find_peaks
# Parameters:
# - prominence: minimum height difference between peak and surrounding valleys
# - distance: minimum separation between peaks (in data points)
all_peaks, peak_properties = find_peaks(
    smoothed_data, 
    prominence=peak_prominence,  # Minimum peak height (mV)
    distance=50                  # Minimum 50 data points between peaks
)

print(f"Found {len(all_peaks)} peaks with prominence >= {peak_prominence} mV")

# Alternative peak detection method (commented out but available):
"""
# Wavelet-based peak detection - more sophisticated but slower
# Useful for complex peak shapes or very noisy data
morlet_wavelet = lambda M, s: morlet(M, w=5, s=s, complete=False)
peaks = find_peaks_cwt(smoothed_data, np.arange(30,500), min_snr=0.1)
"""

# ---------------- PEAK TIMING AND TEMPERATURE ----------------
# Convert peak indices to actual time values
all_peaks_times = []
for peak_idx in all_peaks:
    all_peaks_times.append(smtimedata[peak_idx])

# Find the temperature at each peak time
all_peaks_temps = []
for peak_time in all_peaks_times:
    # Find closest temperature measurement time
    closest_temp_idx = min(range(len(temp_time_data)), 
                          key=lambda i: abs(temp_time_data[i] - peak_time))
    peak_temperature = smooth_temp_data[closest_temp_idx]
    all_peaks_temps.append(peak_temperature)

print(f"Peak analysis: Temperature range {min(all_peaks_temps):.1f}°C to {max(all_peaks_temps):.1f}°C")

# ---------------- PEAK RATE ANALYSIS ----------------
def sliding_window_peak_rate(peak_temperatures, temp_array, window_size=0.1):
    """
    Calculate the rate of peaks per degree Celsius using a sliding window.
    
    Parameters:
    - peak_temperatures: List of temperatures where peaks occurred
    - temp_array: Array of all temperature values  
    - window_size: Size of temperature window in degrees C
    
    Returns:
    - Array of peak rates (peaks per degree) for each temperature point
    """
    peak_temps = np.array(peak_temperatures)
    peak_rate = np.zeros_like(temp_array)
    
    for i, current_temp in enumerate(temp_array):
        # Define temperature window centered on current temperature
        window_start = current_temp - window_size / 2
        window_end = current_temp + window_size / 2
        
        # Count peaks within this temperature window
        peaks_in_window = np.sum((peak_temps >= window_start) & 
                                (peak_temps <= window_end))
        
        # Convert to rate (peaks per degree)
        peak_rate[i] = peaks_in_window / window_size
    
    return peak_rate

# Calculate peak rate with 0.03°C sliding window
peak_rate = sliding_window_peak_rate(all_peaks_temps, timed_temp_data, window_size=0.03)

print(f"Peak rate analysis complete. Max rate: {max(peak_rate):.1f} peaks/°C")

# ============================================================================
# PLOTTING SECTION - CREATE 4-PANEL ANALYSIS PLOT
# ============================================================================

print("Creating analysis plots...")

# Set up the figure with 2x2 subplot layout
fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(16, 10))
ax3 = ax5.twinx()  # Secondary y-axis for bottom-right plot

# ---------------- TOP LEFT: INTENSITY vs TIME ----------------
ax1.plot(smtimedata, smchanneldata, color="lightblue", 
         label='Raw data', alpha=0.6, linewidth=1)
ax1.plot(smtimedata, smoothed_data, color="orange", 
         label='Smoothed data', linewidth=2)

if use_channel_b:
    ax1.plot(smtimedata, smoothed_b_data, color="steelblue", 
             label='Channel B (scattering)', linewidth=2)

if show_peak_lines:
    ax1.vlines(all_peaks_times, ymin=min(smoothed_data), ymax=max(smoothed_data), 
               colors="grey", alpha=0.4, label=f'Peaks (n={len(all_peaks)})')

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Signal Value (mV)')
ax1.set_title(f'Optical Signal vs Time\n{len(all_peaks)} peaks detected')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---------------- TOP RIGHT: INTENSITY vs TEMPERATURE ----------------
ax2.plot(timed_temp_data, smchanneldata, color="lightblue", 
         label='Raw data', alpha=0.6, linewidth=1)
ax2.plot(timed_temp_data, smoothed_data, color="orange", 
         label='Smoothed data', linewidth=2)

if use_channel_b:
    ax2.plot(timed_temp_data, smoothed_b_data, color="steelblue",
             label='Channel B (scattering)', linewidth=2)

if show_peak_lines:
    ax2.vlines(all_peaks_temps, ymin=min(smoothed_data), ymax=max(smoothed_data),
               colors="gray", alpha=0.6, label='Peak locations')

ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Signal Value (mV)')
ax2.set_title('Optical Signal vs Temperature')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---------------- BOTTOM LEFT: TEMPERATURE vs TIME ----------------
ax4.plot(temp_time_data, original_temp_data, color="lightgreen", 
         label='Raw temperature', alpha=0.7, linewidth=1)
ax4.plot(temp_time_data, temp_data, color="mediumorchid", 
         label='Exponentially smoothed', linewidth=2)
ax4.plot(temp_time_data, smooth_temp_data, color="purple", 
         label='Savitzky-Golay smoothed', linewidth=2)

ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Temperature (°C)')
ax4.set_title('Temperature Data Processing')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ---------------- BOTTOM RIGHT: PEAK RATE ANALYSIS ----------------
if use_channel_b:
    # Calculate differential of Channel B (scattering analysis)
    b_rate_of_change = []
    look_ahead = 250  # Points to look ahead for derivative calculation
    
    for i in range(len(smoothed_b_data) - look_ahead):
        # Calculate rate of change over 'look_ahead' points
        rate = (smoothed_b_data[i + look_ahead] - smoothed_b_data[i]) * 100
        b_rate_of_change.append(rate)

    # Plot scattering rate of change on secondary axis
    ax3.plot(timed_temp_data[:len(b_rate_of_change)], b_rate_of_change, 
             color="steelblue", linewidth=2, alpha=0.7,
             label='Channel B rate of change')
    ax3.set_ylabel('Differential of Channel B', color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')

# Smooth the peak rate for cleaner visualization
if len(peak_rate) > 5000:
    smooth_peak_rate = savgol_filter(peak_rate, window_length=5001, polyorder=3)
    ax5.plot(timed_temp_data, smooth_peak_rate, color="red", linewidth=3, 
             alpha=0.8, label='Smoothed peak rate')

# Plot raw peak rate
ax5.plot(timed_temp_data, peak_rate, color="red", linewidth=1, 
         alpha=0.7, label='Peak rate (0.03°C window)')

ax5.set_xlabel('Temperature (°C)')
ax5.set_ylabel('Peaks per °C', color='red')
ax5.set_title('Peak Density Analysis')
ax5.tick_params(axis='y', labelcolor='red')
ax5.grid(True, alpha=0.3)

# Add figure-wide title with key information
fig.suptitle(f'Analysis of {file_name}\n'
             f'Time: {start_time}-{end_time}s, Temp: {min(timed_temp_data):.1f}-{max(timed_temp_data):.1f}°C, '
             f'{len(all_peaks)} peaks detected', 
             fontsize=14, y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Display the plot
print("=== ANALYSIS COMPLETE ===")
print(f"Results summary:")
print(f"  - Data points analyzed: {len(smoothed_data)}")
print(f"  - Time range: {smtimedata[0]:.1f} to {smtimedata[-1]:.1f} seconds")
print(f"  - Temperature range: {min(timed_temp_data):.1f} to {max(timed_temp_data):.1f} °C")
print(f"  - Peaks detected: {len(all_peaks)}")
print(f"  - Average peak rate: {len(all_peaks)/(max(timed_temp_data)-min(timed_temp_data)):.2f} peaks/°C")

print(all_peaks_temps)

plt.show()