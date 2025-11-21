from statistics import mean, median, stdev
import time
import sys
import csv  
import threading
import datetime
import os
import serial
import re
from labjack import ljm  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks
from collections import deque
from datetime import datetime
from rscomm import *
from dotenv import load_dotenv
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
import ctypes

load_dotenv()

"""
This function collects and displays data from our RTDs (temperature), FLUKE (pressure), Picoscope (fringes/raw & scattering/RMS), and octopus (humidity/temperature).

HARDWARE SETUP:
1. Connect the Labjack via USB to the computer.
2. Connect the FLUKE pressure sensor via USB to the computer.
3. Connect the Picoscope via USB to the computer, and connect BNC wires from each photodiode to the Picoscope.
4. Connect RTD wires to the Labjack and to each other:
    200µA — Red RTD-2
    AIN3 — Red RTD-2
        (RTD-2 resistor, ~100 ohms)
    AIN2 — White RTD-2
        (Red RTD-1 — White RTD-2, soldered & covered to avoid shorting)
    AIN1 — Red RTD-1
        (RTD-1 resistor, ~100 ohms)
    AIN0 — White RTD-1
    GND — White RTD-1
5. Connect photodiode bias modules to the Labjack.
    VS — VCC (red)
    GND — GND (black)
6. Connect the octopus sensor to the Labjack.
    FIO0 — Signal (yellow)
    VS — VCC (red)
    GND — GND (black)

VIRTUAL SETUP:
1. This function requires the old Intel architecture rather than the new Apple Silicon hardware. 
    To resolve this, we use a Conda environment to emulate the old hardware.
    TODO: Use "conda activate dashboard" on the lab computer to activate the Conda environment.
2. This function requires a .env file to have the port name for serial communication with the FLUKE pressure sensor, 
    as well as degree offsets for each RTD as determined by our calibration.
    TODO: Create a .env file in the same directory as this script with the following contents:
    FLUKE_PORT = "/dev/tty.usbserial-AV0L2AIU"
    RTD_1_OFFSET = 3.194
    RTD_2_OFFSET = 3.084
    or the appropriate port name for the computer and the offsets for the sensors.

CONFIGURATION:
The following configuration options are available to the user:
- Turning on/off each data stream.
- Adding RTDs and their Labjack AIN ports.
- Adding Picoscope channels, as well as which channels go to raw data vs RMS data.
- Adding an octopus sensor and its FIO port.

- Changing the sampling intervals for each data stream.
- Changing the averaging and RMS intervals for each data stream, 
    or removing the averaging entirely and displaying raw data.
- Changing the threshold for Picoscope fringe detection.

- Changing the display windows for each data stream.
- Changing the display rates for each data stream.

COMMON ERRORS:
- Some error related to DYLD_LIBRARY_PATH:
    The Picoscope SDK requires a specific library path to be set.
    Type this script into the terminal and try again:
    export DYLD_LIBRARY_PATH="/Applications/PicoScope 7 T&M.app/Contents/Resources:$DYLD_LIBRARY_PATH"
- Labjack not connecting:
    Check whether it is powered on (the green light is on); it most likely is not.
    There is likely a short with one of the Picoscope power wires, or another electrical issue.
    If the issue persists, set enable_temperature = False.
- Picoscope not connecting:
    Check whether Picoscope 7 is open. The device can't connect to multiple programs at once.
    Most issues with sensors not connecting/opening can be resolved by running the program again.
"""

# ================================================================================================
# CONFIGURE SETUP
# ================================================================================================

# ---------------- ENABLE FUNCTIONS - Turn data streams on/off ----------------
enable_temperature = True
enable_pressure = True
enable_picoscope = True
enable_octopus = True

# ---------------- SENSOR CONFIGURATION - Choose which sensors are set up ----------------
# Temperature configuration
ACTIVE_RTDS = { # Format: "Display Name": {"pos_channel": AIN_number, "neg_channel": AIN_number, "type": "PT100", "offset": offset_value}
    "RTD-1": {"pos_channel": 1, "neg_channel": 0, "type": "PT100", 
              "offset": float(os.getenv("RTD_1_OFFSET"))},
    "RTD-2": {"pos_channel": 3, "neg_channel": 2, "type": "PT100", 
              "offset": float(os.getenv("RTD_2_OFFSET"))}
} # If RTDs are not active, comment them out rather than deleting them

# Pressure configuration: The USB port that this is connected to should be defined as FLUKE_PORT in your .env file. 
# Default: FLUKE_PORT = "/dev/tty.usbserial-AV0L2AIU"

# Picoscope configuration
picoscope_channels = { # Available ranges: 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0 (volts). Coupling options: 'DC' or 'AC'
    'A': {'enabled': True, 'range': 10.0, 'coupling': 'DC'},
    'B': {'enabled': False, 'range': 10.0, 'coupling': 'DC'}
}
picoscope_raw_channels = ['A'] # Channels to log raw data from (i.e., measuring fringes)
picoscope_rms_channels = ['A'] # Channels to log RMS data from (i.e., measuring scattering)

# Octopus configuration
octopus_fio_channel = 0  # FIO channel for octopus sensor wire

# ================================================================================================
# CONFIGURE PROCESSING & DISPLAY
# ================================================================================================

# ---------------- TEMPERATURE ----------------
temperature_sample_interval = 0.2       # Interval (seconds) between samples
temperature_average_interval = 5      # Window (seconds) for which samples are averaged
temperature_display_window = 60       # Window (seconds) for which data is displayed
temperature_display_raw_data = False  # Display raw readings on graph (false = displays averaged data)
temperature_record_raw_data = True    # Record raw readings in CSV (false = records averaged data)

# ---------------- PRESSURE ----------------
pressure_sample_interval = 0.2        # Interval (seconds) between samples
pressure_average_interval = 30        # Window (seconds) for which samples are averaged
pressure_display_window = 60          # Window (seconds) for which data is displayed
pressure_display_raw_data = False     # Display raw readings on graph (false = displays averaged data)
pressure_record_raw_data = True       # Record raw readings in CSV (false = records averaged data)

# ---------------- PICOSCOPE ----------------
picoscope_data_update_interval = 0.2  # Interval (seconds) between data updates
picoscope_sample_interval_us = 50               # Microseconds between samples (don't go above 50, Picoscope can't go that slow)
picoscope_display_window = 60         # Window (seconds) for Picoscope data displayed 
picoscope_bin_size = 1000               # Number of samples to average for raw data. picoscope_bin_size*picoscope_sample_interval_us = bin sizes in us. 50 & 1000 gives 20Hz
picoscope_rms_window = 30             # Window (seconds) for RMS calculation
fringe_voltage_threshold = 0.05       # Minimum voltage (V) for fringe detection

# ---------------- BACKEND ----------------
csv_write_interval = 1                # Interval (seconds) between CSV writes





# ================================================================================================
# PROGRAM BEGINS
# ================================================================================================

# ================================================================================================
# SYSTEM INITIALIZATION - Automatic configuration based on user settings
# ================================================================================================

# Load USB port names from .env
fluke_port = os.getenv("FLUKE_PORT")

# Generate unique filename with timestamp for data logging
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
csv_filename = f"sensor_readings_{timestamp}.csv"
print(f"Data will be logged to: {csv_filename}")

# ================================================================================================
# DATA STRUCTURES - Storage for sensor data and system state
# ================================================================================================

temperature_time_data = []  # Time points for x-axis
temperature_data = {rtd: [] for rtd in ACTIVE_RTDS}
temperature_average_data = {rtd: [] for rtd in ACTIVE_RTDS}
temperature_std_data = {rtd: [] for rtd in ACTIVE_RTDS}
pressure_time_data = []
pressure_data = []
pressure_average_data = []
pressure_std_data = []
start_time = time.time()  # Program start timestamp for relative timing

picoscope_time_data = []
picoscope_time_rolling_data = []
picoscope_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_rolling_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_rms_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}
picoscope_rms_rolling_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}
picoscope_time_rolling_data = []

picoscope_binned_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_binned_times = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_bin_buffer = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_bin_start_time = {ch: None for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
picoscope_sample_period = picoscope_sample_interval_us / 1_000_000  # Convert microseconds to seconds

class SharedData:
    """
    Thread-safe data container for sharing sensor readings between threads.
    
    This class uses threading locks to prevent data corruption when multiple
    threads access temperature and electrical measurements simultaneously.
    
    Attributes:
        current_temp: Most recent temperature reading (°C)
        avg_thermo: Dictionary of averaged thermocouple readings
        current_voltage: Most recent voltage measurement (V)
        current_current: Most recent current measurement (A)
        current_pressure: Most recent pressure measurement (Bar)
        picoscope_voltages: Dictionary of current PicoScope readings by channel
    """
    
    def __init__(self):
        self.lock = threading.Lock()

        # Temperature data
        self.temp_time_data = []
        self.temp_data = {rtd: [] for rtd in ACTIVE_RTDS}
        self.temp_average_data = {rtd: [] for rtd in ACTIVE_RTDS}
        self.temp_std_data = {rtd: [] for rtd in ACTIVE_RTDS}

        self.temp_time_rolling_data = []
        self.temp_rolling_data = {rtd: [] for rtd in ACTIVE_RTDS}
        self.temp_average_rolling_data = {rtd: [] for rtd in ACTIVE_RTDS}
        self.temp_std_rolling_data = {rtd: [] for rtd in ACTIVE_RTDS}

        # Pressure data
        self.pressure_time_data = []
        self.pressure_data = []
        self.pressure_average_data = []
        self.pressure_std_data = []

        self.pressure_time_rolling_data = []
        self.pressure_rolling_data = []
        self.pressure_average_rolling_data = []
        self.pressure_std_rolling_data = []

        # PicoScope data
        self.picoscope_raw_time_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_raw_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_rms_time_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_rms_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_peak_times = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}

        self.picoscope_raw_time_rolling_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_raw_rolling_data = {ch: [] for ch in picoscope_raw_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_rms_time_rolling_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}
        self.picoscope_rms_rolling_data = {ch: [] for ch in picoscope_rms_channels if picoscope_channels[ch]['enabled']}

        self.octopus_temperature = 0
        self.octopus_humidity = 0

    def update_temperature(self, new_time_data, new_temp_data, new_average_data, new_std_data):
        with self.lock:
            self.temp_time_data.append(new_time_data)
            self.temp_time_rolling_data.append(new_time_data)
            for rtd in ACTIVE_RTDS:
                self.temp_data[rtd].append(new_temp_data[rtd])
                self.temp_average_data[rtd].append(new_average_data[rtd])
                self.temp_std_data[rtd].append(new_std_data[rtd])
                
                self.temp_rolling_data[rtd].append(new_temp_data[rtd])
                self.temp_average_rolling_data[rtd].append(new_average_data[rtd])
                self.temp_std_rolling_data[rtd].append(new_std_data[rtd])
            
            # Trim rolling data to display window
            max_samples = int(temperature_display_window / temperature_sample_interval)
            if len(self.temp_time_rolling_data) > max_samples:
                self.temp_time_rolling_data = self.temp_time_rolling_data[-max_samples:]
                for rtd in ACTIVE_RTDS:
                    self.temp_rolling_data[rtd] = self.temp_rolling_data[rtd][-max_samples:]
                    self.temp_average_rolling_data[rtd] = self.temp_average_rolling_data[rtd][-max_samples:]
                    self.temp_std_rolling_data[rtd] = self.temp_std_rolling_data[rtd][-max_samples:]

    def update_pressure(self, new_time_data, new_pressure_data, new_average_data, new_std_data):
        with self.lock:
            self.pressure_time_data.append(new_time_data)
            self.pressure_data.append(new_pressure_data)
            self.pressure_average_data.append(new_average_data)
            self.pressure_std_data.append(new_std_data)
            
            self.pressure_time_rolling_data.append(new_time_data)
            self.pressure_rolling_data.append(new_pressure_data)
            self.pressure_average_rolling_data.append(new_average_data)
            self.pressure_std_rolling_data.append(new_std_data)
            
            # Trim rolling data to display window
            max_samples = int(pressure_display_window / pressure_sample_interval)
            if len(self.pressure_time_rolling_data) > max_samples:
                self.pressure_time_rolling_data = self.pressure_time_rolling_data[-max_samples:]
                self.pressure_rolling_data = self.pressure_rolling_data[-max_samples:]
                self.pressure_average_rolling_data = self.pressure_average_rolling_data[-max_samples:]
                self.pressure_std_rolling_data = self.pressure_std_rolling_data[-max_samples:]
    
    def update_picoscope_raw(self, channel, new_time_data, new_raw_data):
        with self.lock:
            # FIXED: Append to channel-specific lists
            self.picoscope_raw_time_data[channel].append(new_time_data)
            self.picoscope_raw_data[channel].append(new_raw_data)
            
            self.picoscope_raw_time_rolling_data[channel].append(new_time_data)
            self.picoscope_raw_rolling_data[channel].append(new_raw_data)
            
            # Trim rolling data
            max_samples = int(picoscope_display_window * 1000000 / (picoscope_bin_size * picoscope_sample_interval_us))
            if len(self.picoscope_raw_time_rolling_data[channel]) > max_samples:
                self.picoscope_raw_time_rolling_data[channel] = self.picoscope_raw_time_rolling_data[channel][-max_samples:]
                self.picoscope_raw_rolling_data[channel] = self.picoscope_raw_rolling_data[channel][-max_samples:]
        
            # Detect peaks in the raw data
            peaks, _ = find_peaks(self.picoscope_raw_data[channel][max(0,len(self.picoscope_raw_data[channel])-3*max_samples):], height=fringe_voltage_threshold)
            peak_times = [self.picoscope_raw_time_data[channel][i] for i in peaks]
            self.picoscope_peak_times[channel] = peak_times

    
    def update_picoscope_rms(self, channel, new_time_data, new_rms_data):
        with self.lock:
            # FIXED: Append to channel-specific lists
            self.picoscope_rms_time_data[channel].append(new_time_data)
            self.picoscope_rms_data[channel].append(new_rms_data)
            
            self.picoscope_rms_time_rolling_data[channel].append(new_time_data)
            self.picoscope_rms_rolling_data[channel].append(new_rms_data)
            
            # Trim rolling data
            max_samples = int(picoscope_rms_window / picoscope_data_update_interval)
            if len(self.picoscope_rms_time_rolling_data[channel]) > max_samples:
                self.picoscope_rms_time_rolling_data[channel] = self.picoscope_rms_time_rolling_data[channel][-max_samples:]
                self.picoscope_rms_rolling_data[channel] = self.picoscope_rms_rolling_data[channel][-max_samples:]

    def update_octopus(self, temperature, humidity):
        with self.lock:
            self.octopus_temperature = temperature
            self.octopus_humidity = humidity

    
    def get_temperature_rolling_data(self):
        with self.lock:
            return self.temp_time_rolling_data, self.temp_rolling_data, self.temp_average_rolling_data, self.temp_std_rolling_data
        
    def get_pressure_rolling_data(self):
        with self.lock:
            return self.pressure_time_rolling_data, self.pressure_rolling_data, self.pressure_average_rolling_data, self.pressure_std_rolling_data
        
    def get_picoscope_raw_rolling_data(self):
        with self.lock:
            return self.picoscope_raw_time_rolling_data, self.picoscope_raw_rolling_data, self.picoscope_peak_times
    
    def get_picoscope_rms_rolling_data(self):
        with self.lock:
            return self.picoscope_rms_time_rolling_data, self.picoscope_rms_rolling_data
        
    def get_octopus_data(self):
        with self.lock:
            return self.octopus_temperature, self.octopus_humidity

# Create global shared data instance
shared_data = SharedData()

# Thread synchronization objects
data_lock = threading.Lock()  # Protects plotting data
exit_event = threading.Event()  # Signals all threads to stop

# ================================================================================================
# CSV FILE INITIALIZATION - Create data logging file with appropriate headers
# ================================================================================================

def create_csv_file():
    """Create CSV file with headers for all enabled sensors."""
    headers = ["Timestamp", "Time(s)"]
    
    if enable_temperature:
        for rtd in ACTIVE_RTDS:
            headers.extend([f"{rtd}_Temp(C)"])

    if enable_pressure:
        headers.extend(["Pressure(Bar)"])

    if enable_picoscope:
        for ch in picoscope_raw_channels:
            if picoscope_channels[ch]['enabled']:
                headers.append(f"Pico_Ch{ch}_Raw(V)")

    if enable_picoscope:
        for ch in picoscope_rms_channels:
            if picoscope_channels[ch]['enabled']:
                headers.append(f"Pico_Ch{ch}_RMS(V)")
    
    if enable_octopus:
        headers.extend(["Octopus_Temp(C)", "Octopus_Humidity(%)"])
    
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Initialize CSV file
create_csv_file()

def write_csv_row():
    """
    Write a single row to CSV with latest values from all sensors.
    Uses shared_data to get most recent measurements.
    """
    try:
        current_time = time.time()
        relative_time = current_time - start_time
        
        row = [f"{current_time:.3f}", f"{relative_time:.3f}"]
        
        with data_lock:
            # Temperature data
            if enable_temperature:
                for rtd in ACTIVE_RTDS:
                    if len(shared_data.temp_average_data[rtd]) > 0:
                        row.append(f"{shared_data.temp_average_data[rtd][-1]:.4f}")
                    else:
                        row.extend([''])
            
            # Pressure data
            if enable_pressure:
                if len(shared_data.pressure_average_data) > 0:
                    row.append(f"{shared_data.pressure_average_data[-1]:.6f}")
                else:
                    row.extend([''])
            
            # PicoScope raw data
            if enable_picoscope:
                for ch in picoscope_raw_channels:
                    if picoscope_channels[ch]['enabled']:
                        if ch in shared_data.picoscope_raw_data and len(shared_data.picoscope_raw_data[ch]) > 0:
                            row.append(f"{shared_data.picoscope_raw_data[ch][-1]:.6f}")
                        else:
                            row.append('')
            
            # PicoScope RMS data
            if enable_picoscope:
                for ch in picoscope_rms_channels:
                    if picoscope_channels[ch]['enabled']:
                        if ch in shared_data.picoscope_rms_data and len(shared_data.picoscope_rms_data[ch]) > 0:
                            row.append(f"{shared_data.picoscope_rms_data[ch][-1]:.6f}")
                        else:
                            row.append('')
            
            # Octopus data (would need to add to SharedData class)
            if enable_octopus:
                # Add octopus_temp and octopus_humidity to SharedData first
                # For now, placeholder:
                row.extend(['', ''])
        
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
            
    except Exception as e:
        print(f"CSV write error: {e}")

def csv_writer_thread():
    """Dedicated thread for writing data to CSV at regular intervals."""
    print(f"CSV writer thread started (interval: {csv_write_interval}s)")
    
    while not exit_event.is_set():
        time.sleep(csv_write_interval)
        write_csv_row()

# ================================================================================================
# SENSOR CLASSES
# ================================================================================================

class PicoScopeReader:
    """
    PicoScope 2205A interface for voltage measurements in streaming mode.
    """

    def __init__(self):
        self.chandle = ctypes.c_int16()
        self.status = {}
        self.maxADC = ctypes.c_int16(32767)
        self.enabled_channels = []
        self.channel_ranges = {}
        self.streaming = False
        self.latest_data = {ch: [] for ch in ['A', 'B']}  # Use lists, not numpy arrays
        self.callback_ptr = None
        self._streaming_callback = None

    def open(self, picoscope_channels):
        """Open the PicoScope and configure channels."""
        try:
            self.status["openunit"] = ps.ps2000_open_unit()
            self.chandle = ctypes.c_int16(self.status["openunit"])
            if self.chandle.value <= 0:
                raise Exception(f"Failed to open PicoScope (code {self.chandle.value})")

            print("PicoScope connected")

            channel_map = {'A': 0, 'B': 1}
            coupling_map = {'DC': 1, 'AC': 0}
            range_map = {
                0.02: 1, 0.05: 2, 0.1: 3, 0.2: 4, 0.5: 5,
                1.0: 6, 2.0: 7, 5.0: 8, 10.0: 9, 20.0: 10
            }

            for ch_name, config in picoscope_channels.items():
                ch_num = channel_map[ch_name]
                coupling = coupling_map[config['coupling']]
                vrange = range_map[config['range']]

                if config['enabled']:
                    self.status[f"setCh{ch_name}"] = ps.ps2000_set_channel(
                        self.chandle, ch_num, 1, coupling, vrange
                    )
                    self.enabled_channels.append(ch_name)
                    self.channel_ranges[ch_name] = vrange  # Store the range index
                    self.latest_data[ch_name] = []  # Initialize list for this channel
                else:
                    ps.ps2000_set_channel(self.chandle, ch_num, 0, 1, 6)

            print(f"Configured channels: {self.enabled_channels}")
            return True

        except Exception as e:
            print(f"Error initializing PicoScope: {e}")
            return False

    # ---------------- STREAMING CALLBACK ----------------

    def _get_overview_buffers(self, buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values):
        """Callback function called by PicoScope streaming to copy data."""
        try:
            if n_values > 0:
                # buffers is a pointer to array of pointers
                # buffers[0] points to channel A data, buffers[1] to channel B, etc.
                # We need to access based on which channels are enabled
                
                # According to docs, we should use buffers[0] for the first enabled channel
                # For PS2000, channel order is A=0, B=1, C=2, D=3
                channel_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                
                for idx, ch in enumerate(self.enabled_channels):
                    data = buffers[idx][0:n_values]
                    self.latest_data[ch].extend(data)
                        
        except Exception as e:
            print(f"Callback error: {e}")
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------

    def start_streaming(self, picoscope_sample_interval_us=1000):
        """Start PicoScope streaming using ps2000_run_streaming_ns."""
        try:
            if self.streaming:
                print("Streaming already active.")
                return True

            # Clear any old data
            for ch in self.enabled_channels:
                self.latest_data[ch] = []

            # Prepare callback prototype
            STREAMING_CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
                None,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
                ctypes.c_int16,
                ctypes.c_uint32,
                ctypes.c_int16,
                ctypes.c_int16,
                ctypes.c_uint32
            )

            self.callback_ptr = STREAMING_CALLBACK(self._get_overview_buffers)

            # Start streaming using run_streaming_ns (nanosecond version)
            # Parameters: handle, sample_interval, time_unit, max_samples, auto_stop, downsample_ratio, overview_buffer_size
            sample_interval_ns = picoscope_sample_interval_us * 1000
            
            self.status["runStreaming"] = ps.ps2000_run_streaming_ns(
                self.chandle,
                sample_interval_ns,  # sampling interval in ns
                2,  # time unit: 2 = nanoseconds
                100_000,  # max samples per call
                False,  # auto_stop
                1,  # downsample ratio (1 = no downsampling)
                50_000
            )
            
            print(sample_interval_ns)
            
            if self.status["runStreaming"] == 0:
                raise Exception("ps2000_run_streaming_ns failed.")

            self.streaming = True
            print(f"Streaming started at {1000000/picoscope_sample_interval_us:.1f} Hz")
            return True

        except Exception as e:
            print(f"Error starting streaming: {e}")
            import traceback
            traceback.print_exc()
            return False

    def read_voltages(self):
        """
        Poll the PicoScope for the latest streaming data using callback.
        """
        if not self.streaming:
            print("Streaming not active.")
            return {ch: None for ch in self.enabled_channels}

        try:
            # Clear the latest data before collecting new batch
            for ch in self.enabled_channels:
                self.latest_data[ch] = []

            # Poll for data - this calls the callback which fills latest_data
            ps.ps2000_get_streaming_last_values(self.chandle, self.callback_ptr)
            
            # Convert accumulated ADC values to voltages
            voltages = {}
            for ch in self.enabled_channels:
                adc_list = self.latest_data[ch]
                
                if len(adc_list) > 0:
                    adc_values = np.array(adc_list)
                    
                    # Convert ADC to volts
                    vrange = self.channel_ranges[ch]
                    mv = adc2mV(adc_values, vrange, self.maxADC)
                    voltages[ch] = [value / 1000.0 for value in mv]
                else:
                    voltages[ch] = np.array([])

            return voltages

        except Exception as e:
            print(f"Streaming read error: {e}")
            import traceback
            traceback.print_exc()
            return {ch: None for ch in self.enabled_channels}

    def stop_streaming(self):
        try:
            if self.streaming:
                ps.ps2000_stop(self.chandle)
                self.streaming = False
                print("Streaming stopped.")
        except Exception as e:
            print(f"Error stopping streaming: {e}")

    def close(self):
        try:
            self.stop_streaming()
            ps.ps2000_close_unit(self.chandle)
            print("PicoScope closed.")
        except Exception as e:
            print(f"Close error: {e}")

# Global PicoScope instance
picoscope = None

DHT11_LUA_SCRIPT = """
-- DHT11 Sensor Reader for LabJack T7
-- Reads temperature and humidity from DHT11 sensor
-- Results stored in USER_RAM0_F32 (humidity) and USER_RAM1_F32 (temperature)
-- USER_RAM2_F32 stores success flag (1=success, 0=fail)

local FIO_PIN = 0
local mbRead = MB.R
local mbWrite = MB.W
local checkInterval = 2000

-- Helper function to set pin direction (0=input, 1=output)
local function setPinDirection(dir)
  mbWrite(6000 + FIO_PIN, 0, dir)
end

-- Helper function to write digital state
local function writePin(state)
  mbWrite(2000 + FIO_PIN, 0, state)
end

-- Cache the pin read address for speed
local pinReadAddr = 2000 + FIO_PIN

-- Initialize pin as output, high
setPinDirection(1)
writePin(1)

-- Main reading function using iteration counting
local function readDHT11()
  -- Send start signal
  setPinDirection(1)  -- Output
  writePin(0)         -- Pull LOW
  LJ.IntervalConfig(1, 18)
  LJ.CheckInterval(1) -- Wait 18ms
  
  writePin(1)         -- Release (pull high)
  
  -- Switch to input
  setPinDirection(0)
  
  -- Small delay
  for i=1,100 do end
  
  -- Wait for DHT11 response LOW pulse
  local timeout = 0
  while mbRead(pinReadAddr, 0) == 1 do
    timeout = timeout + 1
    if timeout > 10000 then
      return false
    end
  end
  
  -- Wait for HIGH pulse
  timeout = 0
  while mbRead(pinReadAddr, 0) == 0 do
    timeout = timeout + 1
    if timeout > 10000 then
      return false
    end
  end
  
  -- Wait for next LOW (start of data)
  timeout = 0
  while mbRead(pinReadAddr, 0) == 1 do
    timeout = timeout + 1
    if timeout > 10000 then
      return false
    end
  end
  
  -- Read 40 bits by counting loop iterations during HIGH pulses
  local bitCounts = {}
  for i = 1, 40 do
    -- Wait for HIGH
    timeout = 0
    while mbRead(pinReadAddr, 0) == 0 do
      timeout = timeout + 1
      if timeout > 10000 then
        return false
      end
    end
    
    -- Count iterations while HIGH
    local count = 0
    while mbRead(pinReadAddr, 0) == 1 do
      count = count + 1
      if count > 10000 then
        return false
      end
    end
    
    bitCounts[i] = count
  end
  
  -- Find threshold: average of all counts, or use fixed value
  local sumCounts = 0
  for i = 1, 40 do
    sumCounts = sumCounts + bitCounts[i]
  end
  local threshold = sumCounts / 40
  
  -- Convert counts to bits (longer count = 1, shorter = 0)
  local data = {}
  for i = 1, 40 do
    if bitCounts[i] > threshold then
      data[i] = 1
    else
      data[i] = 0
    end
  end
  
  -- Convert bits to bytes
  local bytes = {}
  for i = 1, 5 do
    local byte = 0
    for j = 1, 8 do
      byte = byte * 2 + data[(i-1)*8 + j]
    end
    bytes[i] = byte
  end
  
  -- Verify checksum
  local checksum = (bytes[1] + bytes[2] + bytes[3] + bytes[4]) % 256
  if checksum ~= bytes[5] then
    return false
  end
  
  -- Extract values
  local humidity = bytes[1] + bytes[2] * 0.1
  local temperature = bytes[3] + bytes[4] * 0.1
  
  -- Write to USER_RAM registers
  mbWrite(46000, 3, humidity)
  mbWrite(46002, 3, temperature)
  mbWrite(46004, 3, 1)
  
  return true
end

-- Main loop
LJ.IntervalConfig(0, checkInterval)

while true do
  if LJ.CheckInterval(0) then
    local success = readDHT11()
    if not success then
      mbWrite(46004, 3, 0)
    end
    
    -- Set pin back to output high
    setPinDirection(1)
    writePin(1)
  end
end
"""

class OctopusReader:
    """
    DHT11 temperature and humidity sensor reader using LabJack T7 Lua scripting.
    
    This class manages the DHT11 sensor by:
    1. Uploading a Lua script to the T7 that handles the timing-critical protocol
    2. Reading the results from USER_RAM registers that the Lua script updates
    
    The Lua script runs continuously on the T7's processor, reading the DHT11
    every 2 seconds and storing results in USER_RAM registers.
    """
    
    def __init__(self, handle, fio_channel=0):
        """
        Initialize DHT11 reader with LabJack connection.
        
        Args:
            handle: LabJack device handle from ljm.openS()
            fio_channel: FIO pin number for DHT11 data line (default: 0)
        """
        self.handle = handle
        self.fio_pin = fio_channel
        self.lua_loaded = False
        
        # Modify the Lua script to use the correct FIO pin
        self.lua_script = DHT11_LUA_SCRIPT.replace("local FIO_PIN = 0", 
                                                     f"local FIO_PIN = {fio_channel}")
        
        # Load the Lua script onto the T7
        self._load_lua_script()
    
    def _load_lua_script(self):
        """
        Load and start the Lua script on the LabJack T7.
        
        The script will run continuously in the background, updating
        USER_RAM registers with temperature and humidity readings.
        """
        try:
            print(f"Loading DHT11 Lua script onto LabJack (FIO{self.fio_pin})...")
            
            # First, stop any running Lua script
            ljm.eWriteName(self.handle, "LUA_RUN", 0)
            time.sleep(0.5)
            
            # Clear any existing script
            ljm.eWriteName(self.handle, "LUA_SOURCE_SIZE", 0)
            time.sleep(0.5)
            
            # Write the Lua script in chunks (LJM may have size limits)
            script_bytes = self.lua_script.encode('utf-8')
            
            ljm.eWriteName(self.handle, "LUA_SOURCE_SIZE", len(script_bytes)+1)
            ljm.eWriteNameByteArray(self.handle, "LUA_SOURCE_WRITE", len(script_bytes)+1, script_bytes)
            
            # Check for compilation errors
            size = ljm.eReadName(self.handle, "LUA_SOURCE_SIZE")
            if size == 0:
                raise Exception("Lua script failed to load (size = 0)")
            
            ljm.eWriteName(self.handle, "LUA_DEBUG_ENABLE", 1)
            ljm.eWriteName(self.handle, "LUA_DEBUG_ENABLE_DEFAULT", 1)
            ljm.eWriteName(self.handle, "LUA_RUN", 1)
            
            print(f"✓ DHT11 Lua script loaded and running (script size: {size} bytes)")
            self.lua_loaded = True
            
            # Give the script time to perform first reading
            time.sleep(2.5)
            
        except Exception as e:
            print(f"✗ Error loading Lua script: {e}")
            import traceback
            traceback.print_exc()
            self.lua_loaded = False
    
    def read_sensor(self):
        """
        Read temperature and humidity from DHT11 via USER_RAM registers.
        
        The Lua script continuously updates these registers every 2 seconds.
        This method simply reads the most recent values.
        
        Returns:
            tuple: (humidity, temperature, success)
                   humidity: Relative humidity (%)
                   temperature: Temperature (°C)
                   success: True if reading is valid, False otherwise
        """
        if not self.lua_loaded:
            print("Lua script not loaded, cannot read sensor")
            return None, None, False
        
        try:
            # Read values from USER_RAM registers
            humidity = ljm.eReadName(self.handle, "USER_RAM0_F32")
            temperature = ljm.eReadName(self.handle, "USER_RAM1_F32")
            success_flag = ljm.eReadName(self.handle, "USER_RAM2_F32")
            
            # Check if reading was successful (Lua script sets flag to 1)
            if success_flag == 1:
                return humidity, temperature, True
            else:
                return None, None, False
                
        except Exception as e:
            print(f"Error reading DHT11 data: {e}")
            return None, None, False
    
    def stop(self):
        """Stop the Lua script running on the LabJack."""
        try:
            if self.lua_loaded:
                ljm.eWriteName(self.handle, "LUA_RUN", 0)
                print("DHT11 Lua script stopped")
                self.lua_loaded = False
        except Exception as e:
            print(f"Error stopping Lua script: {e}")

# Global octopus instance
octopus = None

# ================================================================================================
# HARDWARE CONFIGURATION FUNCTIONS
# ================================================================================================

def configure_rtd(handle):
    """
    Configure LabJack channels for RTD (Resistance Temperature Detector) measurements.
    
    Configures analog input channels with extended features for RTD sensors.
    Supports PT100, PT500, and PT1000 RTD types.
    
    Reference: https://labjack.com/support/datasheets/t7/digital-io/extended-features/rtd
    
    Args:
        handle: LabJack device handle from ljm.openS()
    """
    print("Configuring RTD sensors...")
    for name, config in ACTIVE_RTDS.items():
        pos_channel = config["pos_channel"]
        neg_channel = config["neg_channel"]
        rtd_type = config["type"]
        
        # Map RTD type to resistance value for EF_CONFIG_A
        rtd_resistance_map = {
            "PT100": 100,
            "PT500": 500,
            "PT1000": 1000
        }
        
        # Set input range & resolution
        ljm.eWriteName(handle, f"AIN{pos_channel}_RANGE", 0.1)
        ljm.eWriteName(handle, f"AIN{pos_channel}_RESOLUTION_INDEX", 12) # 12 = 24-bit, most precise
        ljm.eWriteName(handle, f"AIN{neg_channel}_RANGE", 0.1)
        ljm.eWriteName(handle, f"AIN{neg_channel}_RESOLUTION_INDEX", 12) # 12 = 24-bit, most precise

        print(f"  Configured {name} RTD ({rtd_type}) on AIN{pos_channel} and AIN{neg_channel}")

# ================================================================================================
# SENSOR READING FUNCTIONS
# ================================================================================================

def read_temp_sensors():
    """
    Read temperature from all configured sensors.
    
    Connects to LabJack, reads all active sensor channels, and returns
    organized temperature data. Handles connection errors gracefully.
    
    Returns:
        tuple: (rtd_temps)
               Each is a dictionary mapping sensor names/channels to temperatures
               Returns (None, None, None, None) on error
    """
    try:        
        # Read RTD temperatures using extended features
        rtd_temps = {}
        for name, config in ACTIVE_RTDS.items():
            pos_channel = config['pos_channel']
            neg_channel = config['neg_channel']
            # Read processed temperature from extended feature
            pos_voltage = ljm.eReadName(handle, f"AIN{pos_channel}")
            neg_voltage = ljm.eReadName(handle, f"AIN{neg_channel}")
            resistance = (pos_voltage - neg_voltage)/0.0002 # R = V/I, I = 200 uA
            temp = 20 + (resistance-107.794)/0.385 + config['offset']
            rtd_temps[name] = temp
        
        return rtd_temps
        
    except Exception as e:
        print(f"Sensor read error: {e}")
        return None, None, None, None
    
def send_fluke_command(ser, command):
    """Send SCPI command to Fluke 2700G"""
    ser.write((command + "\r").encode())  
    ser.flush()
    time.sleep(0.1)  # Shortened delay for speed

def read_fluke_response(ser):
    """Read response from Fluke 2700G"""
    response = ser.readlines()
    if not response:
        return None

    last_response = response[-1].decode(errors='replace').strip()
    return last_response

def parse_fluke_value(response, unit_to_remove):
    """Parses numeric value, removes unit, and extracts only the last valid float."""
    try:
        if not response:
            return None

        response = response.replace(unit_to_remove, "").strip()
        parts = response.split("\r")

        for part in reversed(parts):  
            if re.search(r"\d", part):  
                numeric_part = part.split(",")[0].strip()  
                value = float(numeric_part)
                return float(f"{value:.10g}")  

        print(f"⚠️ Could not parse response: {repr(response)}")
        return None

    except ValueError:
        print(f"⚠️ ValueError: Could not convert {repr(response)}")
        return None
    
def read_pressure_sensor():
    try:
        with serial.Serial(fluke_port, 9600, timeout=0.01, 
                          parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
            send_fluke_command(ser, "*CLS")  
            send_fluke_command(ser, "*IDN?")  
            time.sleep(0.1)

            # Query Pressure
            send_fluke_command(ser, "VAL?")
            pressure_response = read_fluke_response(ser)

            if pressure_response is None:
                print("⚠️ No response from pressure sensor")
                return None
            
            pressure = parse_fluke_value(pressure_response, "BAR")
            return pressure

    except serial.SerialException as e:
        # print(f"⚠️ Serial error: {e}")
        return None
    
# ================================================================================================
# DATA COLLECTION THREADS
# ================================================================================================

def collect_picoscope_data():
    print("PicoScope collection thread started")
    picoscope.start_streaming(picoscope_sample_interval_us=picoscope_sample_interval_us)
    
    while not exit_event.is_set():
        time.sleep(picoscope_data_update_interval)
        stream_data = picoscope.read_voltages()
        
        for ch, values in stream_data.items():
            if values is not None and len(values) > 0:
                current_time = time.time()  # Absolute timestamp
                
                if picoscope_bin_start_time[ch] is None:
                    picoscope_bin_start_time[ch] = current_time
                
                picoscope_bin_buffer[ch].extend(values)
                
                # Process complete bins
                while len(picoscope_bin_buffer[ch]) >= picoscope_bin_size:
                    bin_samples = picoscope_bin_buffer[ch][:picoscope_bin_size]
                    picoscope_bin_buffer[ch] = picoscope_bin_buffer[ch][picoscope_bin_size:]
                    
                    bin_average = float(np.mean(bin_samples))
                    bin_timestamp = picoscope_bin_start_time[ch] + (picoscope_bin_size / 2) * picoscope_sample_period
                    
                    # FIXED: Update shared data for each channel
                    shared_data.update_picoscope_raw(ch, bin_timestamp, bin_average)
                    
                    picoscope_bin_start_time[ch] += picoscope_bin_size * picoscope_sample_period
        
        # Calculate and update RMS for each RMS channel
        for ch in picoscope_rms_channels:
            if picoscope_channels[ch]['enabled']:
                with data_lock:
                    if len(shared_data.picoscope_raw_data[ch]) > 0:
                        window_samples = int(picoscope_rms_window / picoscope_data_update_interval)
                        recent_data = shared_data.picoscope_raw_data[ch][-window_samples:]
                        rms_value = float(np.sqrt(np.mean(np.array(recent_data)**2)))
                        
                        current_time = time.time()  # Absolute timestamp
                        shared_data.update_picoscope_rms(ch, current_time, rms_value)
        
def collect_temperature_data():
    print(f"Temperature logging thread started")
    
    while not exit_event.is_set():
        time.sleep(temperature_sample_interval)
        
        current_time = time.time()  # Absolute timestamp
        temperature = read_temp_sensors()
        
        if temperature:
            # Calculate averages and std devs
            avg_temps = {}
            std_temps = {}
            
            with data_lock:
                for rtd in ACTIVE_RTDS:
                    temperature_data[rtd].append(temperature[rtd])
                    temperature_time_data.append(current_time)
                    
                    # Calculate over averaging window
                    window_samples = min(len(temperature_data[rtd]), 
                                       int(temperature_average_interval / temperature_sample_interval))
                    avg_temps[rtd] = float(np.mean(temperature_data[rtd][-window_samples:]))
                    std_temps[rtd] = float(np.std(temperature_data[rtd][-window_samples:]))
                
                # FIXED: Update shared data with all values
                shared_data.update_temperature(current_time, temperature, avg_temps, std_temps)
            
def collect_pressure_data():
    print(f"Pressure logging thread started")
    
    while not exit_event.is_set():
        time.sleep(pressure_sample_interval)
        
        current_time = time.time()  # Absolute timestamp
        pressure = read_pressure_sensor()
        
        if pressure:
            with data_lock:
                pressure_data.append(pressure)
                pressure_time_data.append(current_time)
                
                # Calculate over averaging window
                window_samples = min(len(pressure_data), 
                                   int(pressure_average_interval / pressure_sample_interval))
                avg_pressure = float(np.mean(pressure_data[-window_samples:]))
                std_pressure = float(np.std(pressure_data[-window_samples:]))
                
                # FIXED: Update shared data
                shared_data.update_pressure(current_time, pressure, avg_pressure, std_pressure)

def collect_octopus_data():
    while not exit_event.is_set():
        octopus_humidity, octopus_temperature, octopus_success = octopus.read_sensor()
        print(f"Octopus data read: {octopus_success}. Humidity {octopus_humidity}, temperature {octopus_temperature}")
        
        time.sleep(2)  # DHT11 requires at least 2 second interval between readings

# ================================================================================================
# PLOTTING FUNCTIONS
# ================================================================================================

def update_temperature_plot(frame):
    with data_lock:
        ax_temp.clear()
        times, temps, averages, stds = shared_data.get_temperature_rolling_data()
        
        for rtd in ACTIVE_RTDS:
            # Convert to relative time for plotting
            times_arr = np.array(times) - start_time
            
            if temperature_display_raw_data:
                temps_arr = np.array(temps[rtd])
                stds_arr = np.array(stds[rtd])
                ax_temp.plot(times_arr, temps_arr, label=f"{rtd}", linewidth=2)
                ax_temp.fill_between(times_arr, temps_arr - stds_arr, temps_arr + stds_arr, alpha=0.2)
            else:
                avgs_arr = np.array(averages[rtd])
                stds_arr = np.array(stds[rtd])
                ax_temp.plot(times_arr, avgs_arr, label=f"{rtd}", linewidth=2)
                ax_temp.fill_between(times_arr, avgs_arr - stds_arr, avgs_arr + stds_arr, alpha=0.2)
        
        ax_temp.legend(loc='upper left')
        ax_temp.set_xlabel("Time (s)")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_title(f"Temperature ({temperature_average_interval}s avg, ±1σ)")
        ax_temp.grid(True, alpha=0.3)

def update_pressure_plot(frame):
    with data_lock:
        ax_pressure.clear()
        times, pressures, averages, stds = shared_data.get_pressure_rolling_data()
        
        # Convert to relative time for plotting
        times_arr = np.array(times) - start_time
        
        if pressure_display_raw_data:
            press_arr = np.array(pressures)
            stds_arr = np.array(stds)
            ax_pressure.plot(times_arr, press_arr, linewidth=2)
            ax_pressure.fill_between(times_arr, press_arr - stds_arr, press_arr + stds_arr, alpha=0.2)
        else:
            avgs_arr = np.array(averages)
            stds_arr = np.array(stds)
            ax_pressure.plot(times_arr, avgs_arr, linewidth=2)
            ax_pressure.fill_between(times_arr, avgs_arr - stds_arr, avgs_arr + stds_arr, alpha=0.2)
        
        ax_pressure.set_xlabel("Time (s)")
        ax_pressure.set_ylabel("Pressure (Bar)")
        ax_pressure.set_title(f"Pressure ({pressure_average_interval}s avg, ±1σ)")
        ax_pressure.grid(True, alpha=0.3)

def update_picoscope_raw_plot(frame):
    with data_lock:
        ax_pico_raw.clear()
        times, voltages, peak_times = shared_data.get_picoscope_raw_rolling_data()
        
        for ch in picoscope_raw_channels:
            if picoscope_channels[ch]['enabled'] and ch in times:
                # Convert to relative time for plotting
                times_arr = np.array(times[ch]) - start_time
                volts_arr = np.array(voltages[ch])
                ax_pico_raw.plot(times_arr, volts_arr, label=f"Ch{ch}", linewidth=2)

                # Plot peaks if available
                peak_times_arr = np.array(peak_times[ch]) - start_time
                peak_volts_arr = volts_arr[[np.searchsorted(times_arr, pt) for pt in peak_times_arr]]
                ax_pico_raw.scatter(peak_times_arr, peak_volts_arr, color='red', label=f"Ch{ch} Peaks", zorder=5)

        ax_pico_raw.legend(loc='upper left')
        ax_pico_raw.set_xlabel("Time (s)")
        ax_pico_raw.set_ylabel("Voltage (V)")
        ax_pico_raw.set_title("Picoscope Raw Data")
        ax_pico_raw.grid(True, alpha=0.3)

def update_picoscope_rms_plot(frame):
    with data_lock:
        ax_pico_rms.clear()
        times, voltages = shared_data.get_picoscope_rms_rolling_data()
        
        for ch in picoscope_rms_channels:
            if picoscope_channels[ch]['enabled'] and ch in times:
                # Convert to relative time for plotting
                times_arr = np.array(times[ch]) - start_time
                volts_arr = np.array(voltages[ch])
                ax_pico_rms.plot(times_arr, volts_arr, label=f"Ch{ch} RMS", linewidth=2)
        
        ax_pico_rms.legend(loc='upper left')
        ax_pico_rms.set_xlabel("Time (s)")
        ax_pico_rms.set_ylabel("RMS Voltage (V)")
        ax_pico_rms.set_title(f"Picoscope RMS Data ({picoscope_rms_window}s window)")
        ax_pico_rms.grid(True, alpha=0.3)

def update_text_display(frame):
    # Get current values from your data (adjust these to match your actual data sources)
    current_temp = round(shared_data.get_temperature_rolling_data()[0][-1], 2)
    current_pressure = round(shared_data.get_pressure_rolling_data()[0][-1], 2)
    current_pico_raw = round(shared_data.get_picoscope_raw_rolling_data()[0][-1], 4)
    current_pico_rms = round(shared_data.get_picoscope_rms_rolling_data()[0][-1], 4)
    peaks = shared_data.get_picoscope_raw_rolling_data()[2]
    
    # Get temperature and humidity from other stream
    current_humidity, current_temperature = shared_data.get_octopus_data()
    current_temperature = round(current_temperature, 2)
    current_humidity = round(current_humidity, 1)

    # Format the text display
    text_content = f"""
    Temperature: {current_temp:.2f}°C

    Pressure: {current_pressure:.2f} bar

    Picoscope Raw: {current_pico_raw:.4f} V

    Picoscope RMS: {current_pico_rms:.4f} V

    Recent Fringe Peaks: {', '.join(f'{p:.2f}s' for p in peaks.get('A', [])[-5:])}

    {'='*25}

    Ambient Temp: {current_temperature:.2f}°C

    Humidity: {current_humidity:.1f}%
    """
    
    text_display.set_text(text_content)
    return text_display

# ================================================================================================
# MAIN FUNCTION
# ================================================================================================

if __name__ == "__main__":
    """
    Main program execution.
    
    This section:
    1. Initializes hardware connections
    2. Configures sensor channels  
    3. Starts background threads for data logging and control
    4. Runs the real-time plotting interface
    5. Handles clean shutdown on user interrupt
    """
    
    try:
        print(f"Configuration:")
        print(f"  PicoScope: {'ON' if enable_picoscope else 'OFF'}")
        print(f"  Active sensors: {len(ACTIVE_RTDS)} RTDs")
        print("-" * 80)
        
        # Initialize PicoScope if enabled
        if enable_picoscope:
            print("Initializing PicoScope 2205A...")
            picoscope = PicoScopeReader()
            if not picoscope.open(picoscope_channels):
                print("WARNING: PicoScope initialization failed, continuing without it")
                enable_picoscope = False
                picoscope = None

        # Initialize LabJack connection and configure sensors
        print("Initializing LabJack connection...")
        handle = ljm.openS("T7", "ANY", "ANY")

        if enable_octopus:
            print("Initializing octopus...")
            octopus = OctopusReader(handle)
        
        # Configure all sensor types
        configure_rtd(handle)

        print(f"\nStarting system... Data logging to: {csv_filename}")
        print("Close the plot window or press Ctrl+C to stop.\n")

        # ================================================================================================
        # THREAD STARTUP
        # ================================================================================================
        if enable_picoscope:
            pico_thread = threading.Thread(target=collect_picoscope_data, daemon=True)
            pico_thread.start()
        
        if enable_temperature:
            temp_thread = threading.Thread(target=collect_temperature_data, daemon=True)
            temp_thread.start()

        if enable_pressure:
            pressure_thread = threading.Thread(target=collect_pressure_data, daemon=True)
            pressure_thread.start()

        if enable_octopus:
            octopus_thread = threading.Thread(target=collect_octopus_data, daemon=True)
            octopus_thread.start()

        csv_writer_thread = threading.Thread(target=csv_writer_thread, daemon=True)
        csv_writer_thread.start()

        print("All threads started successfully!")
        print("-" * 80)

        # ================================================================================================
        # MAIN LOOP
        # ================================================================================================
        
        # Initialize matplotlib for real-time plotting with gridspec for a custom layout
        print("Starting display...")
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.4], hspace=0.15, wspace=0.15)

        # Create the 2x2 plot grid on the left
        ax_temp = fig.add_subplot(gs[0, 0])
        ax_pressure = fig.add_subplot(gs[0, 1])
        ax_pico_raw = fig.add_subplot(gs[1, 0])
        ax_pico_rms = fig.add_subplot(gs[1, 1])

        fig.suptitle('Sensor Monitoring Dashboard', fontsize=16, fontweight='bold')

        # Create text display area on the right (spans both rows)
        ax_text = fig.add_subplot(gs[:, 2])
        ax_text.axis('off')  # Hide axes for text area
        text_display = ax_text.text(0.1, 0.5, '', fontsize=12, verticalalignment='center', family='monospace', transform=ax_text.transAxes)
        
        # Create animations for all four plots
        ani_temp = animation.FuncAnimation(fig, update_temperature_plot, interval=100, cache_frame_data=False)
        ani_pressure = animation.FuncAnimation(fig, update_pressure_plot, interval=100, cache_frame_data=False)
        ani_pico_raw = animation.FuncAnimation(fig, update_picoscope_raw_plot, interval=100, cache_frame_data=False)
        ani_pico_rms = animation.FuncAnimation(fig, update_picoscope_rms_plot, interval=100, cache_frame_data=False)
        ani_text = animation.FuncAnimation(fig, update_text_display, interval=100, cache_frame_data=False)
        
        # Start matplotlib GUI (blocks until window is closed)
        plt.show()

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("SHUTDOWN: User interrupt received (Ctrl+C)")
        print("=" * 80)
        
        print(f"Data saved to: {csv_filename}")
        print("Stopping all threads...")
        
        # Signal all threads to stop
        exit_event.set()  
        time.sleep(1)  # Give threads time to clean up
        
        # Close PicoScope connection
        if enable_picoscope and picoscope:
            picoscope.close()
        
        print("Shutdown complete.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n⚠️  CRITICAL ERROR: {e}")
        print("Check hardware connections and configuration.")
        
        # Close PicoScope connection on error
        if enable_picoscope and picoscope:
            picoscope.close()
        
        exit_event.set()
        sys.exit(1)

# ================================================================================================
# END OF PROGRAM
# ================================================================================================