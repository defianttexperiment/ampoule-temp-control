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
from collections import deque
from datetime import datetime
from rscomm import *
from dotenv import load_dotenv

# PicoScope imports
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
import ctypes

print([attr for attr in dir(ps) if 'stream' in attr.lower()])

# ================================================================================================
# CONFIGURATION SECTION - Modify these parameters to customize system behavior
# ================================================================================================

# ---------------- PICOSCOPE CONFIGURATION ----------------
enable_picoscope = True                     # Enable PicoScope data acquisition (photodiode light intensity)
picoscope_channels = {
    'A': {'enabled': True, 'range': 10.0, 'coupling': 'DC'},
    'B': {'enabled': False, 'range': 10.0, 'coupling': 'DC'}
}
# Available ranges: 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0 (volts). Coupling options: 'DC' or 'AC'

sample_interval_us = 50    # Microseconds between samples (don't go above 50, Picoscope can't go that slow)
picoscope_rms_window = 30  # Window (seconds) for RMS calculation
picoscope_display_window = 60   # Window (seconds) for Picoscope data displayed 

# ---------------- DATA LOGGING CONFIGURATION ----------------
log_data_average_interval = 30     # Window (seconds) for temp & pressure averaging
log_data_display_window = 120      # Window (seconds) for temp & pressure display
log_data_record_raw_data = True    # Record individual readings vs averaged data in CSV
                                   # True = raw 1-second readings, False = averaged readings
BIN_SIZE_SAMPLES = 1000            # Average every 1000 samples (50ms at 20kHz)

# ---------------- SENSOR CONFIGURATION ----------------
# Define which sensors are connected and active

# Format: "Display Name": {"channel": AIN_number, "type": rtd_type}
# Types: "PT100", "PT500", "PT1000"
ACTIVE_RTDS = {
    "RTD-1": {"channel": 3, "type": "PT100"}  # Example: PT100 on AIN3
}

# ---------------- HARDWARE CONFIGURATION ----------------
load_dotenv()
supply_port = os.getenv("SERIAL_PORT")     # USB port name that connects via RS232 to power supply
fluke_port = "/dev/tty.usbserial-AV0L2AIU"

# ================================================================================================
# SYSTEM INITIALIZATION - Automatic configuration based on user settings
# ================================================================================================

# Generate unique filename with timestamp for data logging
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
csv_filename = f"sensor_readings_{timestamp}.csv"
print(f"Data will be logged to: {csv_filename}")

# ================================================================================================
# DATA STRUCTURES - Storage for sensor data and system state
# ================================================================================================

# Live plotting data storage - keeps recent data in memory for real-time display
time_data = []  # Time points for x-axis
rtd_temp_data = {rtd: [] for rtd in ACTIVE_RTDS}  # RTD temperature history
rtd_temp_std = {rtd: [] for rtd in ACTIVE_RTDS}  # RTD temperature standard deviation
voltage_data = []      # Power supply voltage history
current_data = []      # Power supply current history
start_time = time.time()  # Program start timestamp for relative timing

# PicoScope data storage
picoscope_data = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_rolling_data = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}

# PicoScope RMS data storage (for 30s rolling window)
picoscope_rms_data = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_rms_rolling_data = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_binned_data = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_binned_times = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_bin_buffer = {ch: [] for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
picoscope_bin_start_time = {ch: None for ch in picoscope_channels if picoscope_channels[ch]['enabled']}
SAMPLE_PERIOD = sample_interval_us / 1_000_000  # Convert microseconds to seconds

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
        self.lock = threading.Lock()  # Prevents simultaneous access from multiple threads
        self.current_temp = None      # None indicates no data available yet
        self.avg_rtd = {}             # Averaged RTD data by sensor name
        self.current_voltage = None   # Latest voltage reading
        self.current_current = None   # Latest current reading
        self.current_pressure = None  # Latest pressure reading
        self.picoscope_voltages = {}  # Latest PicoScope readings

    def update_temperature(self, temp,  avg_rtd_data):
        """Update temperature data in thread-safe manner."""
        with self.lock:
            self.current_temp = temp
            self.avg_rtd = avg_rtd_data.copy()

    def update_voltage(self, voltage):
        """Update voltage measurement in thread-safe manner."""
        with self.lock:
            self.current_voltage = voltage

    def update_current(self, current):
        """Update current measurement in thread-safe manner."""
        with self.lock:
            self.current_current = current
    
    def update_pressure(self, pressure):
        """Update pressure measurement in thread-safe manner."""
        with self.lock:
            self.current_pressure = pressure
    
    def update_picoscope(self, channel, voltage):
        """Update PicoScope voltage measurement in thread-safe manner."""
        with self.lock:
            self.picoscope_voltages[channel] = voltage
    
    def get_temperature(self):
        """Get current temperature reading (thread-safe)."""
        with self.lock:
            return self.current_temp

    def get_avg_temperature(self):
        """Get average of all active sensor readings (thread-safe)."""
        with self.lock:
            all_temps = []
            if self.avg_rtd:
                all_temps.extend(self.avg_rtd.values())
            
            if all_temps:
                return np.mean(all_temps)  # Changed from median to mean
            return None
    
    def get_all_data(self):
        """Get complete snapshot of all temperature data (thread-safe)."""
        with self.lock:
            return (self.current_temp,
                   self.avg_rtd.copy(), self.current_pressure, 
                   self.picoscope_voltages.copy())
    
    def get_picoscope_rms(self, channel):
        """Get RMS value for PicoScope channel (thread-safe)."""
        with self.lock:
            return self.picoscope_voltages.get(f"{channel}_rms", None)
    
    def update_picoscope_rms(self, channel, rms_value):
        """Update PicoScope RMS value in thread-safe manner."""
        with self.lock:
            self.picoscope_voltages[f"{channel}_rms"] = rms_value

# Create global shared data instance
shared_data = SharedData()

# Rolling data storage for averaging - keeps recent readings for smoothing
rtd_rolling_data = {rtd: [] for rtd in ACTIVE_RTDS}  # RTD rolling data for averaging

# Add rolling buffer for pressure for smoothing/plotting
pressure_data = []  # List[float]: pressure values matching time_data
pressure_std = []   # List[float]: pressure standard deviation values
pressure_rolling_data = [] # list of the most recent values (for smoothing)

# Thread synchronization objects
data_lock = threading.Lock()  # Protects plotting data
exit_event = threading.Event()  # Signals all threads to stop

# ================================================================================================
# CSV FILE INITIALIZATION - Create data logging file with appropriate headers
# ================================================================================================

def create_csv_file():
    """Create CSV file with headers based on active sensors and configuration."""
    headers = ["Timestamp (HH-MM-SS)", "Time (s)"]
    
    # Add mean and std dev columns for each RTD
    for rtd in ACTIVE_RTDS:
        headers.extend([f"{rtd} RTD Mean (°C)", f"{rtd} RTD StdDev (°C)"])
    
    # Add mean and std dev for pressure
    headers.extend(["Pressure Mean (Bar)", "Pressure StdDev (Bar)"])
    
    # Add PicoScope raw columns
    if enable_picoscope:
        for ch in picoscope_channels:
            if picoscope_channels[ch]['enabled']:
                headers.append(f"Photodiode Ch{ch} Raw (V)")
    
    # Add PicoScope RMS columns
    if enable_picoscope:
        for ch in picoscope_channels:
            if picoscope_channels[ch]['enabled']:
                headers.append(f"Photodiode Ch{ch} RMS (V)")
    
    # Create file and write headers
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Initialize CSV file
create_csv_file()

# ================================================================================================
# PICOSCOPE FUNCTIONS - PicoScope 2205A configuration and data acquisition
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

    def start_streaming(self, sample_interval_us=1000):
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
            sample_interval_ns = sample_interval_us * 1000
            
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
            print(f"Streaming started at {1000000/sample_interval_us:.1f} Hz")
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
                    print(f"Channel {ch}: No data collected")
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

# ================================================================================================
# HARDWARE CONFIGURATION FUNCTIONS - Set up LabJack channels for different sensor types
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
        ain_channel = config["channel"]
        rtd_type = config["type"]  # "PT100", "PT500", or "PT1000"
        
        # Map RTD type to resistance value for EF_CONFIG_A
        rtd_resistance_map = {
            "PT100": 100,
            "PT500": 500,
            "PT1000": 1000
        }
        
        if rtd_type not in rtd_resistance_map:
            print(f"  ⚠️ Warning: Unknown RTD type '{rtd_type}' for {name}, skipping")
            continue
        
        resistance = rtd_resistance_map[rtd_type]
        
        # Set input range
        ljm.eWriteName(handle, f"AIN{ain_channel}_RANGE", 1.0)
        # Set high resolution for precise measurement
        ljm.eWriteName(handle, f"AIN{ain_channel}_RESOLUTION_INDEX", 12)
        range_readback = ljm.eReadName(handle, f"AIN{ain_channel}_RANGE")
        print(f"  AIN{ain_channel} range set to: {range_readback}V")
        
        # Configure extended feature for RTD processing
        # Extended Feature Index 40 = RTD
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_INDEX", 40)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_A", 1)  # Output in °C
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_B", 0)  # Excitation circuit 0, 200 µA source
        
        print(f"  Configured {name} RTD ({rtd_type}) on AIN{ain_channel}")

# ================================================================================================
# SENSOR READING FUNCTIONS - Acquire data from hardware
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
        handle = ljm.openS("T7", "ANY", "ANY")  # Connect to any available LabJack T7
        
        # Read RTD temperatures using extended features
        rtd_temps = {}
        for name, config in ACTIVE_RTDS.items():
            channel = config['channel']
            # Read processed temperature from extended feature
            temp = ljm.eReadName(handle, f"AIN{channel}_EF_READ_A")
            rtd_temps[name] = temp
        
        ljm.close(handle)  # Clean up connection
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
        print(f"⚠️ Serial error: {e}")
        return None
    
# ================================================================================================
# PICOSCOPE DATA COLLECTION THREAD - High-speed continuous acquisition
# ================================================================================================

def collect_picoscope_data():
    """
    Dedicated high-speed thread for PicoScope data acquisition.
    
    Runs at ~20Hz (50ms intervals) to continuously collect and process
    streaming data from the PicoScope independently of other sensors.
    """
    global start_time
    
    print("PicoScope collection thread started")
    
    # Start PicoScope streaming
    if picoscope:
        picoscope.start_streaming(sample_interval_us=sample_interval_us)
        print(f"PicoScope streaming started at {1000000/sample_interval_us:.0f} Hz")
    
    while not exit_event.is_set():
        time.sleep(0.05)  # 50ms = 20Hz update rate
        
        if not picoscope:
            continue
            
        # Get most recent streaming samples
        stream_data = picoscope.read_voltages()

        for ch, values in stream_data.items():
            if values is not None and len(values) > 0:
                current_time = time.time() - start_time
                
                # Initialize bin start time if needed
                if picoscope_bin_start_time[ch] is None:
                    picoscope_bin_start_time[ch] = current_time
                
                # Add all new samples to bin buffer
                picoscope_bin_buffer[ch].extend(values)
                
                # Process complete bins
                while len(picoscope_bin_buffer[ch]) >= BIN_SIZE_SAMPLES:
                    # Extract one bin worth of samples
                    bin_samples = picoscope_bin_buffer[ch][:BIN_SIZE_SAMPLES]
                    picoscope_bin_buffer[ch] = picoscope_bin_buffer[ch][BIN_SIZE_SAMPLES:]
                    
                    # Calculate average for this bin
                    bin_average = float(np.mean(bin_samples))
                    
                    # Calculate the timestamp for the middle of this bin
                    bin_timestamp = picoscope_bin_start_time[ch] + (BIN_SIZE_SAMPLES / 2) * SAMPLE_PERIOD
                    
                    # Store binned data with timestamp
                    with data_lock:
                        picoscope_binned_data[ch].append(bin_average)
                        picoscope_binned_times[ch].append(bin_timestamp)
                        
                        # Limit to last picoscope_display_window seconds of binned data
                        if len(picoscope_binned_data[ch]) > picoscope_display_window*20:
                            picoscope_binned_data[ch] = picoscope_binned_data[ch][-picoscope_display_window*20:]
                            picoscope_binned_times[ch] = picoscope_binned_times[ch][-picoscope_display_window*20:]
                    
                    # Update bin start time for next bin
                    picoscope_bin_start_time[ch] += BIN_SIZE_SAMPLES * SAMPLE_PERIOD
                
                # Use last bin average for current value, or last sample if no complete bins yet
                if picoscope_binned_data[ch]:
                    voltage = picoscope_binned_data[ch][-1]
                else:
                    voltage = float(values[-1])

                # --- Update shared data structures ---
                shared_data.update_picoscope(ch, voltage)
                
                with data_lock:
                    picoscope_rolling_data[ch].append(voltage)
                    if len(picoscope_rolling_data[ch]) > log_data_display_window:
                        picoscope_rolling_data[ch].pop(0)

                    # Maintain RMS window with all samples
                    picoscope_rms_rolling_data[ch].extend(values)
                    # Keep only last picoscope_rms_window seconds worth of data
                    num_picoscope_samples = int(picoscope_rms_window*1000000/sample_interval_us)
                    if len(picoscope_rms_rolling_data[ch]) > num_picoscope_samples:
                        picoscope_rms_rolling_data[ch] = picoscope_rms_rolling_data[ch][-num_picoscope_samples:]

                    # Compute RMS and share
                    rms_value = np.sqrt(np.mean(np.square(picoscope_rms_rolling_data[ch])))
                    shared_data.update_picoscope_rms(ch, rms_value)

# ================================================================================================
# DATA LOGGING THREAD - Background data acquisition and storage
# ================================================================================================

def log_data():
    """
    Background thread for continuous data acquisition and logging.
    
    This function runs continuously, reading sensors every second and:
    1. Maintaining rolling averages for noise reduction
    2. Updating shared data for other threads
    3. Logging data to CSV file
    4. Printing status to console
    
    Args:
        supply: Power supply object for voltage/current measurements
    """
    global start_time

    print(f"Data logging thread started with {log_data_average_interval}s averaging window")
    
    while not exit_event.is_set():
        # Read all sensors
        rtd_temps = read_temp_sensors()
        pressure = read_pressure_sensor()
        
        # Get current PicoScope values from shared data (collected by separate thread)
        pico_voltages = {}
        if enable_picoscope:
            for ch in picoscope_channels:
                if picoscope_channels[ch]['enabled']:
                    voltage = shared_data.picoscope_voltages.get(ch)
                    if voltage is not None:
                        pico_voltages[ch] = voltage
        
        # Process temperature data if readings were successful
        if rtd_temps is not None:
            with data_lock:  # Thread-safe data update
                # Update rolling averages for RTDs
                for rtd in ACTIVE_RTDS:
                    if rtd_temps and rtd in rtd_temps:
                        rtd_rolling_data[rtd].append(rtd_temps[rtd])
                        if len(rtd_rolling_data[rtd]) > log_data_average_interval:
                            rtd_rolling_data[rtd].pop(0)
        
        # Store pressure for smoothing
        if pressure is not None:
            pressure_rolling_data.append(pressure)
            if len(pressure_rolling_data) > log_data_display_window:
                pressure_rolling_data.pop(0)
        
        # Smoothing/averaging - use mean instead of median
        if pressure_rolling_data:
            avg_pressure = float(np.mean(pressure_rolling_data))
        else:
            avg_pressure = None
        shared_data.update_pressure(avg_pressure)

        # Determine if we have enough data to process and log
        should_process = len(time_data) == 0  # Always process first iteration
        
        # Check if we have any sensor data to process
        if ACTIVE_RTDS:
            first_rtd = next(iter(ACTIVE_RTDS))
            should_process = should_process or len(rtd_rolling_data[first_rtd]) > 0
            
        # Skip processing if no active sensors
        if not ACTIVE_RTDS:
            should_process = False

        # Process and log data
        if should_process:
            current_time = round(time.time() - start_time, 1)
            timestamp_str = datetime.now().strftime('%H-%M-%S')

            with data_lock:
                # Calculate means and standard deviations
                avg_rtd = {rtd: np.mean(rtd_rolling_data[rtd]) for rtd in ACTIVE_RTDS if rtd_rolling_data[rtd]}
                std_rtd = {rtd: np.std(rtd_rolling_data[rtd]) if len(rtd_rolling_data[rtd]) > 1 else 0.0 
                          for rtd in ACTIVE_RTDS if rtd_rolling_data[rtd]}
                
                avg_picoscope = {ch: np.mean(picoscope_rolling_data[ch]) for ch in picoscope_rolling_data if picoscope_rolling_data[ch]}
                
                # Calculate RMS values for PicoScope channels
                rms_picoscope = {}
                for ch in picoscope_rms_rolling_data:
                    if picoscope_rms_rolling_data[ch]:
                        rms_picoscope[ch] = np.sqrt(np.mean(np.array(picoscope_rms_rolling_data[ch])**2))
                
                # Calculate pressure statistics
                if pressure_rolling_data:
                    avg_pressure = float(np.mean(pressure_rolling_data))
                    std_pressure = float(np.std(pressure_rolling_data)) if len(pressure_rolling_data) > 1 else 0.0
                else:
                    avg_pressure = None
                    std_pressure = None
                
                current_rtd = {rtd: rtd_rolling_data[rtd][-1] for rtd in ACTIVE_RTDS if rtd_rolling_data[rtd]}
                current_picoscope = {ch: picoscope_rolling_data[ch][-1] for ch in picoscope_rolling_data if picoscope_rolling_data[ch]}
                
                # Update shared data for control threads
                # Determine current temperature from available sensors
                current_temp = None
                if ACTIVE_RTDS and avg_rtd:
                    current_temp = list(avg_rtd.values())[0]
                
                if current_temp is not None:
                    shared_data.update_temperature(current_temp, avg_rtd)

                # Update plotting data
                time_data.append(current_time)
                for rtd in ACTIVE_RTDS:
                    if rtd in avg_rtd:
                        rtd_temp_data[rtd].append(avg_rtd[rtd])
                        rtd_temp_std[rtd].append(std_rtd[rtd])
                
                # Update PicoScope plotting data
                for ch in picoscope_data:
                    if ch in avg_picoscope:
                        picoscope_data[ch].append(avg_picoscope[ch])
                
                # Update PicoScope RMS plotting data
                for ch in picoscope_rms_data:
                    if ch in rms_picoscope:
                        picoscope_rms_data[ch].append(rms_picoscope[ch])

                # Limit plotting data to last log_data_display_window seconds
                if len(time_data) > log_data_display_window*10:
                    time_data.pop(0)
                    for rtd in rtd_temp_data:
                        if rtd_temp_data[rtd]:
                            rtd_temp_data[rtd].pop(0)
                            rtd_temp_std[rtd].pop(0)
                    for ch in picoscope_data:
                        if picoscope_data[ch]:
                            picoscope_data[ch].pop(0)
                    for ch in picoscope_rms_data:
                        if picoscope_rms_data[ch]:
                            picoscope_rms_data[ch].pop(0)

                # Add mean pressure and std dev from rolling buffer
                if avg_pressure is not None:
                    pressure_data.append(avg_pressure)
                    pressure_std.append(std_pressure)
                else:
                    pressure_data.append('')
                    pressure_std.append('')
            
            # Write to CSV file
            write_to_csv(timestamp_str, current_time, current_rtd,
                        avg_rtd, std_rtd, avg_pressure, std_pressure,
                        current_picoscope, avg_picoscope, rms_picoscope)

def write_to_csv(timestamp_str, current_time, current_rtd, 
                 avg_rtd, std_rtd, avg_pressure, std_pressure,
                 current_picoscope, avg_picoscope, rms_picoscope):
    """
    Write sensor data to CSV file.
    
    Chooses between raw data or averaged data based on configuration.
    Includes electrical measurements if enabled.
    """
    try:
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Prepare data row
            row = [timestamp_str, current_time]
            
            # Add RTD mean and std dev
            for rtd in ACTIVE_RTDS:
                if log_data_record_raw_data:
                    row.extend([current_rtd.get(rtd, ''), ''])  # Raw data has no std dev
                else:
                    row.extend([avg_rtd.get(rtd, ''), std_rtd.get(rtd, '')])
            
            # Add pressure mean and std dev
            if log_data_record_raw_data:
                if pressure_rolling_data:
                    row.extend([pressure_rolling_data[-1], ''])  # Raw data has no std dev
                else:
                    row.extend(['', ''])
            else:
                row.extend([avg_pressure if avg_pressure is not None else '', 
                           std_pressure if std_pressure is not None else ''])
            
            # Add PicoScope raw data (raw or averaged)
            if enable_picoscope:
                for ch in picoscope_channels:
                    if picoscope_channels[ch]['enabled']:
                        if log_data_record_raw_data:
                            row.append(current_picoscope.get(ch, ''))
                        else:
                            row.append(avg_picoscope.get(ch, ''))
            
            # Add PicoScope RMS data
            if enable_picoscope:
                for ch in picoscope_channels:
                    if picoscope_channels[ch]['enabled']:
                        row.append(rms_picoscope.get(ch, ''))
            
            writer.writerow(row)
            
    except Exception as e:
        print(f"CSV write error: {e}")

# ================================================================================================
# PLOTTING FUNCTIONS - Real-time data visualization with dual windows
# ================================================================================================

def update_temperature_plot(frame):
    """
    Update temperature plot with all temperature sensors and error bars.
    
    Args:
        frame: Animation frame number (unused but required by matplotlib)
    """
    with data_lock:
        ax_temp.clear()
        
        # Plot RTD data with error bars (standard deviation)
        for rtd in ACTIVE_RTDS:
            if rtd_temp_data[rtd] and rtd_temp_std[rtd]:
                # Convert to numpy arrays for plotting
                times = np.array(time_data)
                temps = np.array(rtd_temp_data[rtd])
                stds = np.array(rtd_temp_std[rtd])
                
                # Plot line
                ax_temp.plot(times, temps, label=f"{rtd}", 
                           linestyle='-', linewidth=2)
                
                # Add shaded error region (±1 standard deviation)
                ax_temp.fill_between(times, temps - stds, temps + stds, 
                                    alpha=0.2)

        ax_temp.legend(loc='upper left')

    # Format plot appearance
    ax_temp.set_xlabel("Time (s)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_title("Temperature Measurements (15s mean, ±1 StdDev)")
    ax_temp.grid(True, alpha=0.3)
    
    if time_data:
        ax_temp.set_xlim(max(0, time_data[-1] - log_data_display_window), time_data[-1] + 1)


def update_pressure_plot(frame):
    """
    Update pressure plot with error bars.
    
    Args:
        frame: Animation frame number (unused but required by matplotlib)
    """
    with data_lock:
        ax_pressure.clear()
        
        # Plot pressure data with error bars (standard deviation)
        if pressure_data and any(type(p)==float for p in pressure_data):
            # Filter out empty strings and convert to numpy arrays
            valid_indices = [i for i, p in enumerate(pressure_data) if type(p)==float]
            if valid_indices:
                times = np.array([time_data[i] for i in valid_indices])
                pressures = np.array([pressure_data[i] for i in valid_indices])
                stds = np.array([pressure_std[i] if type(pressure_std[i])==float else 0.0 
                               for i in valid_indices])
                
                # Plot line
                ax_pressure.plot(times, pressures, color="tab:green", 
                               linestyle='-', linewidth=2, label="Pressure")
                
                # Add shaded error region (±1 standard deviation)
                ax_pressure.fill_between(times, pressures - stds, pressures + stds, 
                                        color="tab:green", alpha=0.2)
                
                ax_pressure.legend(loc='upper left')

    # Format plot appearance
    ax_pressure.set_xlabel("Time (s)")
    ax_pressure.set_ylabel("Pressure (Bar)")
    ax_pressure.set_title("Pressure Measurements (15s mean, ±1 StdDev)")
    ax_pressure.grid(True, alpha=0.3)
    
    if time_data:
        ax_pressure.set_xlim(max(0, time_data[-1] - log_data_display_window), time_data[-1] + 1)


def update_picoscope_raw_plot(frame):
    """
    Update PicoScope raw data plot with 20Hz binned averages.
    
    Args:
        frame: Animation frame number (unused but required by matplotlib)
    """
    with data_lock:
        ax_pico_raw.clear()
        
        # Plot binned PicoScope data (20 points/second)
        if enable_picoscope:
            color_map = {'A': 'tab:orange', 'B': 'tab:purple'}
            for ch in picoscope_binned_data:
                if picoscope_binned_data[ch] and picoscope_binned_times[ch]:
                    ax_pico_raw.plot(picoscope_binned_times[ch], picoscope_binned_data[ch], 
                                   color=color_map.get(ch, 'tab:orange'),
                                   linestyle='-', linewidth=1, 
                                   label=f"Photodiode Ch{ch}")
            
            ax_pico_raw.legend(loc='upper left')
            
            # Set x-axis limits based on latest data
            all_times = []
            for ch in picoscope_binned_times:
                if picoscope_binned_times[ch]:
                    all_times.extend(picoscope_binned_times[ch])
            
            if all_times:
                latest_time = max(all_times)
                ax_pico_raw.set_xlim(max(0, latest_time - picoscope_display_window), latest_time + 1)

    # Format plot appearance
    ax_pico_raw.set_xlabel("Time (s)")
    ax_pico_raw.set_ylabel("Voltage (V)")
    ax_pico_raw.set_title("PicoScope Raw Data (0.05s bins)")
    ax_pico_raw.grid(True, alpha=0.3)


def update_picoscope_rms_plot(frame):
    """
    Update PicoScope RMS data plot.
    
    Args:
        frame: Animation frame number (unused but required by matplotlib)
    """
    with data_lock:
        ax_pico_rms.clear()
        
        # Plot RMS PicoScope data
        if enable_picoscope:
            color_map_rms = {'A': 'tab:red', 'B': 'tab:pink'}
            for ch in picoscope_rms_data:
                if picoscope_rms_data[ch]:
                    ax_pico_rms.plot(time_data, picoscope_rms_data[ch], 
                                   color=color_map_rms.get(ch, 'tab:red'),
                                   linestyle='-', linewidth=2, 
                                   label=f"Photodiode Ch{ch} RMS")
            
            ax_pico_rms.legend(loc='upper left')

    # Format plot appearance
    ax_pico_rms.set_xlabel("Time (s)")
    ax_pico_rms.set_ylabel("RMS Voltage (V)")
    ax_pico_rms.set_title("PicoScope RMS Data (30s window)")
    ax_pico_rms.grid(True, alpha=0.3)
    
    if time_data:
        ax_pico_rms.set_xlim(max(0, time_data[-1] - log_data_display_window), time_data[-1] + 1)

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
        print(f"Connected to LabJack T7")
        
        # Configure all sensor types
        configure_rtd(handle)
        ljm.close(handle)  # Close configuration connection

        print(f"\nStarting system... Data logging to: {csv_filename}")
        print("Close the plot window or press Ctrl+C to stop.\n")

        # ================================================================================================
        # THREAD STARTUP - Launch background processes
        # ================================================================================================
        pico_thread = threading.Thread(target=collect_picoscope_data, daemon=True)
        pico_thread.start()
        print("✓ Picoscope data logging thread started")
        
        log_thread = threading.Thread(target=log_data, daemon=True)
        log_thread.start()
        print("✓ Main data logging thread started")

        print("\nAll threads started successfully!")
        print("-" * 80)

        # ================================================================================================
        # MAIN LOOP - Real-time plotting interface with four separate plots
        # ================================================================================================
        
        # Initialize matplotlib for real-time plotting with four subplots
        print("Starting real-time plot display...")
        fig, ((ax_temp, ax_pressure), (ax_pico_raw, ax_pico_rms)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Sensor Monitoring Dashboard', fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Create animations for all four plots
        ani_temp = animation.FuncAnimation(fig, update_temperature_plot, interval=50, cache_frame_data=False)
        ani_pressure = animation.FuncAnimation(fig, update_pressure_plot, interval=50, cache_frame_data=False)
        ani_pico_raw = animation.FuncAnimation(fig, update_picoscope_raw_plot, interval=50, cache_frame_data=False)
        ani_pico_rms = animation.FuncAnimation(fig, update_picoscope_rms_plot, interval=50, cache_frame_data=False)
        
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
        time.sleep(2)  # Give threads time to clean up
        
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