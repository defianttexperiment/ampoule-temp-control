"""
Temperature Monitoring and Control System
========================================

This program provides comprehensive temperature monitoring and control for laboratory experiments.
It interfaces with a LabJack device for sensor readings and power supplies for thermal control.

Key Features:
- Real-time temperature monitoring from multiple sensor types (thermocouples, TSic, thermistors)
- Live plotting of temperature data with matplotlib
- CSV data logging with timestamps
- PID-based temperature control
- Automated voltage sweep capabilities
- Multi-threaded operation for concurrent data acquisition and control

Hardware Requirements:
- LabJack T7
- E3644A power supply
- Temperature sensors (J-type thermocouples, TSic sensors, and/or thermistors)
"""

from statistics import mean, median
import time
import sys
import csv  
import threading
import datetime
import os
from labjack import ljm  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rscomm import *

# ================================================================================================
# CONFIGURATION SECTION - Modify these parameters to customize system behavior
# ================================================================================================

# ---------------- HARDWARE CONFIGURATION ----------------
supply_port = "/dev/tty.PL2303G-USBtoUART120"      # USB port name that connects via RS232 to power supply

# ---------------- DATA LOGGING CONFIGURATION ----------------
# Controls what data is collected and how it's processed

read_voltage_current = False       # Enable voltage/current monitoring from power supply
                                   # Requires rscomm library and compatible power supply

run_log_data = True                # Enable temperature data logging to CSV
log_data_average_interval = 15     # Number of readings to average for smoothed plotting (seconds)
log_data_record_raw_data = True    # Record individual readings vs averaged data in CSV
                                   # True = raw 1-second readings, False = averaged readings

# ---------------- TEMPERATURE CONTROL CONFIGURATION ----------------
# Only ONE control method should be enabled at a time

run_slow_control = False           # Basic voltage sweep control (0-1V linear ramp)
run_pid_control = False            # PID control to maintain constant temperature
run_pid_slow_control = True        # PID-assisted temperature ramping between setpoints

# PID Control Parameters (for run_pid_control = True)
pid_desired_temp = 17.3            # Target temperature in Celsius
pid_interval = 15                  # PID update interval in seconds

# PID Slow Control Parameters (for run_pid_slow_control = True)
pid_slow_control_starting_temp = 16.4      # Starting temperature for ramp
pid_slow_control_ending_temp = 18.5        # Ending temperature for ramp
pid_slow_control_voltage_finding_time = 300     # Time to find stable voltage for each endpoint (seconds)
pid_slow_control_intermediate_settling_time = 0 # Wait time between voltage finding phases
pid_slow_control_swing_time = 1800         # Time for temperature ramp between endpoints (seconds)

# ---------------- TIMEOUT CONFIGURATION ----------------
# Automatic program termination

run_timeout = False                # Enable automatic program shutdown
timeout_length = 1800             # Program runtime in seconds (1800 = 30 minutes)

# ---------------- SENSOR CONFIGURATION ----------------
# Define which sensors are connected and active

# J-type Thermocouples connected to LabJack AIN channels
# Format: "Display Name": {"channel": AIN_number, "type": thermocouple_type}
# Type 21 = J-type thermocouple
ACTIVE_THERMOCOUPLES = {
    # "J-2 (center)": {"channel": 2, "type": 21}  # Example: J-type on AIN2
}

# TSic Temperature Sensors (digital sensors with analog output)
# List of AIN channel numbers where TSic sensors are connected
ACTIVE_TSIC_CHANNELS = [0]

# Thermistor Temperature Sensors
# List of AIN channel numbers where thermistors are connected
ACTIVE_THERMISTOR_CHANNELS = []

# ================================================================================================
# SYSTEM INITIALIZATION - Automatic configuration based on user settings
# ================================================================================================

# Ensure only one temperature control method is active
if run_pid_control and run_slow_control:
    print("Warning: Both PID and slow control enabled. Disabling slow control.")
    run_slow_control = False

if run_pid_control and run_pid_slow_control:
    print("Warning: Both PID control modes enabled. Disabling basic PID control.")
    run_pid_control = False

# Generate unique filename with timestamp for data logging
timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
csv_filename = f"sensor_readings_{timestamp}.csv"
print(f"Data will be logged to: {csv_filename}")

# ================================================================================================
# DATA STRUCTURES - Storage for sensor data and system state
# ================================================================================================

# Live plotting data storage - keeps recent data in memory for real-time display
time_data = []  # Time points for x-axis
thermo_temp_data = {tc: [] for tc in ACTIVE_THERMOCOUPLES}  # Thermocouple temperature history
tsic_temp_data = {ch: [] for ch in ACTIVE_TSIC_CHANNELS}    # TSic temperature history
thermistor_temp_data = {tm: [] for tm in ACTIVE_THERMISTOR_CHANNELS}  # Thermistor temperature history
voltage_data = []      # Power supply voltage history
current_data = []      # Power supply current history
pid_voltage_archive = []  # Archive of PID-commanded voltages for analysis
start_time = time.time()  # Program start timestamp for relative timing

class SharedData:
    """
    Thread-safe data container for sharing sensor readings between threads.
    
    This class uses threading locks to prevent data corruption when multiple
    threads access temperature and electrical measurements simultaneously.
    
    Attributes:
        current_temp: Most recent temperature reading (°C)
        avg_thermo: Dictionary of averaged thermocouple readings
        avg_tsic: Dictionary of averaged TSic sensor readings
        avg_thermistor: Dictionary of averaged thermistor readings
        current_voltage: Most recent voltage measurement (V)
        current_current: Most recent current measurement (A)
    """
    
    def __init__(self):
        self.lock = threading.Lock()  # Prevents simultaneous access from multiple threads
        self.current_temp = None      # None indicates no data available yet
        self.avg_thermo = {}          # Averaged thermocouple data by sensor name
        self.avg_tsic = {}            # Averaged TSic data by channel number
        self.avg_thermistor = {}      # Averaged thermistor data by channel number
        self.current_voltage = None   # Latest voltage reading
        self.current_current = None   # Latest current reading
    
    def update_temperature(self, temp, avg_thermo_data, avg_tsic_data, avg_thermistor_data):
        """Update temperature data in thread-safe manner."""
        with self.lock:
            self.current_temp = temp
            self.avg_thermo = avg_thermo_data.copy()
            self.avg_tsic = avg_tsic_data.copy()
            self.avg_thermistor = avg_thermistor_data.copy()

    def update_voltage(self, voltage):
        """Update voltage measurement in thread-safe manner."""
        with self.lock:
            self.current_voltage = voltage

    def update_current(self, current):
        """Update current measurement in thread-safe manner."""
        with self.lock:
            self.current_current = current
    
    def get_temperature(self):
        """Get current temperature reading (thread-safe)."""
        with self.lock:
            return self.current_temp

    def get_avg_temperature(self):
        """Get average of all active sensor readings (thread-safe)."""
        with self.lock:
            all_temps = []
            if self.avg_tsic:
                all_temps.extend(self.avg_tsic.values())
            if self.avg_thermistor:
                all_temps.extend(self.avg_thermistor.values())
            if self.avg_thermo:
                all_temps.extend(self.avg_thermo.values())
            
            if all_temps:
                return median(all_temps)
            return None
    
    def get_all_data(self):
        """Get complete snapshot of all temperature data (thread-safe)."""
        with self.lock:
            return (self.current_temp, self.avg_thermo.copy(), 
                   self.avg_tsic.copy(), self.avg_thermistor.copy())

# Create global shared data instance
shared_data = SharedData()

# Rolling data storage for averaging - keeps recent readings for smoothing
thermo_rolling_data = {tc: [] for tc in ACTIVE_THERMOCOUPLES}
tsic_rolling_data = {ch: [] for ch in ACTIVE_TSIC_CHANNELS}
thermistor_rolling_data = {tm: [] for tm in ACTIVE_THERMISTOR_CHANNELS}

# Thread synchronization objects
data_lock = threading.Lock()  # Protects plotting data
exit_event = threading.Event()  # Signals all threads to stop

# ================================================================================================
# CSV FILE INITIALIZATION - Create data logging file with appropriate headers
# ================================================================================================

def create_csv_file():
    """Create CSV file with headers based on active sensors and configuration."""
    headers = ["Timestamp (HH-MM-SS)", "Time (s)"]
    
    # Add thermocouple columns
    headers.extend([f"{tc} Thermocouple (°C)" for tc in ACTIVE_THERMOCOUPLES])
    
    # Add TSic sensor columns
    headers.extend([f"TSic AIN{ch} (°C)" for ch in ACTIVE_TSIC_CHANNELS])
    
    # Add thermistor columns
    headers.extend([f"Thermistor AIN{tm} (°C)" for tm in ACTIVE_THERMISTOR_CHANNELS])
    
    # Add electrical measurement columns if enabled
    if read_voltage_current:
        headers.extend(["Voltage (V)", "Current (A)"])
    
    # Create file and write headers
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Initialize CSV file
create_csv_file()

# ================================================================================================
# HARDWARE CONFIGURATION FUNCTIONS - Set up LabJack channels for different sensor types
# ================================================================================================

def configure_thermocouple(handle):
    """
    Configure LabJack channels for J-type thermocouple measurements.
    
    Sets up analog input channels with appropriate range, extended features,
    and temperature compensation for accurate thermocouple readings.
    
    Args:
        handle: LabJack device handle from ljm.openS()
    """
    print("Configuring thermocouples...")
    for name, config in ACTIVE_THERMOCOUPLES.items():
        ain_channel = config["channel"]
        tc_type = config["type"]  # 21 = J-type thermocouple
        
        # Set input range (1.0V for thermocouple signals)
        ljm.eWriteName(handle, f"AIN{ain_channel}_RANGE", 1.0)
        range_readback = ljm.eReadName(handle, f"AIN{ain_channel}_RANGE")
        print(f"  AIN{ain_channel} range set to: {range_readback}V")
        
        # Configure extended feature for thermocouple processing
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_INDEX", tc_type)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_A", 1)  # Output in °C
        
        print(f"  Configured {name} thermocouple on AIN{ain_channel}")

def configure_tsic(handle):
    """
    Configure LabJack channels for TSic digital temperature sensors.
    
    TSic sensors output a PWM signal that varies with temperature.
    The LabJack reads the analog voltage level of this signal.
    
    Args:
        handle: LabJack device handle from ljm.openS()
    """
    print("Configuring TSic sensors...")
    for ain_channel in ACTIVE_TSIC_CHANNELS:
        # Set appropriate voltage range for TSic signals
        ljm.eWriteName(handle, f"AIN{ain_channel}_RANGE", 1.0)
        
        # Set high resolution for accurate PWM duty cycle measurement
        ljm.eWriteName(handle, f"AIN{ain_channel}_RESOLUTION_INDEX", 8)
        
        print(f"  Configured TSic sensor on AIN{ain_channel}")

def configure_thermistor(handle):
    """
    Configure LabJack channels for thermistor temperature measurements.
    
    Uses Steinhart-Hart equation with preconfigured coefficients for
    accurate temperature calculation from resistance measurements.
    
    Reference: https://support.labjack.com/docs/14-1-5-thermistor-t-series-datasheet
    
    Args:
        handle: LabJack device handle from ljm.openS()
    """
    print("Configuring thermistors...")
    for ain_channel in ACTIVE_THERMISTOR_CHANNELS:
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_INDEX", 50)
        ljm.eWriteName(handle, f"AIN{ain_channel}_RANGE", 10.0)
        ljm.eWriteName(handle, f"AIN{ain_channel}_RESOLUTION_INDEX", 8)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_A", 1)    # °C output
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_B", 0)    # 10µA excitation current
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_F", 10000)  # R₀ = 10kΩ at 25°C
        
        # Steinhart-Hart equation coefficients for accurate temperature calculation
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_G", 0.003354030191939)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_H", 0.000256479654956)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_I", 0.000002372509468)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_J", 0.000000089964968)
        
        print(f"  Configured thermistor on AIN{ain_channel}")

# ================================================================================================
# SENSOR READING FUNCTIONS - Acquire data from hardware
# ================================================================================================

def read_sensors():
    """
    Read temperature from all configured sensors.
    
    Connects to LabJack, reads all active sensor channels, and returns
    organized temperature data. Handles connection errors gracefully.
    
    Returns:
        tuple: (thermocouple_temps, tsic_temps, thermistor_temps)
               Each is a dictionary mapping sensor names/channels to temperatures
               Returns (None, None, None) on error
    """
    try:
        handle = ljm.openS("T7", "ANY", "ANY")  # Connect to any available LabJack T7
        
        # Read thermocouple temperatures using extended features
        thermo_temps = {}
        for name, config in ACTIVE_THERMOCOUPLES.items():
            channel = config['channel']
            # Read processed temperature from extended feature
            temp = ljm.eReadName(handle, f"AIN{channel}_EF_READ_A")
            thermo_temps[name] = temp
        
        # Read TSic sensor temperatures with voltage-to-temperature conversion
        tsic_temps = {}
        for ch in ACTIVE_TSIC_CHANNELS:
            # Read raw voltage and convert to temperature
            voltage = ljm.eReadName(handle, f"AIN{ch}")
            # TSic conversion: -10°C to +60°C spans 0V to 1V
            temperature = -10 + voltage * 70
            tsic_temps[ch] = temperature

        # Read thermistor temperatures using extended features
        thermistor_temps = {}
        for tm in ACTIVE_THERMISTOR_CHANNELS:
            # Read processed temperature from extended feature
            temp = ljm.eReadName(handle, f"AIN{tm}_EF_READ_A")
            thermistor_temps[tm] = temp
        
        ljm.close(handle)  # Clean up connection
        return thermo_temps, tsic_temps, thermistor_temps
        
    except Exception as e:
        print(f"Sensor read error: {e}")
        return None, None, None

# ================================================================================================
# DATA LOGGING THREAD - Background data acquisition and storage
# ================================================================================================

def log_data(supply):
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
    
    # Adjust averaging interval for PID control compatibility
    average_interval = log_data_average_interval
    if run_pid_control:
        average_interval = pid_interval // 2  # Use half the PID interval
        print(f"Adjusted averaging interval to {average_interval}s for PID compatibility")

    print(f"Data logging thread started with {average_interval}s averaging window")
    
    while not exit_event.is_set():
        # Read all sensors
        thermo_temps, tsic_temps, thermistor_temps = read_sensors()
        
        # Process temperature data if readings were successful
        if thermo_temps is not None or tsic_temps is not None or thermistor_temps is not None:
            with data_lock:  # Thread-safe data update
                # Update rolling averages for thermocouples
                for tc in ACTIVE_THERMOCOUPLES:
                    if thermo_temps and tc in thermo_temps:
                        thermo_rolling_data[tc].append(thermo_temps[tc])
                        # Keep only recent readings for averaging window
                        if len(thermo_rolling_data[tc]) > average_interval:
                            thermo_rolling_data[tc].pop(0)

                # Update rolling averages for TSic sensors
                for ch in ACTIVE_TSIC_CHANNELS:
                    if tsic_temps and ch in tsic_temps:
                        tsic_rolling_data[ch].append(tsic_temps[ch])
                        if len(tsic_rolling_data[ch]) > average_interval:
                            tsic_rolling_data[ch].pop(0)

                # Update rolling averages for thermistors
                for tm in ACTIVE_THERMISTOR_CHANNELS:
                    if thermistor_temps and tm in thermistor_temps:
                        thermistor_rolling_data[tm].append(thermistor_temps[tm])
                        if len(thermistor_rolling_data[tm]) > average_interval:
                            thermistor_rolling_data[tm].pop(0)
        
        # Read electrical measurements from power supply
        if read_voltage_current:
            try:
                supply_voltage = supply.get_measured_voltage()
                voltage_data.append(supply_voltage)
                shared_data.update_voltage(supply_voltage)

                supply_current = supply.get_measured_current()
                current_data.append(supply_current)
                shared_data.update_current(supply_current)
            except Exception as e:
                print(f"Power supply read error: {e}")

        time.sleep(1)  # Maintain 1Hz data acquisition rate

        # Determine if we have enough data to process and log
        should_process = len(time_data) == 0  # Always process first iteration
        
        # Check if we have any sensor data to process
        if ACTIVE_THERMOCOUPLES:
            first_tc = next(iter(ACTIVE_THERMOCOUPLES))
            should_process = should_process or len(thermo_rolling_data[first_tc]) > 0
            
        if ACTIVE_TSIC_CHANNELS:
            first_ch = ACTIVE_TSIC_CHANNELS[0]
            should_process = should_process or len(tsic_rolling_data[first_ch]) > 0

        if ACTIVE_THERMISTOR_CHANNELS:
            first_tm = ACTIVE_THERMISTOR_CHANNELS[0]
            should_process = should_process or len(thermistor_rolling_data[first_tm]) > 0
            
        # Skip processing if no active sensors
        if not ACTIVE_THERMOCOUPLES and not ACTIVE_TSIC_CHANNELS and not ACTIVE_THERMISTOR_CHANNELS:
            should_process = False

        # Process and log data
        if should_process:
            current_time = round(time.time() - start_time, 1)
            timestamp_str = datetime.datetime.now().strftime('%H-%M-%S')

            with data_lock:
                # Calculate averages and current values
                avg_thermo = {tc: np.median(thermo_rolling_data[tc]) for tc in ACTIVE_THERMOCOUPLES if thermo_rolling_data[tc]}
                avg_tsic = {ch: np.median(tsic_rolling_data[ch]) for ch in ACTIVE_TSIC_CHANNELS if tsic_rolling_data[ch]}
                avg_thermistor = {tm: np.median(thermistor_rolling_data[tm]) for tm in ACTIVE_THERMISTOR_CHANNELS if thermistor_rolling_data[tm]}
                
                current_thermo = {tc: thermo_rolling_data[tc][-1] for tc in ACTIVE_THERMOCOUPLES if thermo_rolling_data[tc]}
                current_tsic = {ch: tsic_rolling_data[ch][-1] for ch in ACTIVE_TSIC_CHANNELS if tsic_rolling_data[ch]}
                current_thermistor = {tm: thermistor_rolling_data[tm][-1] for tm in ACTIVE_THERMISTOR_CHANNELS if thermistor_rolling_data[tm]}
                
                # Update shared data for control threads
                # Determine current temperature from available sensors
                current_temp = None
                if ACTIVE_TSIC_CHANNELS and avg_tsic:
                    current_temp = avg_tsic[ACTIVE_TSIC_CHANNELS[0]]
                elif ACTIVE_THERMISTOR_CHANNELS and avg_thermistor:
                    current_temp = avg_thermistor[ACTIVE_THERMISTOR_CHANNELS[0]]
                elif ACTIVE_THERMOCOUPLES and avg_thermo:
                    current_temp = list(avg_thermo.values())[0]
                
                if current_temp is not None:
                    shared_data.update_temperature(current_temp, avg_thermo, avg_tsic, avg_thermistor)

                # Update plotting data
                time_data.append(current_time)
                for tc in ACTIVE_THERMOCOUPLES:
                    if tc in avg_thermo:
                        thermo_temp_data[tc].append(avg_thermo[tc])
                for ch in ACTIVE_TSIC_CHANNELS:
                    if ch in avg_tsic:
                        tsic_temp_data[ch].append(avg_tsic[ch])
                for tm in ACTIVE_THERMISTOR_CHANNELS:
                    if tm in avg_thermistor:
                        thermistor_temp_data[tm].append(avg_thermistor[tm])

                # Limit plotting data to last hour for performance
                if len(time_data) > 3600:
                    time_data.pop(0)
                    for tc in thermo_temp_data:
                        if thermo_temp_data[tc]:
                            thermo_temp_data[tc].pop(0)
                    for ch in tsic_temp_data:
                        if tsic_temp_data[ch]:
                            tsic_temp_data[ch].pop(0)
                    for tm in thermistor_temp_data:
                        if thermistor_temp_data[tm]:
                            thermistor_temp_data[tm].pop(0)

            # Print status to console
            status_parts = []
            if ACTIVE_THERMOCOUPLES and avg_thermo:
                tc_status = " | ".join([f"{tc}: {avg_thermo[tc]:.2f}°C" for tc in avg_thermo])
                status_parts.append(tc_status)
            if ACTIVE_TSIC_CHANNELS and avg_tsic:
                tsic_status = " | ".join([f"TSic AIN{ch}: {avg_tsic[ch]:.2f}°C" for ch in avg_tsic])
                status_parts.append(tsic_status)
            if ACTIVE_THERMISTOR_CHANNELS and avg_thermistor:
                thermistor_status = " | ".join([f"Thermistor AIN{tm}: {avg_thermistor[tm]:.2f}°C" for tm in avg_thermistor])
                status_parts.append(thermistor_status)
                
            if status_parts:
                print(f"[{timestamp_str}] " + " || ".join(status_parts))
            
            # Write to CSV file
            write_to_csv(timestamp_str, current_time, current_thermo, current_tsic, current_thermistor, avg_thermo, avg_tsic, avg_thermistor)

def write_to_csv(timestamp_str, current_time, current_thermo, current_tsic, current_thermistor, avg_thermo, avg_tsic, avg_thermistor):
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
            
            # Add thermocouple data (raw or averaged)
            if log_data_record_raw_data:
                row.extend([current_thermo.get(tc, '') for tc in ACTIVE_THERMOCOUPLES])
            else:
                row.extend([avg_thermo.get(tc, '') for tc in ACTIVE_THERMOCOUPLES])
            
            # Add TSic data (raw or averaged)
            if log_data_record_raw_data:
                row.extend([current_tsic.get(ch, '') for ch in ACTIVE_TSIC_CHANNELS])
            else:
                row.extend([avg_tsic.get(ch, '') for ch in ACTIVE_TSIC_CHANNELS])

            # Add thermistor data (raw or averaged)
            if log_data_record_raw_data:
                row.extend([current_thermistor.get(tm, '') for tm in ACTIVE_THERMISTOR_CHANNELS])
            else:
                row.extend([avg_thermistor.get(tm, '') for tm in ACTIVE_THERMISTOR_CHANNELS])
            
            # Add electrical measurements if enabled
            if read_voltage_current:
                row.extend([shared_data.current_voltage, shared_data.current_current])
            
            writer.writerow(row)
            
    except Exception as e:
        print(f"CSV write error: {e}")

# ================================================================================================
# PLOTTING FUNCTIONS - Real-time data visualization
# ================================================================================================

def update_plot(frame):
    """
    Update matplotlib plot with current data.
    
    This function is called by matplotlib's animation system to refresh
    the real-time temperature plot display.
    
    Args:
        frame: Animation frame number (unused but required by matplotlib)
    """
    with data_lock:
        ax.clear()  # Clear previous plot

        # Plot thermocouple data with solid lines
        for tc in ACTIVE_THERMOCOUPLES:
            if thermo_temp_data[tc]:  # Only plot if data exists
                ax.plot(time_data, thermo_temp_data[tc], 
                       label=f"{tc} Thermocouple (°C)", 
                       linestyle='-', linewidth=2)

        # Plot TSic sensor data with dashed lines
        for ch in ACTIVE_TSIC_CHANNELS:
            if tsic_temp_data[ch]:  # Only plot if data exists
                ax.plot(time_data, tsic_temp_data[ch], 
                       label=f"TSic AIN{ch} (°C)", 
                       linestyle='-', linewidth=2)
                
        # Plot thermistor data with solid lines
        for tm in ACTIVE_THERMISTOR_CHANNELS:
            if thermistor_temp_data[tm]:  # Only plot if data exists
                ax.plot(time_data, thermistor_temp_data[tm], 
                       label=f"Thermistor AIN{tm} (°C)", 
                       linestyle='-', linewidth=2)

    # Format plot appearance
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Live Temperature Readings")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits if data exists
    if time_data:
        ax.set_xlim(max(0, time_data[-1] - 300), time_data[-1] + 10)  # Show last 5 minutes

# ================================================================================================
# CONTROL ALGORITHMS - Temperature regulation and automation
# ================================================================================================

def voltage_lookup(input_temp):
    """
    Convert desired temperature to required voltage using lookup table.
    
    Uses empirically determined voltage-temperature relationship from
    overnight calibration data. Performs linear interpolation between
    known points for temperatures between measured values.
    
    Args:
        input_temp (float): Desired temperature in Celsius
        
    Returns:
        float: Required voltage setting for power supply
    """
    # Calibration data: voltage settings and corresponding equilibrium temperatures
    voltage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    temp_list = [19.500, 18.862, 18.413, 17.945, 17.582, 17.212, 16.853, 16.567, 16.218, 15.953, 15.732, 15.527, 15.357, 15.246, 15.185]
    
    # Find temperatures that bracket the input temperature
    below_index = 0
    temp_above = 0
    temp_below = 0
    
    for i in range(len(temp_list)):
        if temp_list[i] < input_temp:
            temp_below = temp_list[i]
            temp_above = temp_list[i-1]
            below_index = i
            break
    
    # Calculate interpolation weights
    diff_above = temp_above - input_temp
    diff_below = input_temp - temp_below
    print(f"Interpolating: above={diff_above:.3f}, below={diff_below:.3f}")

    # Linear interpolation between voltage points
    voltage_set = voltage_list[below_index] - 0.1 * (diff_below) / (diff_above + diff_below)
    return voltage_set

def slow_control(temp_step, supply):
    """
    Simple voltage sweep control thread.
    
    Performs a linear voltage ramp from 0V to 1V over 1000 seconds.
    This provides a basic temperature sweep for characterization experiments.
    
    Args:
        temp_step (float): Temperature step parameter (currently unused)
        supply: Power supply control object
    """
    print(f"Slow control thread started with temp_step={temp_step}")
    print("Beginning voltage sweep from 0V to 1V over 1000 seconds...")
    
    while not exit_event.is_set():
        for i in range(1001):
            if exit_event.is_set():
                break
                
            voltage = 0.001 * i  # 1mV increments
            print(f"Voltage set to {voltage:.3f}V")
            
            try:
                supply.set_voltage(voltage)
            except Exception as e:
                print(f"Error setting voltage: {e}")
                
            time.sleep(1)  # 1 second per step
        
        print("Voltage sweep completed")
        break

def pid_control(desired_temp, interval):
    """
    PID controller for maintaining constant temperature.
    
    Implements a Proportional-Integral-Derivative controller to maintain
    temperature at a setpoint by adjusting power supply voltage.
    
    PID Tuning Parameters (Ziegler-Nichols method):
    - Based on critical gain Ku = 1.84, critical period Tu = 58.2s
    - Kp = 0.2 * Ku = 0.368
    - Ki = 0.40 * Ku / Tu = 0.0127  
    - Kd = 0.067 * Ku * Tu = 7.17
    
    Args:
        desired_temp (float): Target temperature in Celsius
        interval (float): PID update interval in seconds
    """
    print(f"Starting PID control: target={desired_temp}°C, interval={interval}s")
    
    # Wait for initial temperature reading
    time.sleep(interval)
    
    # Initialize PID variables
    current_temp = shared_data.get_avg_temperature()
    integral = 0
    previous_error = 0
    voltage = supply.get_measured_voltage()
    
    # PID tuning parameters (Ziegler-Nichols tuned)
    k_prop = 0.368    # Proportional gain
    k_int = 0         # Integral gain (disabled to prevent windup)
    k_deriv = 7.17    # Derivative gain
    
    iteration = 0
    
    while not exit_event.is_set():
        # Get current temperature
        current_temp = shared_data.get_avg_temperature()
        if current_temp is None:
            print("Waiting for temperature data...")
            time.sleep(interval)
            continue

        # PID calculation
        error = desired_temp - current_temp
        
        # Proportional term
        P_out = k_prop * error
        
        # Integral term (accumulates error over time)
        integral += error * interval
        I_out = k_int * integral
        
        # Derivative term (rate of change of error)
        derivative = (error - previous_error) / interval
        D_out = k_deriv * derivative
        
        # Calculate voltage correction
        previous_error = error
        correction = (P_out + I_out + D_out) * -1  # Negative for cooling system
        voltage = voltage + correction
        
        # Clamp voltage to safe limits
        voltage = max(0, min(voltage, 1.5))  # 0-1.5V range because over 1.5 is inefficient
        
        # Apply voltage to power supply
        try:
            supply.set_voltage(voltage)
            pid_voltage_archive.append(voltage)
        except Exception as e:
            print(f"Error setting voltage: {e}")
            time.sleep(interval)
            continue

        # Debug output
        print(f"PID Control [Iter {iteration}]:")
        print(f"  Current: {current_temp:.2f}°C, Target: {desired_temp:.2f}°C")
        print(f"  Error: {error:.3f}°C, Voltage: {voltage:.3f}V")
        print(f"  P:{P_out:.3f} I:{I_out:.3f} D:{D_out:.3f} Correction:{correction:.3f}")

        time.sleep(interval)
        iteration += 1

def pid_slow_control(interval, supply):
    """
    PID-assisted temperature ramping between setpoints.
    
    This advanced control algorithm performs a complete temperature characterization:
    1. Uses PID control to find stable voltages for start and end temperatures
    2. Performs a controlled temperature ramp between these points
    3. Returns to starting temperature for complete cycle
    
    This is useful for thermal cycling experiments and characterization.
    
    Args:
        interval (float): PID update interval in seconds
        supply: Power supply control object
    """
    print(f"Starting PID-based slow control:")
    print(f"  Temperature range: {pid_slow_control_starting_temp}°C to {pid_slow_control_ending_temp}°C")
    print(f"  Voltage finding time: {pid_slow_control_voltage_finding_time}s per endpoint")
    print(f"  Ramp time: {pid_slow_control_swing_time}s")

    # Wait for initial temperature readings
    time.sleep(interval)

    # Initialize PID variables
    current_temp = shared_data.get_avg_temperature()
    integral = 0
    previous_error = 0
    voltage = 0.5  # Start with moderate voltage

    # PID tuning parameters (same as regular PID control)
    k_prop = 0.368    # Proportional gain
    k_int = 0         # Integral gain (disabled to prevent windup)  
    k_deriv = 7.17    # Derivative gain

    print("\n=== Phase 1: Finding voltage for ending temperature ===")
    # Phase 1: Find stable voltage for ending temperature
    for i in range(int(pid_slow_control_voltage_finding_time / interval)):
        desired_temp = pid_slow_control_ending_temp
        
        # Get current temperature
        current_temp = shared_data.get_avg_temperature()
        if current_temp is None:
            print("Waiting for temperature data...")
            time.sleep(interval)
            continue

        # PID calculation for ending temperature
        error = desired_temp - current_temp
        P_out = k_prop * error
        integral += error * interval
        I_out = k_int * integral
        derivative = (error - previous_error) / interval
        D_out = k_deriv * derivative
        
        # Apply PID correction
        previous_error = error
        correction = (P_out + I_out + D_out) * -1
        voltage = voltage + correction
        
        # Clamp voltage to safe limits
        voltage = max(0, min(voltage, 1.5))  # 0-1.5V range because over 1.5 is inefficient
        
        try:
            supply.set_voltage(voltage)
            pid_voltage_archive.append(voltage)
        except Exception as e:
            print(f"Error setting voltage: {e}")

        # Progress indication
        if i % 5 == 0:  # Print every 5 iterations
            print(f"  Finding ending voltage: {current_temp:.2f}°C → {desired_temp:.2f}°C, V={voltage:.3f}")
            print(f"    P:{P_out:.3f} I:{I_out:.3f} D:{D_out:.3f} Error:{error:.3f}")

        time.sleep(interval)

    ending_voltage = voltage
    print(f"Ending voltage found: {ending_voltage:.3f}V for {pid_slow_control_ending_temp}°C")

    print("\n=== Phase 2: Finding voltage for starting temperature ===")
    # Phase 2: Find stable voltage for starting temperature  
    integral = 0  # Reset integral term
    for i in range(int(pid_slow_control_voltage_finding_time / interval)):
        desired_temp = pid_slow_control_starting_temp

        current_temp = shared_data.get_avg_temperature()
        if current_temp is None:
            time.sleep(interval)
            continue

        # PID calculation for starting temperature
        error = desired_temp - current_temp
        P_out = k_prop * error
        integral += error * interval
        I_out = k_int * integral
        derivative = (error - previous_error) / interval
        D_out = k_deriv * derivative
        
        previous_error = error
        correction = (P_out + I_out + D_out) * -1
        voltage = voltage + correction
        
        # Clamp voltage to safe limits
        voltage = max(0, min(voltage, 1.5))  # 0-1.5V range because over 1.5 is inefficient
        pid_voltage_archive.append(voltage)
        
        try:
            supply.set_voltage(voltage)
        except Exception as e:
            print(f"Error setting voltage: {e}")
            continue

        # Progress indication
        if i % 5 == 0:
            print(f"  Finding starting voltage: {current_temp:.2f}°C → {desired_temp:.2f}°C, V={voltage:.3f}")
            print(f"    P:{P_out:.3f} I:{I_out:.3f} D:{D_out:.3f} Error:{error:.3f}")

        time.sleep(interval)

    starting_voltage = voltage
    print(f"Starting voltage found: {starting_voltage:.3f}V for {pid_slow_control_starting_temp}°C")
    print(f"Voltage range: {starting_voltage:.3f}V to {ending_voltage:.3f}V")

    # Optional settling time between phases
    if pid_slow_control_intermediate_settling_time > 0:
        print(f"\n=== Settling time: {pid_slow_control_intermediate_settling_time}s ===")
        time.sleep(pid_slow_control_intermediate_settling_time)

    print(f"\n=== Phase 3: Temperature ramp (forward) ===")
    # Phase 3: Controlled temperature ramp from start to end
    voltage_range = abs(ending_voltage - starting_voltage)
    num_steps = int(voltage_range * 1000)  # 1mV steps
    step_time = pid_slow_control_swing_time / num_steps if num_steps > 0 else 1
    
    if starting_voltage < ending_voltage:
        # Ramping voltage up (cooling down)
        print(f"Ramping voltage UP from {starting_voltage:.3f}V to {ending_voltage:.3f}V")
        for i in range(num_steps):
            if exit_event.is_set():
                break
            voltage = starting_voltage + 0.001 * i
            print(f"  Forward ramp: {voltage:.3f}V ({i+1}/{num_steps})")
            try:
                supply.set_voltage(voltage)
            except Exception as e:
                print(f"Error setting voltage: {e}")
            time.sleep(step_time)
    else:
        # Ramping voltage down (heating up)
        print(f"Ramping voltage DOWN from {starting_voltage:.3f}V to {ending_voltage:.3f}V") 
        for i in range(num_steps):
            if exit_event.is_set():
                break
            voltage = starting_voltage - 0.001 * i
            print(f"  Forward ramp: {voltage:.3f}V ({i+1}/{num_steps})")
            try:
                supply.set_voltage(voltage)
            except Exception as e:
                print(f"Error setting voltage: {e}")
            time.sleep(step_time)

    # Hold at ending temperature
    print(f"\n=== Holding at ending temperature for 300s ===")
    time.sleep(300)

    print(f"\n=== Phase 4: Temperature ramp (return) ===")
    # Phase 4: Return ramp from end back to start
    if starting_voltage < ending_voltage:
        # Ramping voltage down (heating up)
        print(f"Ramping voltage DOWN from {ending_voltage:.3f}V to {starting_voltage:.3f}V")
        for i in range(num_steps):
            if exit_event.is_set():
                break
            voltage = ending_voltage - 0.001 * i
            print(f"  Return ramp: {voltage:.3f}V ({i+1}/{num_steps})")
            try:
                supply.set_voltage(voltage)
            except Exception as e:
                print(f"Error setting voltage: {e}")
            time.sleep(step_time)
    else:
        # Ramping voltage up (cooling down)
        print(f"Ramping voltage UP from {ending_voltage:.3f}V to {starting_voltage:.3f}V")
        for i in range(num_steps):
            if exit_event.is_set():
                break
            voltage = ending_voltage + 0.001 * i
            print(f"  Return ramp: {voltage:.3f}V ({i+1}/{num_steps})")
            try:
                supply.set_voltage(voltage)
            except Exception as e:
                print(f"Error setting voltage: {e}")
            time.sleep(step_time)

    # Final hold at starting temperature
    print(f"\n=== Final hold at starting temperature for 300s ===")
    time.sleep(300)
    
    print("PID slow control sequence completed!")

# ================================================================================================
# MAIN PROGRAM - System initialization and execution
# ================================================================================================

def exit_after_timeout():
    """
    Timeout thread that terminates the program after specified duration.
    
    Provides a safety mechanism for unattended operation. Saves the current
    plot before shutdown and ensures clean program termination.
    """
    print(f"Starting {timeout_length} second timeout timer...")
    time.sleep(timeout_length)
    print(f"\nTimeout reached after {timeout_length}s. Shutting down...")

    # Save final plot with timestamp
    try:
        exit_timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        plot_filename = f'temperature_plot_{exit_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Final plot saved as: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)
    exit_event.set()
    time.sleep(1)  # Allow threads to clean up
    os._exit(0)  # Force program termination

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
        print(f"  Data logging: {'ON' if run_log_data else 'OFF'}")
        print(f"  Voltage/current monitoring: {'ON' if read_voltage_current else 'OFF'}")
        print(f"  Control mode: ", end="")
        if run_slow_control:
            print("Basic voltage sweep")
        elif run_pid_control:
            print(f"PID control (target: {pid_desired_temp}°C)")
        elif run_pid_slow_control:
            print(f"PID temperature ramp ({pid_slow_control_starting_temp}°C to {pid_slow_control_ending_temp}°C)")
        else:
            print("Monitoring only")
        print(f"  Timeout: {'ON' if run_timeout else 'OFF'} ({timeout_length}s)" if run_timeout else "  Timeout: OFF")
        print(f"  Active sensors: {len(ACTIVE_THERMOCOUPLES)} thermocouples, {len(ACTIVE_TSIC_CHANNELS)} TSic, {len(ACTIVE_THERMISTOR_CHANNELS)} thermistors")
        print("-" * 80)
        
        # Initialize LabJack connection and configure sensors
        print("Initializing LabJack connection...")
        handle = ljm.openS("T7", "ANY", "ANY")
        print(f"Connected to LabJack T7")
        
        # Configure all sensor types
        configure_thermocouple(handle)
        configure_tsic(handle) 
        configure_thermistor(handle)
        ljm.close(handle)  # Close configuration connection
        
        # Initialize power supply connection if needed
        supply = None
        if read_voltage_current or run_slow_control or run_pid_control or run_pid_slow_control:
            print("Initializing power supply connection...")
            try:
                supply = E3644A(supply_port)  # Adjust port as needed
                print("Connected to E3644A power supply")
            except Exception as e:
                print(f"Power supply connection failed: {e}")
                if run_slow_control or run_pid_control or run_pid_slow_control:
                    print("ERROR: Control modes require power supply connection!")
                    sys.exit(1)

        print(f"\nStarting system... Data logging to: {csv_filename}")
        print("Close the plot window or press Ctrl+C to stop.\n")

        # ================================================================================================
        # THREAD STARTUP - Launch background processes
        # ================================================================================================
        
        # Start data logging thread
        if run_log_data:
            log_thread = threading.Thread(target=log_data, args=[supply], daemon=True)
            log_thread.start()
            print("✓ Data logging thread started")
        
        # Start control threads (only one should be active)
        if run_slow_control:
            # Get temperature step from command line (optional)
            try:
                temp_step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
            except ValueError:
                temp_step = 0.5
                print(f"Invalid temp_step argument, using default: {temp_step}")
            
            slow_thread = threading.Thread(target=slow_control, args=[temp_step, supply], daemon=True)
            slow_thread.start()
            print(f"✓ Slow control thread started (temp_step={temp_step})")

        if run_pid_slow_control:
            pid_slow_thread = threading.Thread(target=pid_slow_control, args=[pid_interval, supply], daemon=True)
            pid_slow_thread.start()
            print("✓ PID slow control thread started")

        if run_pid_control:
            pid_thread = threading.Thread(target=pid_control, args=[pid_desired_temp, pid_interval], daemon=True)
            pid_thread.start()
            print(f"✓ PID control thread started (target={pid_desired_temp}°C)")

        # Start timeout thread if enabled
        if run_timeout:
            timeout_thread = threading.Thread(target=exit_after_timeout, daemon=True)
            timeout_thread.start()
            print(f"✓ Timeout thread started ({timeout_length}s)")

        print("\nAll threads started successfully!")
        print("-" * 80)

        # ================================================================================================
        # MAIN LOOP - Real-time plotting interface
        # ================================================================================================
        
        # Initialize matplotlib for real-time plotting
        print("Starting real-time plot display...")
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Temperature Monitoring System', fontsize=16, fontweight='bold')
        
        # Create animation for live plot updates
        ani = animation.FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
        
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
        print("Shutdown complete.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n⚠️  CRITICAL ERROR: {e}")
        print("Check hardware connections and configuration.")
        
        # Print debugging information
        print("\nDebugging information:")
        print(f"  Active thermocouples: {list(ACTIVE_THERMOCOUPLES.keys())}")
        print(f"  Active TSic channels: {ACTIVE_TSIC_CHANNELS}")
        print(f"  Active thermistor channels: {ACTIVE_THERMISTOR_CHANNELS}")
        print(f"  Power supply required: {read_voltage_current or run_slow_control or run_pid_control or run_pid_slow_control}")
        
        exit_event.set()
        sys.exit(1)

# ================================================================================================
# END OF PROGRAM
# ================================================================================================

"""
USAGE EXAMPLES:
==============

1. Basic temperature monitoring:
   python temp_control.py
   
2. Temperature monitoring with voltage sweep:
   Set run_slow_control = True
   python temp_control.py [temp_step]
   
3. PID temperature control:
   Set run_pid_control = True, pid_desired_temp = 17.5
   python temp_control.py
   
4. PID temperature ramping:
   Set run_pid_slow_control = True
   Configure pid_slow_control_* parameters
   python temp_control.py

CONFIGURATION TIPS:
==================

1. Sensor Configuration:
   - Add thermocouples to ACTIVE_THERMOCOUPLES dictionary
   - Add TSic channels to ACTIVE_TSIC_CHANNELS list
   - Add thermistor channels to ACTIVE_THERMISTOR_CHANNELS list
   - Verify AIN channel numbers match hardware connections

2. PID Tuning:
   - Start with provided Ziegler-Nichols values
   - Reduce k_prop if system oscillates
   - Enable k_int for steady-state error correction
   - Adjust k_deriv for noise vs. responsiveness trade-off

3. Safety Limits:
   - Voltage is clamped to 0-3V range
   - Temperature readings are validated before use
   - Timeout protection prevents runaway operation

4. Data Analysis:
   - CSV files contain timestamped temperature data
   - Plot files are saved on timeout or error
   - PID voltage archive available for tuning analysis

HARDWARE CONNECTIONS:
====================

LabJack T7:
- AIN0-N: TSic temperature sensors
- AIN2: J-type thermocouple (if enabled) 
- AIN3: Thermistor (if enabled)
- USB connection to computer

E3644A Power Supply:
- Serial connection via USB-to-serial adapter
- Voltage output connected to thermal control element
- Current sensing for load monitoring

TROUBLESHOOTING:
===============

1. "Sensor read error": Check LabJack connection and sensor wiring
2. "Power supply connection failed": Verify serial port and cable
3. "No temperature data": Check sensor configuration and connections  
4. PID oscillation: Reduce proportional gain or increase derivative gain
5. CSV write errors: Check file permissions and disk space

"""