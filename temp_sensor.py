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
from pid_loop import main as main_pid

# ---------------- CONFIGURATION ----------------
# Active Thermocouples (J-type) on LabJack (AIN channels)
ACTIVE_THERMOCOUPLES = {
    # "J-2 (test)": {"channel": 0, "type": 21}
    # "J-1 (rear hole cell)": {"channel": 1, "type": 21},  
    # "J-2 (front center hole cell)": {"channel": 2, "type": 21},  
    # "J-5 (mid slot inner wall)": {"channel": 3, "type": 21},  
    # "J-4": {"channel": 4, "type": 21},  
}

# Active TSic sensors on LabJack (AIN channels)
ACTIVE_TSIC_CHANNELS = [0]  # peter: changed from 8, 10

# Generate timestamp in YYYY_MM_DD_HH_MM_SS format for filename
timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
csv_filename = f"sensor_readings_{timestamp}.csv"

# Data storage for live plotting (rolling buffer size: 70)
time_data = []
thermo_temp_data = {tc: [] for tc in ACTIVE_THERMOCOUPLES}
tsic_temp_data = {ch: [] for ch in ACTIVE_TSIC_CHANNELS}
start_time = time.time()

# Averaging storage
thermo_rolling_data = {tc: [] for tc in ACTIVE_THERMOCOUPLES}
tsic_rolling_data = {ch: [] for ch in ACTIVE_TSIC_CHANNELS}

# Lock for thread safety
data_lock = threading.Lock()
exit_event = threading.Event() 

# Create & initialize CSV file with new timestamp format
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Timestamp (HH-MM-SS)", "Time (s)"] +
        [f"{tc} Thermocouple (°C)" for tc in ACTIVE_THERMOCOUPLES] +
        [f"TSic AIN{ch} (°C)" for ch in ACTIVE_TSIC_CHANNELS]
    )

def configure_thermocouple(handle):
    """Configures thermocouples on the LabJack."""
    for name, config in ACTIVE_THERMOCOUPLES.items():
        ain_channel = config["channel"]
        tc_type = config["type"]
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_INDEX", tc_type)
        ljm.eWriteName(handle, f"AIN{ain_channel}_EF_CONFIG_A", 1)  # °C output
        print(f"Configured {name} thermocouple on AIN{ain_channel}")

def read_sensors():
    """Reads all active thermocouples and TSic sensors from LabJack."""
    try:
        handle = ljm.openS("T7", "ANY", "ANY")
        
        # Read thermocouples
        thermo_temps = {
            name: ljm.eReadName(handle, f"AIN{config['channel']}_EF_READ_A") 
            for name, config in ACTIVE_THERMOCOUPLES.items()
        }
        
        # Read TSic sensors
        tsic_temps = {
            ch: -10 + (ljm.eReadName(handle, f"AIN{ch}") * 70)  
            for ch in ACTIVE_TSIC_CHANNELS
        }
        
        # ljm.close(handle)
        return thermo_temps, tsic_temps
    except Exception as e:
        print(f"Sensor read error: {e}")
        return None, None

def log_data():
    """Background thread: Queries sensors every 1s, logs data to CSV."""
    global start_time

    while not exit_event.is_set():
        thermo_temps, tsic_temps = read_sensors()
        if thermo_temps or tsic_temps:
            with data_lock:
                for tc in ACTIVE_THERMOCOUPLES:
                    thermo_rolling_data[tc].append(thermo_temps[tc])
                    if len(thermo_rolling_data[tc]) > 5:  # Keep only last 5 readings for averaging
                        thermo_rolling_data[tc].pop(0)

                for ch in ACTIVE_TSIC_CHANNELS:
                    tsic_rolling_data[ch].append(tsic_temps[ch])
                    if len(tsic_rolling_data[ch]) > 5:
                        tsic_rolling_data[ch].pop(0)

        time.sleep(1)  # Query every second

        # Determine whether to process data
        should_process = len(time_data) == 0  # Always process first iteration
        if ACTIVE_THERMOCOUPLES:
            first_tc = next(iter(ACTIVE_THERMOCOUPLES))
            should_process = should_process or len(thermo_rolling_data[first_tc]) > 0
        if ACTIVE_TSIC_CHANNELS:
            first_ch = ACTIVE_TSIC_CHANNELS[0]
            should_process = should_process or len(tsic_rolling_data[first_ch]) > 0
        if not ACTIVE_THERMOCOUPLES and not ACTIVE_TSIC_CHANNELS:
            should_process = False

        if should_process:
            current_time = round(time.time() - start_time, 1)
            timestamp_str = datetime.datetime.now().strftime('%H-%M-%S')  # HH-MM-SS format

            with data_lock:
                time_data.append(current_time)
                avg_thermo = {tc: np.mean(thermo_rolling_data[tc]) for tc in ACTIVE_THERMOCOUPLES}
                avg_tsic = {ch: np.mean(tsic_rolling_data[ch]) for ch in ACTIVE_TSIC_CHANNELS}

                for tc in ACTIVE_THERMOCOUPLES:
                    thermo_temp_data[tc].append(avg_thermo[tc])
                for ch in ACTIVE_TSIC_CHANNELS:
                    tsic_temp_data[ch].append(avg_tsic[ch])

                # Keep only last 70 seconds of data in the plot (CSV keeps all)
                if len(time_data) > 6000:
                    time_data.pop(0)
                    for tc in thermo_temp_data:
                        thermo_temp_data[tc].pop(0)
                    for ch in tsic_temp_data:
                        tsic_temp_data[ch].pop(0)

            # Print results to terminal
            print(f"[{timestamp_str}] " + 
                  " | ".join([f"{tc}: {avg_thermo[tc]:.2f}°C" for tc in ACTIVE_THERMOCOUPLES]) +
                  " || " + 
                  " | ".join([f"TSic AIN{ch}: {avg_tsic[ch]:.2f}°C" for ch in ACTIVE_TSIC_CHANNELS]))

            # Append data to CSV (saves all data)
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [timestamp_str, current_time] + 
                    [avg_thermo[tc] for tc in ACTIVE_THERMOCOUPLES] + 
                    [avg_tsic[ch] for ch in ACTIVE_TSIC_CHANNELS]
                )

def update_plot(frame):
    """Runs in the main thread: Updates Matplotlib live plot."""
    with data_lock:
        ax.clear()

        # Plot thermocouples
        for tc in ACTIVE_THERMOCOUPLES:
            ax.plot(time_data, thermo_temp_data[tc], label=f"{tc} Thermocouple (°C)", linestyle='-')

        # Plot TSic sensors
        for ch in ACTIVE_TSIC_CHANNELS:
            ax.plot(time_data, tsic_temp_data[ch], label=f"TSic AIN{ch} (°C)", linestyle='--')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Live Temperature Readings (Thermocouples & TSic)")
    ax.legend()
    ax.grid()

if __name__ == "__main__":
    try:
        handle = ljm.openS("T7", "ANY", "ANY")
        configure_thermocouple(handle)
        # ljm.close(handle)

        print(f"Starting live temperature monitoring... Data will be saved in {csv_filename}\n")

        # Start background logging thread
        log_thread = threading.Thread(target=log_data, daemon=True)
        log_thread.start()

        pid_thread = threading.Thread(target=main_pid, daemon=True)
        pid_thread.start()

        def exit_after_timeout():
            # Sleep for 30 minutes (1800 seconds)
            print("5s sleep in temp_sensor")
            time.sleep(1800)
            print("\nReached 30-minute timeout. Shutting down...")

            # Save figure
            exittimestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            plt.savefig(f'overnight_{exittimestamp}.png')
            
            plt.close(fig)

            exit_event.set()
            # Give a moment for threads to clean up
            time.sleep(1)
            # Force exit if needed
            os._exit(0)
        
        timeout_thread = threading.Thread(target=exit_after_timeout, daemon=True)
        timeout_thread.start()

        # Run Matplotlib in main thread
        fig, ax = plt.subplots(figsize=(8, 6))
        ani = animation.FuncAnimation(fig, update_plot, interval=1000)

        plt.show()  # Starts Matplotlib GUI

    except KeyboardInterrupt:
        print("\n Exiting temperature monitoring...")
        exit_event.set()  
        time.sleep(1)  
        sys.exit(0)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        sys.exit(1)
