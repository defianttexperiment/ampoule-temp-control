# ampoule-temp-control

# Physical setup

## Setting up data collection
Temperature measurement: The TSic sensor has three wires: a ground, a voltage source, and a signal wire. These should go into the LabJack in GND, VS, and an AIN port (recommended AIN0 but doesn't really matter) respectively. Once that's done, connect the USB-C attachment to your computer to connect it to the LabJack and run "python temp_sensor.py" or "python log_data.py".

PicoScope measurement: Download PicoScope 7 and connect the USB-C attachment to connect to the PicoScope 2000. This should automatically be set up for collecting data. Channel A refers to data collected for fringe counting; Channel B refers to data collected for scattering measurement. On the left, you can change the y-axis for data collection; both channels should be set to 50 or 100 mV to view data more easily. At the top, you can change the x-axis scale; this should be set to 10 or 20 s/div to view data more easily.

Voltage control: To connect to the power supply, we use a PL2303 USB-to-serial connection. This requires a driver installed on your computer. On MacOS, you can download "PL2303 Serial" from the App Store, then go to Login Items & Extensions -> Driver Extensions and allow both of them to run, following this guide: https://kb.plugable.com/serial-adapter/how-to-install-prolific-serial-port-drivers-on-macos. I have no clue how it works on Windows. 

# Code

## temp_sensor.py
The main function in this repo. This does three things:

1. Measures temperature using TSic or thermocouple sensors (based on the list of sensors given in the code) and outputs it to a live-updating graph. The graph smooths data Current default is to use a single TSic sensor through the LabJack AIN0 port. 
2. Outputs temperature data in a sensor_readings file. Note that this data is raw temperature measurements, while the graph represents smoothed data over log_data_average_interval seconds.
3. (Optional) Controls the voltage of the power supply to control the temperature. See options below for how to do this.

### Options in temp_sensor.py
slow_control: Updates the voltage over time, typically from one value to another (e.g. a gradual transition from 0.4-1.4) based on whatever code is in the slow_control() function at the time. 
pid_control: Uses a proportional-integral-derivative loop for maintaining the temperature of the system at pid_desired_temp. Updates at an interval given by pid_interval; recommended value is 15 seconds.
pid_slow_control: Updates the voltage slowly over time to go from one temperature to another. First finds the corresponding voltages to each temperature using the PID controller, then goes from one to another over the given time period.

Feel free to overwrite the code for slow_control and pid_slow_control, although if it's significantly different than the current version please save the current version in a comment block.

Only one of these programs can run at a time. To run a given program, label it as True at the start of the code and label the others as False. If all are labeled false, the program will only log data.

## log_data.py
A fork of temp_sensor.py that has all control options turned off. Used as an option to quickly log data when I don't want to change the variables in temp_sensor.

## pico_data.py
The main function in this repo for data analysis. Compresses and smooths CSV data from the PicoScope, detects fringes in the data, and plots fringe count with respect to temperature.

### Data format for usage in pico_data
First, choose a name for a single run/experiment. This should have the date and some information about the run, e.g. 0731_slow_rise for a slow rise in temperature or 0716_swings_random for random swings in temperature. 

Temperature data will be saved as a CSV file with a name like sensor_readings_[date].csv. Rename this file to {your_chosen_name}Tdata.csv and leave it where it is.

PicoScope data will be saved as a folder of CSV files in another folder called Waveforms somewhere on your computer. Move the folder for this run to the ampoule-temp-control folder and rename it to run_name. (Waveforms can stay where it is, PicoScope likes having it there.) pico_data will recognize this folder and create {your_chosen_name}data.csv and {your_chosen_name}smdata, two versions of the PicoScope data for future use. After these files are created, you can delete the original folder with the data.

### data_archive folder
Has records of all data taken from summer 2025. Put any new data from pico_data.py in here. If you need to analyze data from this folder, take it out of data_archive before trying to use it with pico_data.py.

## pico_data_multi
A fork of pico_data that takes in multiple data sets and allows you to switch between them. Rarely necessary but sometimes useful when comparing between data sets.

## rscomm.py
Used to communicate with the power supply. Mostly used to import functions; generally does not need to be called on its own.

## motorcomm.py
Used to communicate with the Z-axis motor. Used on its own, but currently a work in progress.

# Installation Instructions

1. conda environment -> 

```
conda env create -f environment_new.yml
```
then
```
conda activate ampoule-temp-control
```

2. Install the labjack drivers from here: appropriate for your system: https://files.labjack.com/installers/LJM/

3. Install appropriate Picoscope software from here: https://www.picotech.com/downloads 

4. Check the serial port where the USB hub is on your Mac via: `ls /dev/tty*` while you are connected to the USB hub

5. Replace all instances of the old USB serial port in `log_data.py` and `temp_sensor.py`

GLHF :)
