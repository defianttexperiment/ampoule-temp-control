# ampoule-temp-control

## Setting up PicoScope

## temp_sensor.py
The main function in this repo. This does three things:

1. Measures temperature using TSic or thermocouple sensors (based on the list of sensors given in the code) and outputs it to a live-updating graph. The graph smooths data Current default is to use a single TSic sensor through the LabJack AIN0 port. 
2. Outputs temperature data in a sensor_readings file. Note that this data is raw temperature measurements, while the graph represents smoothed data over log_data_average_interval seconds.
3. Controls the voltage of the power supply to control the temperature. See options below for how to do this.

### Options in temp_sensor.py
slow_control: 
pid_control:
pid_slow_control: 

(In progress)

## log_data.py
A fork of temp_sensor.py that has all control options turned off. Used as an option to quickly log data when I don't want to change the variables in temp_sensor.

## pico_data.py
The main function in this repo for data analysis. Compresses and smooths CSV data from the PicoScope, detects fringes in the data, and plots fringe count with respect to temperature.

### Data format for usage in pico_data
(In progress)

## rscomm.py
Used to communicate with the power supply. Mostly used to import functions; generally does not need to be called on its own.

## motorcomm.py
Used to communicate with the Z-axis motor. Used on its own, but currently a work in progress.