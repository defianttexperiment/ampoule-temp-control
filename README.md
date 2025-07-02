# ampoule-temp-control

## temp_sensor.py
The main function in this repo. This reads the temperature of the system and controls the voltage to affect temperature as necessary.

Forked from Barkotel's temp_sensor function.

### Threads in temp_sensor.py
log_data: Logs temperature data in a CSV and plots it on the screen.
slow_control: Sweeps across a certain voltage range. Used and edited for overnight runs.
pid: Formerly used to call pid_loop.py.
timeout: Used to stop the program after a certain amount of time.

## pico_data.py
The main function in this repo for data analysis. Compresses and smooths CSV data from the PicoScope, detects fringes in the data, and plots fringe count with respect to temperature.

## rscomm.py
Used to communicate with the power supply. Mostly used to import functions; generally does not need to be called on its own.

## overnight.py
NOW OUTDATED. Use temp_sensor with run_slow_control = True. 

Used to collect overnight data. Can be called without parameters. Tries a certain set of voltages at even steps and records data for a certain time interval (1800s = 30m by default).

## pid_loop.py
NOW OUTDATED. Old code used to keep temperature stable.