# ampoule-temp-control

## temp_sensor.py
The main function in this repo. This calls pid_loop.py to regulate the temperature of the system, if applicable, and uses its own code to generate a graph of the temperature with time.

To use this function, you might type into the terminal "python temp_sensor.py -1 18 1 1 1". The first parameter is the number of steps and should generally be set to -1 to continue to infinity. The other parameters only matter if a PID system is used; the second parameter (18) is the desired temperature and the last parameters are hyperparameters used for tuning the PID system.

Forked from Barkotel's temp_sensor function.

## pico_data.py
IN PROGRESS. Used to smooth CSV data from the PicoScope. Outputs the timestamps of fringe maxima and minima.

## overnight.py
Used to collect overnight data. Can be called without parameters. Tries a certain set of voltages at even steps and records data for a certain time interval (1800s = 30m by default).

## rscomm.py
Used by overnight.py to communicate with the power supply. Mostly used to import functions; generally does not need to be called on its own.

## data_process.py
Used to interpret data from overnight.py. Outputs the equilibrium temperature and standard deviation of each CSV from an overnight run.