import time
import os
from rscomm import *

supply = E3644A("/dev/tty.PL2303G-USBtoUART130") # ADD PORT
print(f"Connected to: {supply.identify()}")

print("Running log_data")
os.system('python log_data.py')

print("Running PID loop in temp_sensor")
os.system('python temp_sensor.py')

print("Running log_data with cooldown")
os.system('python log_data.py')

print("Running PID loop in temp_sensor with cooldown")
os.system('python temp_sensor.py')

