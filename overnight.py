import time
import os
from rscomm import *

supply = E3644A("/dev/tty.PL2303G-USBtoUART130") # ADD PORT
print(f"Connected to: {supply.identify()}")
supply.reset()
supply.clear()
supply.output_on()
supply.set_voltage(0.5)
print("Running temp_sensor")
os.system('python temp_sensor.py -1 15 1 1 1')
for i in range(1001):
    voltage = 0.5 + 0.001*i
    print(f"Starting run with voltage {voltage}")
    supply.set_voltage(voltage)
    time.sleep(2)
    print(f"New voltage: {supply.get_measured_voltage()}")
    time.sleep(10)
supply.output_off()
supply.output_off()
supply.output_off()


    


