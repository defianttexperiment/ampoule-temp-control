import time
import os
from rscomm import *

supply = E3644A("/dev/tty.PL2303G-USBtoUART130") # ADD PORT
print(f"Connected to: {supply.identify()}")
supply.reset()
supply.clear()
supply.output_on()
supply.set_current_limit(2)
for i in range(19):
    voltage = 0.2 + 0.1*i
    print(f"Starting run with voltage {voltage}")
    supply.set_voltage(voltage)
    time.sleep(1)
    print(f"New voltage: {supply.get_measured_voltage()}")
    print("Running temp_sensor")
    os.system('python temp_sensor.py -1 15 1 1 1')
    time.sleep(5)
supply.output_off()
supply.output_off()
supply.output_off()


    


