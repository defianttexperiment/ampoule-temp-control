import csv
from pandas import *
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

meanlist = []
stdevlist = []
voltagelist = []

for i in range(12):
    voltage = 1 + i
    voltage = voltage/10
    voltagelist.append(voltage)
    data = read_csv(f"0529overnight_{voltage}V.csv")
    temp = data['TSic AIN0 (°C)'].tolist()
    meanlist.append(mean(temp[200:]))
    stdevlist.append(stdev(temp[200:]))
    print(f"{voltage}V Mean: {meanlist[i]}")
    print(f"{voltage}V Standard Deviation: {stdevlist[i]}")

fig, ax = plt.subplots()
ax.errorbar(voltagelist, meanlist,yerr=stdevlist)
ax.set(xlabel='Voltage (V)', ylabel='Mean temperature (°C)', title='Average Temperatures')
ax.grid()
fig.savefig("overnight_data.png")
plt.show()