"""
Performs an initial call to eWriteNames to write configuration values, and then
calls eWriteNames and eReadNames repeatedly in a loop.

Relevant Documentation:
 
LJM Library:
    LJM Library Installer:
        https://labjack.com/support/software/installers/ljm
    LJM Users Guide:
        https://labjack.com/support/software/api/ljm
    Opening and Closing:
        https://labjack.com/support/software/api/ljm/function-reference/opening-and-closing
    Single Value Functions(such as eReadName):
        https://labjack.com/support/software/api/ljm/function-reference/single-value-functions
    Multiple Value Functions(such as eWriteNames):
        https://labjack.com/support/software/api/ljm/function-reference/multiple-value-functions
    Timing Functions(such as StartInterval):
        https://labjack.com/support/software/api/ljm/function-reference/timing-functions
 
T-Series and I/O:
    Modbus Map:
        https://labjack.com/support/software/api/modbus/modbus-map
    Analog Inputs:
        https://labjack.com/support/datasheets/t-series/ain
    Digital I/O:
        https://labjack.com/support/datasheets/t-series/digital-io
    DAC:
        https://labjack.com/support/datasheets/t-series/dac

"""
import sys
import time
from labjack import ljm # type: ignore

# apply conditions
# sys.argv values: times to loop, desired temp, kprop, kint, kderiv
loopMessage = ""
if len(sys.argv) > 5:
    # 5 arguments were passed.
    try:
        loopAmount = int(sys.argv[1])
        if loopAmount == -1:
            loopAmount = "infinite"
            loopMessage = " Press Ctrl+C to stop."
    except:
        raise Exception("Invalid first argument \"%s\". This specifies how many"
                        " times to loop and needs to be a number." %
                        str(sys.argv[1]))
    try:
        desiredTemp = float(sys.argv[2])
    except:
        raise Exception("Invalid second argument \"%s\". This specifies the"
                        " desired temperature and needs to be a number." %
                        str(sys.argv[2]))
    try:
        kprop = float(sys.argv[3])
        kint = float(sys.argv[4])
        kderiv = float(sys.argv[5])
    except:
        raise Exception("Invalid arguments. The last three arguments specify K values and need to be numbers.")
else:
    raise Exception("Not enough arguments passed. Arguments required: Times to loop, desired temperature, Kprop, Kint, Kderiv.")

# Open first found LabJack
handle = ljm.openS("ANY", "ANY", "ANY")
info = ljm.getHandleInfo(handle)
print("Opened a LabJack with Device type: %i, Connection type: %i,\n" \
    "Serial number: %i, IP address: %s, Port: %i,\nMax bytes per MB: %i" % \
    (info[0], info[1], info[2], ljm.numberToIP(info[3]), info[4], info[5]))
deviceType = info[0]

"""
aNames = ["AIN0_NEGATIVE_CH", "AIN0_RANGE", "AIN0_RESOLUTION_INDEX", "AIN0_SETTLING_US"]
aValues = [199, 10, 0, 0]
numFrames = len(aNames)
ljm.eWriteNames(handle, numFrames, aNames, aValues)

print("\nSet configuration:")
for i in range(numFrames):
    print("    %s : %f" % (aNames[i], aValues[i]))
"""

def main():
    print("\nStarting %s read loops.%s\n" % (str(loopAmount), loopMessage))
    i = 0
    intervalHandle = 1
    timeStep = 1000000 # timestep in microseconds
    integral = 0
    previousError = 0
    correctionRate = 0 # hyperparameter of correction
    clockRollValue = 5000 # frequency = 80 MHz / clockRollValue
    configAValue = 5000
    configAValueInitial = configAValue
    ljm.startInterval(intervalHandle, timeStep)
    ljm.eWriteName(handle, "DIO0_EF_ENABLE", 0)
    ljm.eWriteName(handle, "DIO_EF_CLOCK0_ENABLE", 0)
    ljm.eWriteName(handle, "DIO_EF_CLOCK0_ROLL_VALUE", clockRollValue)
    ljm.eWriteName(handle, "DIO0_EF_INDEX", 0)
    ljm.eWriteName(handle, "DIO0_EF_CONFIG_A", configAValue)
    ljm.eWriteName(handle, "DIO_EF_CLOCK0_ENABLE", 1)
    ljm.eWriteName(handle, "DIO0_EF_ENABLE", 1)

    # loop through and correct for error
    while True:
        try:
            # DONE read temp
            currentTemp = 0 
            tsicVoltage = 0
            tsicVoltage = ljm.eReadName(handle, "AIN0") # DONE BUT variable might give an error
            currentTemp = tsicVoltage * 70 - 10.7 # convert V to T, see tsic docs. -0.7 done using ice water calibration.
            print("Current temp: %s" % currentTemp)
            """ CODE FOR THERMOCOUPLE
            ljm.eWriteName(handle,"AIN0_EF_INDEX", 21)
            ljm.eWriteName(handle, "AIN0_EF_CONFIG_A", 1)  # °C output
            tcVoltage = 0
            tcVoltage = ljm.eReadName(handle,"AIN0_EF_READ_A")
            print("TC Voltage: %s" % tcVoltage)
            currentTemp = tcVoltage
            print("Current temp: %s" % currentTemp) """
            
            # DONE calculate PID
            error = desiredTemp - currentTemp

            P_out = kprop * error
            integral += error * intervalHandle
            I_out = kint * integral
            derivative = (error - previousError) / intervalHandle
            D_out = kderiv * derivative

            previousError = error
            print("Error: %s" % error)

            # DONE: update PWM based on PID
            configAValue = configAValue - error * correctionRate * configAValueInitial

            if configAValue > clockRollValue:
                configAValue = clockRollValue
            if configAValue < 0:
                configAValue = 0        
            ljm.eWriteName(handle, "DIO0_EF_CONFIG_A", configAValue)
            print("configAValue: %s" % configAValue)

            # DONE Repeat every 1 second
            time.sleep(1)
            """skippedIntervals = ljm.waitForNextInterval(intervalHandle)
            if skippedIntervals > 0:
                print("\nSkippedIntervals: %s" % skippedIntervals)
            i += 1
            if loopAmount != "infinite":
                if i >= loopAmount:
                    break"""
        except KeyboardInterrupt:
            break
        except Exception:
            import sys
            print(sys.exc_info()[1])
            break

    # close
    # ljm.cleanInterval(intervalHandle)
    # ljm.close(handle)
    



    """
    pseudocode:
    open labjack, get info DONE
    set ideal temperature
    set kprop, kint, kderiv
    set initial state
        DIO0_EF_ENABLE = 0 // Cannot change index if enabled
        DIO_EF_CLOCK0_ENABLE = 0 // Disable the clock for config
        DIO_EF_CLOCK0_ROLL_VALUE = 4000 // T4 and T7: 80 MHz baseline, divided by this number
        DIO0_EF_INDEX = 0 // PWM out index
        DIO0_EF_CONFIG_A = 2000 // Half of the roll value above
        DIO_EF_CLOCK0_ENABLE = 1 // Enable the clock
        DIO0_EF_ENABLE = 1 // Enable the PWM
    every 1s:
        read temperature
        calculate error
        DIO0_EF_CONFIG_A = x // based on temperature
    """
if __name__ == "__main__":
    main()