#!/usr/bin/env python3
"""
E3644A Power Supply Control Script
---------------------------------
This script communicates with an Agilent/HP E3644A power supply via RS232
to control voltage output.
"""

import serial
import time
import argparse

class E3644A:
    """Controller for Agilent/HP E3644A Power Supply"""
    
    def __init__(self, port, baudrate=9600, timeout=3):
        """Initialize connection to the power supply
        
        Args:
            port (str): Serial port name (e.g., 'COM1', '/dev/ttyUSB0')
            baudrate (int): Baud rate (default: 9600)
            timeout (float): Communication timeout in seconds
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        
        # Clear any existing commands and reset
        self.reset()
        time.sleep(5)  # Allow time for the reset to complete

    def close(self):
        """Close the serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            
    def send_command(self, command):
        """Send a command to the power supply
        
        Args:
            command (str): SCPI command to send
        """
        # Add newline if not present
        if not command.endswith('\n'):
            command += '\n'
        
        self.ser.write(command.encode('ascii'))
        time.sleep(0.1)  # Brief pause to allow processing
    
    def query(self, command):
        """Send a query and return the response
        
        Args:
            command (str): SCPI query command
            
        Returns:
            str: Response from the instrument
        """
        self.send_command(command)
        time.sleep(0.1)
        response = self.ser.readline().decode('ascii', errors='ignore').strip()
        return response
    
    def identify(self):
        """Query the instrument identification
        
        Returns:
            str: Instrument identification string
        """
        return self.query("*IDN?")
    
    def reset(self):
        """Reset the instrument to default settings"""
        self.send_command("*RST")
        
    def clear(self):
        """Clear the instrument status"""
        self.send_command("*CLS")
    
    def set_voltage(self, voltage):
        """Set the output voltage
        
        Args:
            voltage (float): Desired voltage in volts (0-36V for E3644A)
        """
        # E3644A has maximum 36V in high range
        if not 0 <= voltage <= 36:
            raise ValueError(f"Voltage must be between 0 and 36V, got {voltage}V")
            
        # Set voltage - ensure we're in voltage priority mode
        self.send_command(f"VOLT {voltage}")
    
    def get_voltage_setting(self):
        """Get the voltage setting
        
        Returns:
            float: The voltage setting in volts
        """
        return float(self.query("VOLT?"))
    
    def get_measured_voltage(self):
        """Get the measured output voltage
        
        Returns:
            float: The measured output voltage in volts
        """
        return float(self.query("MEAS:VOLT?"))
    
    def set_current_limit(self, current):
        """Set the current limit
        
        Args:
            current (float): Current limit in amperes (0-5A for E3644A)
        """
        # E3644A has maximum 5A in low range
        if not 0 <= current <= 5:
            raise ValueError(f"Current must be between 0 and 5A, got {current}A")
            
        self.send_command(f"CURR {current}")

        actual_current = self.get_current_setting()
        if abs(actual_current - current) > 0.1:
            raise RuntimeError(f"Failed to set current: requested {current}A but got {actual_current}A")
    
    def get_current_setting(self):
        """Get the current limit setting
        
        Returns:
            float: The current limit in amperes
        """
        print(f"Current setting: {self.query("CURR?")}")
        return float(self.query("CURR?"))
    
    def get_measured_current(self):
        """Get the measured output current
        
        Returns:
            float: The measured output current in amperes
        """
        return float(self.query("MEAS:CURR?"))
    
    def output_on(self):
        """Turn on the power supply output"""
        self.send_command("OUTP ON")
    
    def output_off(self):
        """Turn off the power supply output"""
        self.send_command("OUTP OFF")
    
    def get_output_state(self):
        """Get the output state
        
        Returns:
            bool: True if output is on, False if off
        """
        return int(self.query("OUTP?")) == 1


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Control E3644A Power Supply')
    parser.add_argument('--port', required=True, help='Serial port (e.g., COM1, /dev/ttyUSB0)')
    parser.add_argument('--voltage', type=float, help='Set voltage (in volts)')
    parser.add_argument('--current', type=float, help='Set current limit (in amps)')
    parser.add_argument('--on', action='store_true', help='Turn output on')
    parser.add_argument('--off', action='store_true', help='Turn output off')
    parser.add_argument('--read', action='store_true', help='Read voltage and current measurements')
    
    args = parser.parse_args()
    
    try:
        supply = E3644A(args.port)
        print(f"Connected to: {supply.identify()}")
        
        if args.voltage is not None:
            supply.set_voltage(args.voltage)
            print(f"Voltage set to: {supply.get_voltage_setting()}V")
        
        if args.current is not None:
            supply.set_current_limit(args.current)
            print(f"Current limit set to: {supply.get_current_setting()}A")
        
        if args.on:
            supply.output_on()
            print("Output turned ON")
        
        if args.off:
            supply.output_off()
            print("Output turned OFF")
        
        if args.read:
            print(f"Measured voltage: {supply.get_measured_voltage()}V")
            print(f"Measured current: {supply.get_measured_current()}A")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'supply' in locals():
            supply.close()
            print("Connection closed")


if __name__ == "__main__":
    main()