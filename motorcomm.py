import serial
import time
import threading
from typing import Optional, Union


class SMC100Controller:
    """
    Newport SMC100 Motor Driver Controller
    
    This class provides an interface to communicate with Newport SMC100 
    motor drivers via RS232 serial communication.
    """
    
    def __init__(self, port: str, address: int = 1, baudrate: int = 57600, 
                 timeout: float = 1.0, is_actuator: bool = True):
        """
        Initialize the SMC100 controller
        
        Args:
            port: Serial port name (e.g., 'COM1', '/dev/ttyUSB0')
            address: Controller address (1-31)
            baudrate: Communication baud rate (default: 57600)
            timeout: Serial timeout in seconds
            is_actuator: True for actuators/motors, False for ESP stages
        """
        self.port = port
        self.address = address
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.lock = threading.Lock()
        self.debug = False  # Set to True for debugging
        self.max_retries = 3  # Maximum command retries
        self.is_actuator = is_actuator
        
    def connect(self) -> bool:
        """
        Establish serial connection
        
        Returns:
            True if connection successful, False otherwise
        """
        # Common baudrates for SMC100, in order of likelihood
        baudrates_to_try = [57600, 9600, 19200, 38400, 115200]
        
        for baudrate in baudrates_to_try:
            try:
                if self.debug:
                    print(f"Trying to connect at {baudrate} baud...")
                
                if self.serial_connection:
                    self.serial_connection.close()
                
                self.serial_connection = serial.Serial(
                    port=self.port,
                    baudrate=baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    rtscts=False,  # Try without hardware flow control first
                    xonxoff=False
                )
                
                # Wait for connection to stabilize
                time.sleep(0.2)
                
                # Clear any existing data in buffers
                self.serial_connection.reset_input_buffer()
                self.serial_connection.reset_output_buffer()
                time.sleep(0.1)
                
                # Test connection with identification command
                response = self.get_identification()
                if response and len(response) > 2:
                    print(f"Connected at {baudrate} baud to: {response}")
                    self.baudrate = baudrate
                    return True
                
                # If ID failed, try a simple command
                if self.debug:
                    print(f"ID command failed, trying position query...")
                
                response = self.get_position()
                if response is not None:
                    print(f"Connected at {baudrate} baud (position: {response} mm)")
                    self.baudrate = baudrate
                    return True
                
            except Exception as e:
                if self.debug:
                    print(f"Failed at {baudrate} baud: {e}")
                continue
        
        print("Failed to connect at any baudrate")
        return False

    def wait_for_motion_complete_with_monitoring(self, timeout: float = 30.0) -> bool:
        """
        Wait for motion to complete with state monitoring
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if motion completed, False if timeout or error
        """
        start_time = time.time()
        last_position = None
        position_unchanged_count = 0
        
        while time.time() - start_time < timeout:
            state_info = self.get_state()
            current_position = self.get_position()
            
            if state_info:
                # Check if we're in a moving state
                if state_info['state_code'] == '28':  # MOVING
                    if self.debug:
                        print(f"Moving... Position: {current_position} mm")
                    last_position = current_position
                    position_unchanged_count = 0
                # Check if we're in a ready state (motion complete)
                elif state_info['state_code'] in ['32', '33', '34', '35']:
                    print(f"Motion completed - State: {state_info['state']}")
                    return True
                # Check for errors or disabled states
                elif state_info['state_code'] in ['3C', '3D', '3E']:
                    print(f"Motion failed - Controller disabled: {state_info['state']}")
                    return False
                else:
                    # For other states, check if position is changing
                    if current_position == last_position:
                        position_unchanged_count += 1
                        if position_unchanged_count > 5:  # Position unchanged for 5 checks
                            print(f"Motion appears complete (position stable at {current_position} mm)")
                            return True
                    else:
                        position_unchanged_count = 0
                    last_position = current_position
            
            time.sleep(0.5)
        
        print("Motion timeout")
        return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Disconnected from SMC100")
    
    def send_command(self, command: str) -> Optional[str]:
        """
        Send command to SMC100 and return response
        
        Args:
            command: Command string without address prefix
            
        Returns:
            Response string or None if error
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("No active connection")
            return None
        
        # Retry mechanism for corrupted responses
        for attempt in range(self.max_retries):
            with self.lock:
                try:
                    # Clear buffers before sending
                    self.serial_connection.reset_input_buffer()
                    self.serial_connection.reset_output_buffer()
                    time.sleep(0.1)  # Longer delay for stability
                    
                    # Format command with address
                    full_command = f"{self.address}{command}\r\n"
                    
                    # Send command
                    self.serial_connection.write(full_command.encode('ascii'))
                    self.serial_connection.flush()  # Ensure data is sent
                    
                    # For set commands (non-query), SMC100 might not respond
                    # or might take longer to process
                    if command.endswith('?'):
                        # Query command - expect response
                        timeout_delay = 0.2
                    else:
                        # Set/action command - might not respond or take longer
                        timeout_delay = 0.5
                    
                    time.sleep(timeout_delay)
                    
                    # Read response
                    response_bytes = b''
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        if self.serial_connection.in_waiting > 0:
                            response_bytes = self.serial_connection.readline()
                            break
                        time.sleep(0.05)
                    
                    # For set commands, no response might be normal
                    if not response_bytes and not command.endswith('?'):
                        if self.debug:
                            print(f"Attempt {attempt + 1}: No response to set command '{command}' (may be normal)")
                        return ""  # Return empty string to indicate command was sent
                    
                    if not response_bytes:
                        if self.debug:
                            print(f"Attempt {attempt + 1}: No response received")
                        continue
                    
                    # Check if we got binary garbage
                    if self._is_binary_garbage(response_bytes):
                        if self.debug:
                            print(f"Attempt {attempt + 1}: Binary garbage detected: {response_bytes.hex()}")
                        time.sleep(0.2)  # Wait longer before retry
                        continue
                    
                    # Handle different possible encodings
                    try:
                        response = response_bytes.decode('ascii').strip()
                    except UnicodeDecodeError:
                        try:
                            response = response_bytes.decode('latin-1').strip()
                        except UnicodeDecodeError:
                            # If all else fails, ignore errors
                            response = response_bytes.decode('ascii', errors='ignore').strip()
                            if self.debug:
                                print(f"Warning: Had to ignore encoding errors in response")
                    
                    # Debug: print raw bytes if there are issues
                    if self.debug:
                        print(f"Attempt {attempt + 1}:")
                        print(f"  Sent: {full_command.strip()}")
                        print(f"  Received bytes: {response_bytes.hex()}")
                        print(f"  Decoded: '{response}'")
                    
                    # Validate response format
                    if self._is_valid_response(response, command):
                        # Check for address prefix and remove it
                        if response.startswith(f"{self.address}"):
                            return response[1:]  # Remove address prefix
                        elif response:
                            # Some responses might not include address prefix
                            return response
                    else:
                        if self.debug:
                            print(f"  Invalid response format")
                        continue
                        
                except Exception as e:
                    if self.debug:
                        print(f"Attempt {attempt + 1} - Communication error: {e}")
                    continue
        
        # For set commands, return empty string if no response (might be normal)
        if not command.endswith('?'):
            if self.debug:
                print(f"Set command '{command}' completed (no response received)")
            return ""
        
        print(f"Query command '{command}' failed after {self.max_retries} attempts")
        return None
    
    def _is_binary_garbage(self, data: bytes) -> bool:
        """Check if received data looks like binary garbage"""
        if len(data) == 0:
            return False
        
        # Count non-printable characters (excluding CR, LF)
        non_printable = sum(1 for b in data if b < 32 and b not in [10, 13])
        ratio = non_printable / len(data)
        
        # If more than 30% of bytes are non-printable, likely garbage
        return ratio > 0.3
    
    def _is_valid_response(self, response: str, command: str) -> bool:
        """Validate if response format makes sense for the command"""
        if not response:
            return False
        
        # Remove address prefix if present for validation
        clean_response = response
        if response.startswith(f"{self.address}"):
            clean_response = response[1:]
        
        # For query commands, expect some content
        if command.endswith('?'):
            if command == "ID?":
                # ID should contain some text
                return len(clean_response) > 2 and not all(c in '&Õr ' for c in clean_response)
            elif command == "TP?":
                # Position should start with TP or be numeric
                return clean_response.startswith("TP") or self._is_numeric_string(clean_response)
            elif command == "TS?":
                # State should be hex digits, might have TS prefix
                if clean_response.startswith("TS"):
                    state_part = clean_response[2:]
                else:
                    state_part = clean_response
                return len(state_part) >= 2 and all(c in '0123456789ABCDEF' for c in state_part[:2])
            elif command in ["VA?", "AC?"]:
                # Should start with command prefix or be numeric
                prefix = command[:2]
                return clean_response.startswith(prefix) or self._is_numeric_string(clean_response)
        
        # For set commands, often no response or simple acknowledgment
        return True
    
    def _is_numeric_string(self, s: str) -> bool:
        """Check if string represents a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def get_identification(self) -> Optional[str]:
        """Get controller identification"""
        return self.send_command("ID?")
    
    def get_position(self) -> Optional[float]:
        """
        Get current position
        
        Returns:
            Current position in mm or None if error
        """
        response = self.send_command("TP?")
        if response:
            try:
                # Handle response format "TP-X.XXXXX" or "TPX.XXXXX"
                if response.startswith("TP"):
                    position_str = response[2:]  # Remove "TP" prefix
                    return float(position_str)
                else:
                    # Fallback for direct numeric response
                    return float(response)
            except ValueError:
                print(f"Invalid position response: {response}")
                return None
        return None
    
    def move_absolute(self, position: float) -> bool:
        """
        Move to absolute position
        
        Args:
            position: Target position in mm
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"PA{position}")
        return response is not None
    
    def move_relative(self, distance: float) -> bool:
        """
        Move relative distance
        
        Args:
            distance: Distance to move in mm (positive or negative)
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"PR{distance}")
        return response is not None
    
    def stop_motion(self) -> bool:
        """Stop current motion"""
        response = self.send_command("ST")
        return response is not None
    
    def set_velocity(self, velocity: float) -> bool:
        """
        Set velocity
        
        Args:
            velocity: Velocity in mm/s
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"VA{velocity}")
        return response is not None
    
    def set_acceleration(self, acceleration: float) -> bool:
        """
        Set acceleration
        
        Args:
            acceleration: Acceleration in mm/s²
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"AC{acceleration}")
        return response is not None
    
    def get_state(self) -> Optional[dict]:
        """
        Get controller state and error information
        
        Returns:
            Dictionary with 'state', 'error_code', and 'errors' or None if error
        """
        response = self.send_command("TS?")
        if response:
            # Handle response format "TS000010" (6 chars after TS)
            state_data = response
            if response.startswith("TS"):
                state_data = response[2:]  # Remove "TS" prefix
            
            if len(state_data) < 6:
                print(f"Invalid TS response length: {response}")
                return None
            
            # Extract error code (first 4 chars) and state (last 2 chars)
            error_code = state_data[:4]
            state_code = state_data[4:6].upper()
            
            # Parse error code
            errors = self._parse_error_code(error_code)
            
            # Parse state code
            state_codes = {
                "0A": "NOT REFERENCED from reset",
                "0B": "NOT REFERENCED from HOMING",
                "0C": "NOT REFERENCED from CONFIGURATION", 
                "0D": "NOT REFERENCED from DISABLE",
                "0E": "NOT REFERENCED from READY",
                "0F": "NOT REFERENCED from MOVING",
                "10": "NOT REFERENCED ESP stage error",
                "11": "NOT REFERENCED from JOGGING",
                "14": "CONFIGURATION",
                "1E": "HOMING commanded from RS-232-C",
                "1F": "HOMING commanded by SMC-RC",
                "28": "MOVING",
                "32": "READY from HOMING",
                "33": "READY from MOVING", 
                "34": "READY from DISABLE",
                "35": "READY from JOGGING",
                "3C": "DISABLE from READY",
                "3D": "DISABLE from MOVING",
                "3E": "DISABLE from JOGGING",
                "46": "JOGGING from READY",
                "47": "JOGGING from DISABLE"
            }
            
            state_description = state_codes.get(state_code, f"Unknown state: {state_code}")
            
            return {
                'state_code': state_code,
                'state': state_description,
                'error_code': error_code,
                'errors': errors
            }
        return None
    
    def _parse_error_code(self, error_hex: str) -> list:
        """
        Parse 4-character hex error code into list of active errors
        
        Args:
            error_hex: 4-character hex string (e.g., "0000", "0013")
            
        Returns:
            List of error descriptions
        """
        try:
            error_int = int(error_hex, 16)
        except ValueError:
            return [f"Invalid error code: {error_hex}"]
        
        if error_int == 0:
            return ["No errors"]
        
        errors = []
        error_bits = {
            0: "Negative end of run",
            1: "Positive end of run", 
            2: "Peak current limit",
            3: "RMS current limit",
            4: "Short circuit detection",
            5: "Following error",
            6: "Homing time out",
            7: "Wrong ESP stage" + (" (Expected for actuators)" if self.is_actuator else ""),
            8: "DC voltage too low",
            9: "80W output power exceeded"
            # Bits 10-15 are marked as "Not used" in the manual
        }
        
        ignore_hw_limits = True  # Set to False if you actually want to see limit errors

        for bit, description in error_bits.items():
            if error_int & (1 << bit):
                # Ignore ESP stage error for actuators
                if self.is_actuator and bit == 7:
                    continue
                # Ignore hardware limits if configured to do so
                if ignore_hw_limits and bit in (0, 1):
                    continue
                errors.append(description)
        
        # If only ESP stage error was present and we're using an actuator, return no errors
        if not errors and self.is_actuator and (error_int & (1 << 7)):
            return ["No errors (ESP stage error ignored for actuator)"]
        
        return errors if errors else ["No errors"]
    
    def home_actuator(self) -> bool:
        """
        Home actuator to negative limit switch
        For actuators, homing typically moves to the negative limit
        
        Returns:
            True if homing command sent successfully
        """
        if self.is_actuator:
            # For actuators, use OR1 to home to negative limit switch
            response = self.send_command("OR")
        else:
            # For ESP stages, use OR (find home/index)
            response = self.send_command("OR")
        return response is not None
    
    def home(self) -> bool:
        """
        Home the motor (find reference position)
        Calls appropriate homing method based on actuator/stage type
        
        Returns:
            True if homing command sent successfully
        """
        return self.home_actuator()
    
    def configure_for_actuator(self) -> bool:
        """
        Configure controller for actuator/motor use (non-ESP stage)
        This sets up basic parameters for actuator operation
        
        Returns:
            True if configuration successful
        """
        print("Configuring controller for actuator/motor use...")
        
        # First, explicitly enter configuration mode
        print("Entering configuration mode...")
        if not self.enter_configuration_mode():
            print("Failed to enter configuration mode")
            return False
        
        time.sleep(1)  # Wait for mode change
        
        # Verify we're in configuration mode
        config_entered = False
        for i in range(10):  # Try for up to 10 seconds
            state_info = self.get_state()
            if state_info and state_info['state_code'] == '14':  # CONFIGURATION
                config_entered = True
                break
            time.sleep(1)
        
        if not config_entered:
            print("Failed to enter configuration mode")
            # Try alternative: reset and then enter config mode
            print("Trying reset followed by configuration mode...")
            self.reset()
            time.sleep(3)  # Wait longer after reset
            
            if not self.enter_configuration_mode():
                print("Failed to enter configuration mode after reset")
                return False
            
            time.sleep(2)
            state_info = self.get_state()
            if not state_info or state_info['state_code'] != '14':
                print("Still not in configuration mode")
                return False
        
        print("Successfully entered configuration mode")
        
        # Set basic parameters for actuator
        # These are typical values for 850B actuator - adjust as needed
        commands = [
            ("ZTO", "Disable ESP stage type checking"),
            ("SL-25.0", "Set negative software limit to -25mm"),
            ("SR25.0", "Set positive software limit to 25mm"), 
            ("VA1.0", "Set velocity to 1.0 mm/s"),
            ("AC10.0", "Set acceleration to 10.0 mm/s²"),
            ("BA1.0", "Set backlash to 1.0 mm"),
            ("JR0.1", "Set jog step size to 0.1 mm"),
        ]
        
        for command, description in commands:
            if self.debug:
                print(f"Sending: {description}")
            response = self.send_command(command)
            if response is None:
                print(f"Failed to set parameter: {description}")
                return False
            time.sleep(0.2)  # Longer delay between commands
        
        # Leave configuration mode
        print("Leaving configuration mode...")
        if not self.leave_configuration_mode():
            print("Failed to leave configuration mode")
            return False
        
        time.sleep(2)  # Wait for configuration to save
        
        # Verify we left configuration mode
        state_info = self.get_state()
        if state_info and state_info['state_code'] != '14':
            print("Successfully left configuration mode")
            print(f"Current state: {state_info['state']}")
            return True
        else:
            print("Failed to leave configuration mode")
            return False
    
    def reset(self) -> bool:
        """Reset controller"""
        response = self.send_command("RS")
        return response is not None
    
    def enter_configuration_mode(self) -> bool:
        """Enter configuration mode"""
        response = self.send_command("PW1")  # Enter configuration mode
        return response is not None
    
    def leave_configuration_mode(self) -> bool:
        """Leave configuration mode and save parameters"""
        response = self.send_command("PW0")  # Save and exit configuration mode
        return response is not None
    
    def get_velocity(self) -> Optional[float]:
        """Get current velocity setting"""
        response = self.send_command("VA?")
        if response:
            try:
                # Handle response format "VA-X.XXXXX" or "VAX.XXXXX"
                if response.startswith("VA"):
                    velocity_str = response[2:]  # Remove "VA" prefix
                    return float(velocity_str)
                else:
                    return float(response)
            except ValueError:
                print(f"Invalid velocity response: {response}")
                return None
        return None
    
    def set_velocity(self, velocity: float) -> bool:
        """
        Set velocity
        
        Args:
            velocity: Velocity in mm/s
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"VA{velocity}")
        return response is not None
    
    def get_acceleration(self) -> Optional[float]:
        """Get current acceleration setting"""
        response = self.send_command("AC?")
        if response:
            try:
                # Handle response format "AC-X.XXXXX" or "ACX.XXXXX"
                if response.startswith("AC"):
                    accel_str = response[2:]  # Remove "AC" prefix
                    return float(accel_str)
                else:
                    return float(response)
            except ValueError:
                print(f"Invalid acceleration response: {response}")
                return None
        return None
    
    def set_acceleration(self, acceleration: float) -> bool:
        """
        Set acceleration
        
        Args:
            acceleration: Acceleration in mm/s²
            
        Returns:
            True if command sent successfully
        """
        response = self.send_command(f"AC{acceleration}")
        return response is not None
    
    def is_moving(self) -> bool:
        """Check if motor is currently moving"""
        state_info = self.get_state()
        if state_info:
            return state_info['state_code'] == '28' or "MOVING" in state_info['state']
        return False
    
    def wait_for_motion_complete(self, timeout: float = 30.0) -> bool:
        """
        Wait for motion to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if motion completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_moving():
                return True
            time.sleep(0.1)
        
        return False

    def clear_errors(self) -> bool:
        """
        Clear controller errors by disabling and re-enabling
        This can help clear limit switch errors
        
        Returns:
            True if successful
        """
        print("Clearing errors...")
        
        # Disable the controller
        response = self.send_command("MM0")  # Disable
        if response is None:
            print("Failed to disable controller")
            return False
        
        time.sleep(1)
        
        # Re-enable the controller
        response = self.send_command("MM1")  # Enable
        if response is None:
            print("Failed to re-enable controller")
            return False
        
        time.sleep(1)
        
        # Check if errors are cleared
        state_info = self.get_state()
        if state_info:
            print(f"State after clearing errors: {state_info['state']}")
            if "No errors" in str(state_info['errors']):
                print("Errors cleared successfully")
                return True
        
        return False


def main():
    """Example usage of the SMC100 controller with actuator"""
    
    # Initialize controller for actuator use (850B)
    controller = SMC100Controller(port='/dev/cu.PL2303G-USBtoUART1230', address=1, is_actuator=True)
    controller.debug = True  # Enable debug output
    
    try:
        # Connect
        if not controller.connect():
            print("Failed to connect to SMC100")
            print("Common issues:")
            print("1. Check if the correct port is specified")
            print("2. Verify the controller address (default is 1)")
            print("3. Ensure proper cable connections")
            print("4. Try different baudrates if auto-detection failed")
            return
        
        resp = controller.send_command("TS?")
        print("Raw TS?:", repr(resp))
        
        # Get current state
        state_info = controller.get_state()
        if state_info:
            print(f"Current state: {state_info['state']}")
            print(f"Errors: {', '.join(state_info['errors'])}")
        
        # Check for limit switch errors and try to clear them
        if state_info and ("end of run" in str(state_info['errors']).lower()):
            print("Limit switch errors detected - attempting to clear...")
            if controller.clear_errors():
                print("Errors cleared, checking state again...")
                state_info = controller.get_state()
                if state_info:
                    print(f"New state: {state_info['state']}")
                    print(f"Errors: {', '.join(state_info['errors'])}")
        
        # Configure for actuator if needed
        if state_info and ("ESP stage error" in state_info['state'] or 
                          state_info['state_code'] == '10'):
            print("ESP stage error detected - configuring for actuator use...")
            if controller.configure_for_actuator():
                print("Actuator configuration successful")
                # Get state again after configuration
                state_info = controller.get_state()
                if state_info:
                    print(f"New state: {state_info['state']}")
                    print(f"Errors: {', '.join(state_info['errors'])}")
            else:
                print("Failed to configure actuator")
                return
        
        # Home the motor if not ready
        if state_info and "NOT REFERENCED" in state_info['state']:
            print("Homing actuator...")
            if controller.home():
                print("Homing command sent")
                
                # Wait for homing to complete
                homing_timeout = 60  # 60 seconds timeout
                start_time = time.time()
                while time.time() - start_time < homing_timeout:
                    current_state = controller.get_state()  # ONE call only
                    if current_state:
                        if current_state['state_code'] == '32':  # READY from HOMING
                            print("Homing completed successfully!")
                            break  # Exit immediately
                        # ... other state checks
                    time.sleep(2)  # Wait before next check
                
                print("Homing complete")
            else:
                print("Failed to send homing command")
                return
        
        time.sleep(10) 
        
        # Check final state before attempting moves
        print("Checking final state...")
        final_state = controller.get_state()
        if not final_state or final_state['state_code'] not in ['32', '33', '34', '35']:
            print(f"Warning: Controller not in READY state (current: {final_state['state'] if final_state else 'Unknown'})")
            print("Motion commands may not work properly")
        
        # Get current position
        position = controller.get_position()
        if position is not None:
            print(f"Current position: {position} mm")
        else:
            print("Failed to read position")
            
        # Verify controller is ready for motion
        if final_state and final_state['state_code'] in ['32', '33', '34', '35']:
            # Set velocity and acceleration
            print("Setting motion parameters...")
            if controller.set_velocity(1.0):  # 1 mm/s
                print("Velocity set to 1.0 mm/s")
            
            if controller.set_acceleration(10.0):  # 10 mm/s²
                print("Acceleration set to 10.0 mm/s²")
            
            # Move to absolute position
            target_position = 5.0  # 5 mm
            print(f"Moving to {target_position} mm...")
            if controller.move_absolute(target_position):
                print("Move command sent")
                
                # Wait for motion to complete with state monitoring
                if controller.wait_for_motion_complete_with_monitoring(timeout=30):
                    final_position = controller.get_position()
                    if final_position is not None:
                        print(f"Motion complete. Final position: {final_position} mm")
                    else:
                        print("Motion complete but failed to read final position")
                else:
                    print("Motion timeout or failed")
                    controller.stop_motion()
            else:
                print("Failed to send move command")
            
            # Move relative
            relative_distance = -2.0  # Move back 2 mm
            print(f"Moving relative {relative_distance} mm...")
            if controller.move_relative(relative_distance):
                print("Relative move command sent")
                
                if controller.wait_for_motion_complete_with_monitoring(timeout=30):
                    final_position = controller.get_position()
                    if final_position is not None:
                        print(f"Relative motion complete. Final position: {final_position} mm")
                    else:
                        print("Relative motion complete but failed to read final position")
                else:
                    print("Relative motion timeout or failed")
                    controller.stop_motion()
            else:
                print("Failed to send relative move command")
        else:
            print("Skipping motion commands - controller not ready")
        
    except KeyboardInterrupt:
        print("\nStopping motion...")
        controller.stop_motion()
    
    finally:
        # Disconnect
        controller.disconnect()


if __name__ == "__main__":
    main()