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
                 timeout: float = 1.0):
        """
        Initialize the SMC100 controller
        
        Args:
            port: Serial port name (e.g., 'COM1', '/dev/ttyUSB0')
            address: Controller address (1-31)
            baudrate: Communication baud rate (default: 57600)
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.address = address
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        Establish serial connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                rtscts=True  # Hardware flow control
            )
            
            # Clear any existing data in buffers
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            # Test connection with identification command
            response = self.get_identification()
            if response:
                print(f"Connected to: {response}")
                return True
            else:
                print("Failed to get identification response")
                return False
                
        except Exception as e:
            print(f"Connection failed: {e}")
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
        
        with self.lock:
            try:
                # Format command with address
                full_command = f"{self.address}{command}\r\n"
                
                # Send command
                self.serial_connection.write(full_command.encode())
                
                # Read response
                response = self.serial_connection.readline().decode().strip()
                
                # Check for error codes
                if response.startswith(f"{self.address}"):
                    return response[1:]  # Remove address prefix
                else:
                    print(f"Unexpected response format: {response}")
                    return None
                    
            except Exception as e:
                print(f"Communication error: {e}")
                return None
    
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
    
    def get_state(self) -> Optional[str]:
        """
        Get controller state
        
        Returns:
            State string or None if error
        """
        response = self.send_command("TS?")
        if response:
            # Parse state code
            state_codes = {
                "0A": "NOT REFERENCED from RESET",
                "0B": "NOT REFERENCED from HOMING",
                "0C": "NOT REFERENCED from CONFIGURATION",
                "0D": "NOT REFERENCED from DISABLE",
                "0E": "NOT REFERENCED from READY",
                "0F": "NOT REFERENCED from MOVING",
                "10": "NOT REFERENCED ESP stage error",
                "11": "NOT REFERENCED from JOGGING",
                "14": "CONFIGURATION",
                "1E": "HOMING",
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
            
            return state_codes.get(response, f"Unknown state: {response}")
        return None
    
    def home(self) -> bool:
        """
        Home the motor (find reference position)
        
        Returns:
            True if homing command sent successfully
        """
        response = self.send_command("OR")
        return response is not None
    
    def reset(self) -> bool:
        """Reset controller"""
        response = self.send_command("RS")
        return response is not None
    
    def get_velocity(self) -> Optional[float]:
        """Get current velocity setting"""
        response = self.send_command("VA?")
        if response:
            try:
                return float(response)
            except ValueError:
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
                return float(response)
            except ValueError:
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
        state = self.get_state()
        return state is not None and "MOVING" in state
    
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


def main():
    """Example usage of the SMC100 controller"""
    
    # Initialize controller
    controller = SMC100Controller(port='/dev/cu.PL2303G-USBtoUART1230', address=1)
    
    try:
        # Connect
        if not controller.connect():
            print("Failed to connect to SMC100")
            return
        
        # Get current state
        print(f"Current state: {controller.get_state()}")
        
        # Home the motor if not ready
        state = controller.get_state()
        if state and "NOT REFERENCED" in state:
            print("Homing motor...")
            controller.home()
            
            # Wait for homing to complete
            while controller.is_moving():
                time.sleep(0.5)
                print("Homing in progress...")
            
            print("Homing complete")
        
        # Get current position
        position = controller.get_position()
        print(f"Current position: {position} mm")
        
        # Set velocity and acceleration
        controller.set_velocity(1.0)  # 1 mm/s
        controller.set_acceleration(10.0)  # 10 mm/s²
        
        # Move to absolute position
        target_position = 5.0  # 5 mm
        print(f"Moving to {target_position} mm...")
        controller.move_absolute(target_position)
        
        # Wait for motion to complete
        if controller.wait_for_motion_complete():
            final_position = controller.get_position()
            print(f"Motion complete. Final position: {final_position} mm")
        else:
            print("Motion timeout")
        
        # Move relative
        relative_distance = -2.0  # Move back 2 mm
        print(f"Moving relative {relative_distance} mm...")
        controller.move_relative(relative_distance)
        
        if controller.wait_for_motion_complete():
            final_position = controller.get_position()
            print(f"Relative motion complete. Final position: {final_position} mm")
        
    except KeyboardInterrupt:
        print("\nStopping motion...")
        controller.stop_motion()
    
    finally:
        # Disconnect
        controller.disconnect()


if __name__ == "__main__":
    main()