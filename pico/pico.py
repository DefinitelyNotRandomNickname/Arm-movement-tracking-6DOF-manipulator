import machine
import select
import sys
import utime


def calculate_crc16(data):
    crc = 0xFFFF

    for value in data:
        # Convert each value to bytes with big-endian byte order
        bytes_value = value.to_bytes(2, 'big')

        for byte in bytes_value:
            # XOR'ing, Shifting, yaddi-yaddi-yadda
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1

    fin = crc & 0xFFFF
    return "{:04X}".format(fin)


# Define GPIO Pins for servos
SERVO_PINS = [0, 1, 7, 3, 5, 6]

# Create a list of initialized servos
servos = [machine.PWM(machine.Pin(pin)) for pin in SERVO_PINS]

# Set frequency for each of these bad boys
for servo in servos:
    servo.freq(50)

servo_positions_start = [5350, 7000, 5200, 6500, 4500, 8200] # Initial position
servo_positions_end = servo_positions_start                  # Final position
servo_positions_present = servo_positions_start              # Present position
servo_positions_iterations = [0, 0, 0, 0, 0, 0]              # Next step position
steps = 40

utime.sleep(2)

# Get servos to starting position
for i in range(len(servos)):
    print(f"Moving servo {i} to position {servo_positions_start[i]}")
    servos[i].duty_u16(servo_positions_start[i])
    utime.sleep(0.02)

utime.sleep(2)

# Set up the polling object
poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

print("Gettin input")

while True:
    # Wait for input on stdin, with waiting time of 100ms
    poll_results = poll_obj.poll(100)
    
    if poll_results:
        # Read the data from stdin
        frame = sys.stdin.readline().strip()
        
        poll_results = poll_obj.poll(1)
        
        # Loop to get the newest data and clear the old ones
        while poll_results:
            sys.stdin.readline()
            poll_results = poll_obj.poll(1)
                
        # Split the data into individual values
        data = frame.split(' ')
        
        try:
            # Check if frame has starting and ending sign
            if data[0] == 'S' and data[8] == 'E':
                # Get body of the frame
                servo_positions_end = [int(item) for item in data[2:8]]
                
                # Calculate it's CRC-16
                checksum = calculate_crc16(servo_positions_end)
                
                # Check if checksums are equal
                if checksum == data[1]:
                    # If there are 6 values move servos
                    if len(servo_positions_end) == 6:
                        # Calculate steps for each servo
                        for i in range(len(servos)):
                            servo_positions_iterations[i] = int((servo_positions_end[i] - servo_positions_start[i]) / steps)

                        # Iterate over every step
                        for _ in range(steps):
                            # Iterate over every servo
                            for i in range(len(servos)):
                                servo_positions_present[i] += servo_positions_iterations[i]
                                print(f"Moving servo {i} to position {servo_positions_present[i]}")
                                servos[i].duty_u16(servo_positions_present[i])
                            utime.sleep(0.03)
        except ValueError:         
            print(ValueError)