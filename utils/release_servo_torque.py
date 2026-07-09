import serial
import time


# Set serial port parameters
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
TIMEOUT = 0.01

# Arm exoskeleton servo ids
SERVO_IDS = range(7)


def send_command(ser, cmd):
    ser.write(cmd.encode("ascii"))
    time.sleep(0.01)
    response = ser.read_all()
    return response.decode("ascii", errors="ignore")


def release_servo_torque(ser, servo_id):
    cmd = f"#{servo_id:03d}PULK!"
    return send_command(ser, cmd)


def main():
    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=TIMEOUT) as ser:
        print("Serial port opened")

        for servo_id in SERVO_IDS:
            response = release_servo_torque(ser, servo_id)
            print(f"Servo {servo_id} torque released: {response.strip()}")


if __name__ == "__main__":
    main()
