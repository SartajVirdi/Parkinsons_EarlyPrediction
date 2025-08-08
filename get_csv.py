import serial
import csv
import os
import time

# Configuration
PORT = '/dev/tty.usbserial-0001'           # ✅ Set to COM3
BAUD_RATE = 115200
LOG_DURATION = 30       # ✅ Duration in seconds

# Output file path
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'imu_data.csv')

# Create directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {PORT}")
except serial.SerialException as e:
    print(f"Failed to connect to {PORT}: {e}")
    exit()

start_time = time.time()

with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp (ms)', 'Accel X', 'Accel Y', 'Accel Z'])

    try:
        while True:
            # Check for 30-second timeout
            if time.time() - start_time >= LOG_DURATION:
                print("✅ 30 seconds elapsed. Stopping data logging.")
                break

            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            parts = line.split(',')

            if len(parts) != 4:
                print(f"⚠️ Skipped malformed line: {line}")
                continue

            try:
                timestamp, ax, ay, az = map(float, parts)
                writer.writerow([int(timestamp), ax, ay, az])
                print(int(timestamp), ax, ay, az)
            except ValueError:
                print(f"⚠️ Skipped non-numeric line: {line}")
    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")
    finally:
        ser.close()
        print(f"✅ Data saved to: {OUTPUT_FILE}")
