import serial
import sys
import time

# ==========================================
# CONFIGURATION
# ==========================================
SERIAL_PORT = 'COM13'   # <--- CHECK YOUR PORT
BAUD_RATE = 115200
WHEEL_CIRCUMFERENCE = 3.46  # Meters per rotation

print(f"Connecting to {SERIAL_PORT}...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected! Format: Cycles | Distance | Mag (X,Y,Z) | Temp | Pressure")
    print("-" * 95)

    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    
                    # We now expect 6 values: x, y, z, count, temp, pressure
                    if len(parts) >= 6:
                        mag_x = parts[0]
                        mag_y = parts[1]
                        mag_z = parts[2]
                        count = int(parts[3])
                        temp  = float(parts[4])
                        press = float(parts[5])
                        
                        dist = count * WHEEL_CIRCUMFERENCE
                        
                        # Print formatted Dashboard
                        print(f"\rCycles:{count:<5} | Dist:{dist:>6.2f}m | Mag:{mag_x:>5},{mag_y:>5},{mag_z:>5} | Temp:{temp:>5.1f}C | Press:{press:>6.0f}Pa  ", end="")
                        
            except ValueError:
                pass 
            except UnicodeDecodeError:
                pass

        time.sleep(0.01)

except serial.SerialException:
    print(f"\nERROR: Could not open {SERIAL_PORT}.")
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    if 'ser' in locals() and ser.is_open: ser.close()