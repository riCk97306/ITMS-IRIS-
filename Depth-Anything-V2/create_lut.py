import csv

# --------- CONFIGURE THESE VALUES AS YOU LIKE ----------
# Maximum distance (in meters) when depth image is very dark (0)
MAX_DISTANCE_METERS = 100.0   # e.g. 100 meters far away
# Minimum distance (in meters) when depth image is very bright (255)
MIN_DISTANCE_METERS = 1.0     # e.g. 1 meter near
# Name of the LUT file we will create
OUTPUT_LUT_FILE = "lut.csv"
# ------------------------------------------------------


def create_lut():
    """
    Creates lut.csv with 256 rows:
        intensity (0-255), distance_in_meters
    Using a simple linear mapping between MAX_DISTANCE_METERS and MIN_DISTANCE_METERS.
    """
    with open(OUTPUT_LUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        for intensity in range(256):
            # linear interpolation: 0 -> MAX_DISTANCE, 255 -> MIN_DISTANCE
            ratio = intensity / 255.0
            distance = MAX_DISTANCE_METERS - (MAX_DISTANCE_METERS - MIN_DISTANCE_METERS) * ratio
            writer.writerow([intensity, distance])

    print(f"Lookup table saved as {OUTPUT_LUT_FILE}")
    print("Example rows:")
    print("  0   ->", MAX_DISTANCE_METERS, "meters (far)")
    print(" 255  ->", MIN_DISTANCE_METERS, "meters (near)")


if __name__ == "__main__":
    create_lut()
