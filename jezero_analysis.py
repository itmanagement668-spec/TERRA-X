
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MARS_FILE = BASE_DIR / "megt90n000eb.img"
OUTPUT_FILE = BASE_DIR / "jezero_analysis.json"

# NASA MOLA MEGDR grid information
ROWS = 2880
COLS = 5760
PIXELS_PER_DEGREE = 16
MARS_RADIUS_KM = 3396.0
PIXEL_SIZE_METRES = 3704.4613374

# Approximate centre of Jezero Crater
JEZERO_LAT = 18.4
JEZERO_LON = 77.5

# Analyse a square region approximately 100 km across
REGION_WIDTH_KM = 100.0


def calculate_statistics(values):
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return {
        "minimum": round(float(np.min(values)), 2),
        "maximum": round(float(np.max(values)), 2),
        "average": round(float(np.mean(values)), 2),
    }


def analyse_jezero():
    if not MARS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {MARS_FILE.name} in the TERRA-X folder."
        )

    print("Loading NASA MOLA data...")

    mars = np.fromfile(MARS_FILE, dtype=">i2")

    if mars.size != ROWS * COLS:
        raise ValueError(
            f"Unexpected data size: found {mars.size} pixels, "
            f"expected {ROWS * COLS}."
        )

    mars = mars.reshape((ROWS, COLS)).astype(np.float32)

    # MOLA uses zero as its data-ignore value.
    mars[mars == 0] = np.nan

    # Convert the requested physical width into latitude/longitude bounds.
    half_width_km = REGION_WIDTH_KM / 2

    half_latitude = np.degrees(
        half_width_km / MARS_RADIUS_KM
    )

    half_longitude = half_latitude / np.cos(
        np.radians(JEZERO_LAT)
    )

    # MOLA ENVI header: central meridian 180 degrees.
    # Pixel centres are indexed from zero.
    row_centre = 1440 - JEZERO_LAT * PIXELS_PER_DEGREE
    col_centre = 2880 + (JEZERO_LON - 180) * PIXELS_PER_DEGREE

    row_start = max(
        0,
        int(np.floor(row_centre - half_latitude * PIXELS_PER_DEGREE))
    )

    row_end = min(
        ROWS,
        int(np.ceil(row_centre + half_latitude * PIXELS_PER_DEGREE)) + 1
    )

    col_start = max(
        0,
        int(np.floor(col_centre - half_longitude * PIXELS_PER_DEGREE))
    )

    col_end = min(
        COLS,
        int(np.ceil(col_centre + half_longitude * PIXELS_PER_DEGREE)) + 1
    )

    region = mars[row_start:row_end, col_start:col_end].copy()

    if np.count_nonzero(np.isfinite(region)) < 10:
        raise ValueError(
            "Too few valid elevation pixels in the selected region."
        )

    print("Calculating Jezero terrain measurements...")

    elevation_stats = calculate_statistics(region)

    # Approximate physical pixel spacing at Jezero's latitude.
    latitude_at_region = JEZERO_LAT

    dx = PIXEL_SIZE_METRES * np.cos(
        np.radians(latitude_at_region)
    )
    dy = PIXEL_SIZE_METRES

    # Approximate slope from elevation gradients.
    gradient_y, gradient_x = np.gradient(region, dy, dx)

    slope = np.degrees(
        np.arctan(
            np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        )
    )

    slope_stats = calculate_statistics(slope)

    # Local elevation variability using a 3x3 neighbourhood.
    roughness = np.full(region.shape, np.nan, dtype=np.float32)

    for r in range(1, region.shape[0] - 1):
        for c in range(1, region.shape[1] - 1):
            window = region[r - 1:r + 2, c - 1:c + 2]
            valid = window[np.isfinite(window)]

            if valid.size >= 5:
                roughness[r, c] = np.std(valid)

    roughness_stats = calculate_statistics(roughness)

    result = {
        "planet": "Mars",
        "region": "Jezero Crater surroundings",
        "dataset": "NASA MOLA MEGDR",
        "data_source": "NASA Mars Global Surveyor",
        "centre": {
            "latitude_degrees": JEZERO_LAT,
            "longitude_east_degrees": JEZERO_LON
        },
        "region_width_approximate_km": REGION_WIDTH_KM,
        "grid": {
            "rows": int(region.shape[0]),
            "columns": int(region.shape[1]),
            "pixel_spacing_at_equator_m": PIXEL_SIZE_METRES,
            "approximate_pixel_spacing_at_region_m": round(dx, 2),
            "valid_elevation_pixels": int(
                np.count_nonzero(np.isfinite(region))
            )
        },
        "elevation_m": elevation_stats,
        "slope_degrees": slope_stats,
        "local_elevation_variability_m": roughness_stats,
        "limitations": [
            "MOLA data has approximately 3.7 km pixel spacing at the equator.",
            "This is a regional subset of a global gridded elevation model.",
            "Slope is an approximate calculation.",
            "Local elevation variability is not complete surface roughness.",
            "This dataset cannot establish rover-scale safety or mobility."
        ]
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\n========== JEZERO ANALYSIS ==========")
    print(json.dumps(result, indent=4))
    print("\nSaved to:", OUTPUT_FILE.name)
    print("=====================================")


if __name__ == "__main__":
    analyse_jezero()
