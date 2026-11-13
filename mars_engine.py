
from pathlib import Path
import numpy as np
import json

# ==========================================
# TERRA-X: MARS TERRAIN ANALYSIS ENGINE
# NASA MOLA MEGDR DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MARS_IMAGE = BASE_DIR / "megt90n000eb.img"
MARS_HEADER = BASE_DIR / "megt90n000eb.hdr"

# Information confirmed by the NASA ENVI header
WIDTH = 5760
HEIGHT = 2880
PIXEL_SIZE_METRES = 3704.4613374
MARS_RADIUS_METRES = 3396000.0

NODATA_VALUE = 0


def analyse_mars():

    print("TERRA-X: Loading NASA MOLA Mars elevation data...")

    if not MARS_IMAGE.exists():
        raise FileNotFoundError(
            f"Mars image not found: {MARS_IMAGE}"
        )

    if not MARS_HEADER.exists():
        raise FileNotFoundError(
            f"Mars header not found: {MARS_HEADER}"
        )

    expected_pixels = WIDTH * HEIGHT

    # NASA header specifies signed 16-bit,
    # big-endian integer data.
    raw_data = np.fromfile(
        MARS_IMAGE,
        dtype=">i2"
    )

    if raw_data.size != expected_pixels:
        raise ValueError(
            f"Unexpected Mars data size: "
            f"{raw_data.size} pixels. "
            f"Expected {expected_pixels}."
        )

    elevation = raw_data.reshape(
        HEIGHT,
        WIDTH
    ).astype(np.float32)

    # Zero is the declared missing-data value.
    valid = elevation != NODATA_VALUE

    if not np.any(valid):
        raise ValueError("No valid Mars elevation data found.")

    print("TERRA-X: Calculating elevation statistics...")

    valid_elevation = elevation[valid]

    elevation_min = float(np.min(valid_elevation))
    elevation_max = float(np.max(valid_elevation))
    elevation_mean = float(np.mean(valid_elevation))

    # ==========================================
    # SLOPE ANALYSIS
    # ==========================================

    print("TERRA-X: Calculating Mars surface slope...")

    slope_min = float("inf")
    slope_max = float("-inf")
    slope_sum = 0.0
    slope_count = 0

    # Equirectangular projection:
    # east-west ground spacing decreases with latitude.
    for row in range(1, HEIGHT - 1):

        latitude = 90.0 - (row + 0.5) * 180.0 / HEIGHT

        # Avoid unstable east-west calculations
        # very close to the poles.
        if abs(latitude) > 85:
            continue

        dx = PIXEL_SIZE_METRES * np.cos(
            np.radians(latitude)
        )

        center = elevation[row, 1:-1]

        left = elevation[row, :-2]
        right = elevation[row, 2:]

        up = elevation[row - 1, 1:-1]
        down = elevation[row + 1, 1:-1]

        valid_row = (
            valid[row, 1:-1]
            & valid[row, :-2]
            & valid[row, 2:]
            & valid[row - 1, 1:-1]
            & valid[row + 1, 1:-1]
        )

        if not np.any(valid_row):
            continue

        dz_dx = (right - left) / (2.0 * dx)
        dz_dy = (down - up) / (2.0 * PIXEL_SIZE_METRES)

        slope = np.degrees(
            np.arctan(
                np.sqrt(dz_dx ** 2 + dz_dy ** 2)
            )
        )

        slope_values = slope[valid_row]

        slope_min = min(
            slope_min,
            float(np.min(slope_values))
        )

        slope_max = max(
            slope_max,
            float(np.max(slope_values))
        )

        slope_sum += float(np.sum(slope_values))
        slope_count += int(slope_values.size)

    slope_mean = (
        slope_sum / slope_count
        if slope_count else None
    )

    # ==========================================
    # SURFACE ROUGHNESS
    # ==========================================

    print("TERRA-X: Calculating local elevation variability...")

    roughness_sum = 0.0
    roughness_min = float("inf")
    roughness_max = float("-inf")
    roughness_count = 0

    for row in range(1, HEIGHT - 1):

        # Accumulate values from each 3 x 3 window.
        window_sum = np.zeros(WIDTH - 2, dtype=np.float64)
        window_sq_sum = np.zeros(WIDTH - 2, dtype=np.float64)
        window_count = np.zeros(WIDTH - 2, dtype=np.int8)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                values = elevation[
                    row + dr,
                    1 + dc:WIDTH - 1 + dc
                ]

                good = values != NODATA_VALUE

                window_sum += np.where(good, values, 0)
                window_sq_sum += np.where(
                    good,
                    values.astype(np.float64) ** 2,
                    0
                )

                window_count += good

        center_valid = valid[row, 1:-1]

        usable = center_valid & (window_count >= 5)

        if not np.any(usable):
            continue

        means = window_sum[usable] / window_count[usable]

        variance = (
            window_sq_sum[usable] / window_count[usable]
            - means ** 2
        )

        roughness = np.sqrt(np.maximum(variance, 0))

        roughness_sum += float(np.sum(roughness))
        roughness_min = min(
            roughness_min,
            float(np.min(roughness))
        )
        roughness_max = max(
            roughness_max,
            float(np.max(roughness))
        )

        roughness_count += int(roughness.size)

    roughness_mean = (
        roughness_sum / roughness_count
        if roughness_count else None
    )

    # ==========================================
    # RESULTS
    # ==========================================

    results = {
        "planet": "Mars",
        "dataset": "NASA MOLA MEGDR",
        "data_source": "NASA Mars Global Surveyor",
        "measurement_type": "Orbital altimetry-derived elevation",

        "dimensions": {
            "width_pixels": WIDTH,
            "height_pixels": HEIGHT,
            "valid_elevation_pixels": int(np.count_nonzero(valid))
        },

        "resolution": {
            "pixel_spacing_metres": PIXEL_SIZE_METRES,
            "approximate_resolution_km": round(
                PIXEL_SIZE_METRES / 1000, 3
            )
        },

        "elevation": {
            "minimum_m": elevation_min,
            "maximum_m": elevation_max,
            "average_m": elevation_mean
        },

        "slope": {
            "minimum_degrees": slope_min if slope_count else None,
            "maximum_degrees": slope_max if slope_count else None,
            "average_degrees": slope_mean,
            "valid_pixels": slope_count,
            "polar_rows_excluded_beyond_latitude_degrees": 85
        },

        "roughness": {
            "minimum_m": roughness_min if roughness_count else None,
            "maximum_m": roughness_max if roughness_count else None,
            "average_m": roughness_mean,
            "method": "3x3 local elevation standard deviation",
            "valid_pixels": roughness_count
        },

        "limitations": [
            "Global gridded elevation data, not rover-scale terrain data.",
            "Approximately 3.7 km pixel spacing.",
            "Slope is an approximate calculation.",
            "Roughness represents local elevation variability, not complete surface texture.",
            "Polar rows beyond 85 degrees latitude are excluded from slope statistics."
        ]
    }

    print("\n========== TERRA-X MARS RESULTS ==========")

    print(json.dumps(results, indent=4))

    output_file = BASE_DIR / "mars_analysis.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nTERRA-X: Mars analysis completed!")
    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    analyse_mars()

