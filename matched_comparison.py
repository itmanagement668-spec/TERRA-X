
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


BASE_DIR = Path(__file__).resolve().parent

EARTH_FILE = BASE_DIR / "S24W070.hgt"
MARS_FILE = BASE_DIR / "megt90n000eb.img"
OUTPUT_FILE = BASE_DIR / "matched_comparison.json"

# Approximately 30 x 30 cells across the Earth region.
EARTH_GRID_SHAPE = (30, 30)

# Jezero approximate centre.
JEZERO_LATITUDE = 18.4
JEZERO_LONGITUDE = 77.5

MARS_RADIUS_KM = 3396.0
MARS_PIXELS_PER_DEGREE = 16
MARS_PIXEL_EQUATOR_M = 3704.4613374


def calculate_metrics(elevation, pixel_x, pixel_y):

    elevation = np.asarray(elevation, dtype=np.float32)

    valid = elevation[np.isfinite(elevation)]

    if valid.size == 0:
        raise ValueError("No valid elevation measurements.")

    height, width = elevation.shape

    if height < 3 or width < 3:
        raise ValueError("Grid is too small for terrain analysis.")

    slope = np.full((height - 2, width - 2), np.nan, dtype=np.float32)
    roughness = np.full((height - 2, width - 2), np.nan, dtype=np.float32)

    for row in range(1, height - 1):

        top = elevation[row - 1, 1:-1]
        bottom = elevation[row + 1, 1:-1]

        left = elevation[row, :-2]
        right = elevation[row, 2:]

        valid_slope = (
            np.isfinite(top)
            & np.isfinite(bottom)
            & np.isfinite(left)
            & np.isfinite(right)
        )

        dz_dx = (right - left) / (2 * pixel_x)
        dz_dy = (bottom - top) / (2 * pixel_y)

        gradient = np.sqrt(dz_dx**2 + dz_dy**2)

        row_slope = np.degrees(np.arctan(gradient))

        slope[row - 1, valid_slope] = row_slope[valid_slope]

        neighbourhood = np.stack([
            elevation[row - 1, :-2],
            elevation[row - 1, 1:-1],
            elevation[row - 1, 2:],

            elevation[row, :-2],
            elevation[row, 1:-1],
            elevation[row, 2:],

            elevation[row + 1, :-2],
            elevation[row + 1, 1:-1],
            elevation[row + 1, 2:]
        ])

        valid_roughness = np.all(
            np.isfinite(neighbourhood),
            axis=0
        )

        row_roughness = np.full(width - 2, np.nan, dtype=np.float32)

        row_roughness[valid_roughness] = np.std(
            neighbourhood[:, valid_roughness],
            axis=0
        )

        roughness[row - 1] = row_roughness

    valid_slope = slope[np.isfinite(slope)]
    valid_roughness = roughness[np.isfinite(roughness)]

    if valid_slope.size == 0 or valid_roughness.size == 0:
        raise ValueError("Unable to calculate terrain measurements.")

    return {
        "grid_rows": int(height),
        "grid_columns": int(width),

        "valid_elevation_cells": int(valid.size),

        "elevation_m": {
            "minimum": round(float(np.min(valid)), 2),
            "maximum": round(float(np.max(valid)), 2),
            "average": round(float(np.mean(valid)), 2)
        },

        "slope_degrees": {
            "minimum": round(float(np.min(valid_slope)), 2),
            "maximum": round(float(np.max(valid_slope)), 2),
            "average": round(float(np.mean(valid_slope)), 2)
        },

        "local_elevation_variability_m": {
            "minimum": round(float(np.min(valid_roughness)), 2),
            "maximum": round(float(np.max(valid_roughness)), 2),
            "average": round(float(np.mean(valid_roughness)), 2)
        },

        "valid_slope_cells": int(valid_slope.size),
        "valid_roughness_cells": int(valid_roughness.size)
    }


def analyse_atacama():

    print("TERRA-X: Preparing resolution-matched Earth data...")

    with rasterio.open(EARTH_FILE) as dataset:

        elevation = dataset.read(
            1,
            out_shape=EARTH_GRID_SHAPE,
            masked=True,
            resampling=Resampling.average
        )

        elevation_array = elevation.astype(np.float32).filled(np.nan)

        latitude = (dataset.bounds.bottom + dataset.bounds.top) / 2
        latitude_rad = math.radians(latitude)

        metres_per_degree_lat = (
            111132.92
            - 559.82 * math.cos(2 * latitude_rad)
            + 1.175 * math.cos(4 * latitude_rad)
        )

        metres_per_degree_lon = (
            111412.84 * math.cos(latitude_rad)
            - 93.5 * math.cos(3 * latitude_rad)
        )

        pixel_x = (
            abs(dataset.transform.a)
            * metres_per_degree_lon
            * dataset.width
            / EARTH_GRID_SHAPE[1]
        )

        pixel_y = (
            abs(dataset.transform.e)
            * metres_per_degree_lat
            * dataset.height
            / EARTH_GRID_SHAPE[0]
        )

        results = calculate_metrics(
            elevation_array,
            pixel_x,
            pixel_y
        )

        results["region"] = "Atacama Desert, Chile"
        results["planet"] = "Earth"
        results["dataset"] = "NASA SRTM 1 Arc-Second Global"
        results["source_file"] = EARTH_FILE.name
        results["pixel_spacing_approx_m"] = {
            "east_west": round(pixel_x, 2),
            "north_south": round(pixel_y, 2)
        }

        return results


def analyse_jezero():

    print("TERRA-X: Preparing resolution-matched Mars data...")

    expected_pixels = 5760 * 2880

    if not MARS_FILE.exists():
        raise FileNotFoundError(f"Missing Mars dataset: {MARS_FILE}")

    mars_data = np.fromfile(MARS_FILE, dtype=">i2")

    if mars_data.size != expected_pixels:
        raise ValueError(
            f"Unexpected MOLA file size. "
            f"Expected {expected_pixels} pixels, found {mars_data.size}."
        )

    mars_data = mars_data.reshape((2880, 5760)).astype(np.float32)

    # The NASA header identifies zero as the ignore value.
    mars_data[mars_data == 0] = np.nan

    latitude = JEZERO_LATITUDE
    longitude = JEZERO_LONGITUDE

    half_latitude_degrees = math.degrees(50 / MARS_RADIUS_KM)

    half_longitude_degrees = (
        half_latitude_degrees / math.cos(math.radians(latitude))
    )

    row_centre = round(
        1440 - latitude * MARS_PIXELS_PER_DEGREE
    )

    column_centre = round(
        2880 + (longitude - 180) * MARS_PIXELS_PER_DEGREE
    )

    half_rows = math.ceil(
        half_latitude_degrees * MARS_PIXELS_PER_DEGREE
    )

    half_columns = math.ceil(
        half_longitude_degrees * MARS_PIXELS_PER_DEGREE
    )

    row_start = row_centre - half_rows
    row_end = row_centre + half_rows + 1

    column_start = column_centre - half_columns
    column_end = column_centre + half_columns + 1

    region = mars_data[
        row_start:row_end,
        column_start:column_end
    ]

    if region.size == 0:
        raise ValueError("Jezero region could not be extracted.")

    pixel_y = MARS_PIXEL_EQUATOR_M

    pixel_x = (
        MARS_PIXEL_EQUATOR_M
        * math.cos(math.radians(latitude))
    )

    results = calculate_metrics(region, pixel_x, pixel_y)

    results["region"] = "Jezero Crater surroundings"
    results["planet"] = "Mars"
    results["dataset"] = "NASA MOLA MEGDR"
    results["source_file"] = MARS_FILE.name

    results["centre"] = {
        "latitude_degrees": JEZERO_LATITUDE,
        "longitude_east_degrees": JEZERO_LONGITUDE
    }

    results["pixel_spacing_approx_m"] = {
        "east_west": round(pixel_x, 2),
        "north_south": round(pixel_y, 2)
    }

    return results


def main():

    earth_results = analyse_atacama()
    mars_results = analyse_jezero()

    earth_x = earth_results["pixel_spacing_approx_m"]["east_west"]
    earth_y = earth_results["pixel_spacing_approx_m"]["north_south"]

    mars_x = mars_results["pixel_spacing_approx_m"]["east_west"]
    mars_y = mars_results["pixel_spacing_approx_m"]["north_south"]

    output = {
        "project": "TERRA-X",

        "analysis": "Approximate spatial-resolution-matched terrain comparison",

        "comparison_method": {
            "earth_method": "SRTM elevation averaged onto a 30 x 30 grid",
            "mars_method": "MOLA elevation extracted around Jezero",
            "terrain_metrics": [
                "Elevation",
                "Slope",
                "Local elevation variability"
            ],
            "resolution_note": (
                "Both regions are analysed at approximately "
                "3.5 to 3.8 km pixel spacing. "
                "The grids are similar in scale, not identical."
            )
        },

        "resolution_difference_percent": {
            "east_west": round(
                abs(earth_x - mars_x) / mars_x * 100, 2
            ),
            "north_south": round(
                abs(earth_y - mars_y) / mars_y * 100, 2
            )
        },

        "earth": earth_results,
        "mars": mars_results,

        "scientific_limitations": [
            "Earth and Mars measurements come from different instruments.",
            "The regions are geographically different.",
            "SRTM and MOLA have different measurement and processing characteristics.",
            "Slope is an approximate calculation from gridded elevation.",
            "Local elevation variability is not a complete measurement of surface texture.",
            "This comparison does not establish rover-scale safety or mobility.",
            "Similar pixel spacing does not make the datasets scientifically identical."
        ]
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n========== TERRA-X MATCHED COMPARISON ==========")

    print("\nATACAMA — EARTH")
    print(json.dumps(earth_results, indent=4))

    print("\nJEZERO — MARS")
    print(json.dumps(mars_results, indent=4))

    print("\nResolution difference:")
    print(json.dumps(output["resolution_difference_percent"], indent=4))

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("Matched comparison completed!")


if __name__ == "__main__":
    main()
