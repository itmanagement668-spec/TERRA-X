
import rasterio
import numpy as np
import math

DATA_FILE = "S24W070.hgt"


def analyse_elevation():

    print("TERRA-X: Starting NASA terrain analysis...")

    with rasterio.open(DATA_FILE) as dataset:

        elevation = dataset.read(1, masked=True)

        valid_elevation = elevation.compressed()

        if valid_elevation.size == 0:
            raise ValueError("No valid elevation data found.")

        # 1. ELEVATION ANALYSIS

        minimum_elevation = float(np.min(valid_elevation))
        maximum_elevation = float(np.max(valid_elevation))
        average_elevation = float(np.mean(valid_elevation))

        # 2. APPROXIMATE PIXEL DIMENSIONS

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

        pixel_size_x = abs(dataset.transform.a) * metres_per_degree_lon
        pixel_size_y = abs(dataset.transform.e) * metres_per_degree_lat

        # 3. PREPARE ELEVATION ARRAY

        elevation_array = elevation.astype(np.float32).filled(np.nan)

        height, width = elevation_array.shape

        # 4. SLOPE AND ROUGHNESS ANALYSIS

        print("TERRA-X: Calculating terrain slope and roughness...")

        slope = np.full(
            (height - 2, width - 2),
            np.nan,
            dtype=np.float32
        )

        roughness = np.full(
            (height - 2, width - 2),
            np.nan,
            dtype=np.float32
        )

        for row in range(1, height - 1):

            top = elevation_array[row - 1, 1:-1]
            bottom = elevation_array[row + 1, 1:-1]

            left = elevation_array[row, :-2]
            right = elevation_array[row, 2:]

            # SLOPE: central difference method

            valid_slope = (
                np.isfinite(top)
                & np.isfinite(bottom)
                & np.isfinite(left)
                & np.isfinite(right)
            )

            dz_dx = (right - left) / (2 * pixel_size_x)
            dz_dy = (bottom - top) / (2 * pixel_size_y)

            gradient = np.sqrt(dz_dx**2 + dz_dy**2)

            row_slope = np.degrees(np.arctan(gradient))

            slope[row - 1, valid_slope] = row_slope[valid_slope]

            # ROUGHNESS: standard deviation in a 3x3 neighbourhood

            neighbourhood = np.stack([
                elevation_array[row - 1, :-2],
                elevation_array[row - 1, 1:-1],
                elevation_array[row - 1, 2:],

                elevation_array[row, :-2],
                elevation_array[row, 1:-1],
                elevation_array[row, 2:],

                elevation_array[row + 1, :-2],
                elevation_array[row + 1, 1:-1],
                elevation_array[row + 1, 2:]
            ])

            valid_roughness = np.all(
                np.isfinite(neighbourhood),
                axis=0
            )

            row_roughness = np.full(
                width - 2,
                np.nan,
                dtype=np.float32
            )

            row_roughness[valid_roughness] = np.std(
                neighbourhood[:, valid_roughness],
                axis=0
            )

            roughness[row - 1] = row_roughness

        # 5. COLLECT VALID MEASUREMENTS

        valid_slope_values = slope[np.isfinite(slope)]

        valid_roughness_values = roughness[
            np.isfinite(roughness)
        ]

        if valid_slope_values.size == 0:
            raise ValueError("No valid slope measurements calculated.")

        if valid_roughness_values.size == 0:
            raise ValueError("No valid roughness measurements calculated.")

        # 6. RESULTS

        results = {
            "dataset": DATA_FILE,

            "width_pixels": dataset.width,
            "height_pixels": dataset.height,

            "minimum_elevation_m": round(minimum_elevation, 2),
            "maximum_elevation_m": round(maximum_elevation, 2),
            "average_elevation_m": round(average_elevation, 2),

            "valid_elevation_pixels": int(valid_elevation.size),

            "minimum_slope_degrees": round(
                float(np.min(valid_slope_values)), 2
            ),

            "maximum_slope_degrees": round(
                float(np.max(valid_slope_values)), 2
            ),

            "average_slope_degrees": round(
                float(np.mean(valid_slope_values)), 2
            ),

            "valid_slope_pixels": int(valid_slope_values.size),

            "minimum_roughness_m": round(
                float(np.min(valid_roughness_values)), 2
            ),

            "maximum_roughness_m": round(
                float(np.max(valid_roughness_values)), 2
            ),

            "average_roughness_m": round(
                float(np.mean(valid_roughness_values)), 2
            ),

            "valid_roughness_pixels": int(valid_roughness_values.size),

            "roughness_method": "3x3 local elevation standard deviation",

            "pixel_width_m_approx": round(pixel_size_x, 2),
            "pixel_height_m_approx": round(pixel_size_y, 2),

            "coordinate_system": str(dataset.crs)
        }

        return results


if __name__ == "__main__":

    analysis = analyse_elevation()

    print("\n========== TERRA-X RESULTS ==========")

    for key, value in analysis.items():
        print(f"{key}: {value}")

    print("====================================")
    print("Terrain analysis completed!")
