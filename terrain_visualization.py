
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.enums import Resampling

DATA_FILE = "S24W070.hgt"


def create_terrain_maps():

    print("TERRA-X: Preparing terrain visualizations...")

    with rasterio.open(DATA_FILE) as dataset:

        output_height = 900
        output_width = 900

        elevation = dataset.read(
            1,
            out_shape=(output_height, output_width),
            masked=True,
            resampling=Resampling.average
        )

        elevation_array = elevation.astype(
            np.float32
        ).filled(np.nan)

        latitude = (
            dataset.bounds.bottom + dataset.bounds.top
        ) / 2

        latitude_rad = np.radians(latitude)

        metres_per_degree_lat = (
            111132.92
            - 559.82 * np.cos(2 * latitude_rad)
            + 1.175 * np.cos(4 * latitude_rad)
        )

        metres_per_degree_lon = (
            111412.84 * np.cos(latitude_rad)
            - 93.5 * np.cos(3 * latitude_rad)
        )

        pixel_size_x = (
            abs(dataset.transform.a)
            * metres_per_degree_lon
            * dataset.width / output_width
        )

        pixel_size_y = (
            abs(dataset.transform.e)
            * metres_per_degree_lat
            * dataset.height / output_height
        )

        # 1. SLOPE CALCULATION

        print("TERRA-X: Calculating slope...")

        gradient_y, gradient_x = np.gradient(
            elevation_array,
            pixel_size_y,
            pixel_size_x
        )

        slope = np.degrees(
            np.arctan(
                np.sqrt(gradient_x**2 + gradient_y**2)
            )
        )

        # 2. ROUGHNESS CALCULATION

        print("TERRA-X: Calculating surface roughness...")

        height, width = elevation_array.shape

        roughness = np.full(
            (height, width),
            np.nan,
            dtype=np.float32
        )

        for row in range(1, height - 1):

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

            valid = np.all(
                np.isfinite(neighbourhood),
                axis=0
            )

            row_roughness = np.full(
                width - 2,
                np.nan,
                dtype=np.float32
            )

            row_roughness[valid] = np.std(
                neighbourhood[:, valid],
                axis=0
            )

            roughness[row, 1:-1] = row_roughness

        # 3. GENERATE ELEVATION MAP

        print("TERRA-X: Generating elevation map...")

        fig, ax = plt.subplots(figsize=(9, 8))

        elevation_plot = ax.imshow(
            elevation_array,
            cmap="terrain"
        )

        ax.set_title(
            "TERRA-X | Elevation Map",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        fig.colorbar(
            elevation_plot,
            ax=ax,
            label="Elevation (metres)"
        )

        plt.tight_layout()

        plt.savefig(
            "elevation_map.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        # 4. GENERATE SLOPE MAP

        print("TERRA-X: Generating slope map...")

        fig, ax = plt.subplots(figsize=(9, 8))

        slope_plot = ax.imshow(
            slope,
            cmap="magma",
            vmin=0,
            vmax=45
        )

        ax.set_title(
            "TERRA-X | Slope Map",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        fig.colorbar(
            slope_plot,
            ax=ax,
            label="Slope (degrees)"
        )

        plt.tight_layout()

        plt.savefig(
            "slope_map.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        # 5. GENERATE ROUGHNESS MAP

        print("TERRA-X: Generating roughness map...")

        fig, ax = plt.subplots(figsize=(9, 8))

        roughness_plot = ax.imshow(
            roughness,
            cmap="inferno"
        )

        ax.set_title(
            "TERRA-X | Local Elevation Variability Map",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        fig.colorbar(
            roughness_plot,
            ax=ax,
            label="Local elevation standard deviation (metres)"
        )

        plt.tight_layout()

        plt.savefig(
            "roughness_map.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("\nTERRA-X: Visualization completed!")
        print("Created: elevation_map.png")
        print("Created: slope_map.png")
        print("Created: roughness_map.png")


if __name__ == "__main__":
    create_terrain_maps()
