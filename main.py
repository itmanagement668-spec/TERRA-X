
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Literal
from pathlib import Path

from terrain_engine import analyse_elevation
from mars_engine import analyse_mars
from matched_comparison import main as generate_matched_comparison


app = FastAPI(
    title="TERRA-X",
    description="Earth-to-Planet Analogue Intelligence Engine",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

MARS_RESULTS_FILE = BASE_DIR / "mars_analysis.json"
JEZERO_RESULTS_FILE = BASE_DIR / "jezero_analysis.json"
MATCHED_RESULTS_FILE = BASE_DIR / "matched_comparison.json"


MISSION_CRITERIA = {
    "Rover Mobility": [
        "Slope",
        "Surface roughness",
        "Terrain"
    ],
    "Surface Geology": [
        "Rock composition",
        "Geological formations"
    ],
    "Habitat Site Evaluation": [
        "Terrain",
        "Slope",
        "Thermal environment"
    ]
}


class ComparisonRequest(BaseModel):
    earth_location: str
    target_planet: Literal["Moon", "Mars"]
    mission_type: Literal[
        "Rover Mobility",
        "Surface Geology",
        "Habitat Site Evaluation"
    ]


class TerrainData(BaseModel):
    earth_location: str
    elevation_m: float | None = None
    slope_degrees: float | None = None
    roughness_m: float | None = None

    data_status: Literal[
        "measured",
        "simulated",
        "missing"
    ]

    source: str | None = None

    @model_validator(mode="after")
    def validate_terrain_data(self):

        measurements = [
            self.elevation_m,
            self.slope_degrees,
            self.roughness_m
        ]

        has_measurement = any(
            value is not None for value in measurements
        )

        if self.data_status in ["measured", "simulated"]:

            if not has_measurement:
                raise ValueError(
                    "Measured or simulated data must contain "
                    "at least one measurement."
                )

            if not self.source:
                raise ValueError(
                    "Measured or simulated data must include a source."
                )

        if self.data_status == "missing" and has_measurement:
            raise ValueError(
                "Missing data cannot contain measurement values."
            )

        return self


def load_json_file(file_path, analysis_function=None):

    if not file_path.exists() and analysis_function is not None:
        analysis_function()

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} was not created."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_mars_results():
    return load_json_file(
        MARS_RESULTS_FILE,
        analyse_mars
    )


def load_jezero_results():
    return load_json_file(
        JEZERO_RESULTS_FILE
    )


def load_matched_results():
    return load_json_file(
        MATCHED_RESULTS_FILE,
        generate_matched_comparison
    )


@app.get("/")
def home():

    return {
        "project": "TERRA-X",
        "status": "Online",
        "version": "1.3.0",

        "planetary_datasets": [
            "Earth",
            "Mars"
        ],

        "regional_studies": [
            "Atacama Desert",
            "Jezero Crater"
        ],

        "capabilities": [
            "Earth terrain analysis",
            "Global Mars terrain analysis",
            "Jezero regional terrain analysis",
            "Resolution-matched planetary comparison",
            "Elevation analysis",
            "Slope analysis",
            "Local elevation variability analysis"
        ]
    }


@app.get("/locations")
def get_locations():

    return {
        "locations": [
            {
                "name": "Atacama Desert",
                "planet": "Earth",
                "terrain": "Desert",
                "dataset": "NASA SRTM"
            },
            {
                "name": "Icelandic Highlands",
                "planet": "Earth",
                "terrain": "Volcanic",
                "dataset_status": "Not connected"
            },
            {
                "name": "Jezero Crater",
                "planet": "Mars",
                "terrain": "Impact crater",
                "dataset": "NASA MOLA MEGDR"
            }
        ]
    }


@app.post("/compare")
def compare_location(request: ComparisonRequest):

    criteria = MISSION_CRITERIA[request.mission_type]

    return {
        "earth_location": request.earth_location,
        "target_planet": request.target_planet,
        "mission_type": request.mission_type,
        "comparison_criteria": criteria,

        "data_availability": {
            "earth": "NASA SRTM analysis available",

            "mars": (
                "NASA MOLA global and Jezero regional analyses available"
                if request.target_planet == "Mars"
                else "Moon dataset not yet connected"
            )
        },

        "message": "Mission profile successfully identified!"
    }


@app.post("/terrain")
def add_terrain_data(data: TerrainData):

    return {
        "message": "Terrain data received successfully!",
        "terrain_record": data.model_dump()
    }


@app.get("/terrain-analysis")
def get_terrain_analysis():

    try:
        results = analyse_elevation()

        return {
            "project": "TERRA-X",
            "analysis": "NASA SRTM terrain analysis",
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Terrain analysis failed: {str(error)}"
        )


@app.get("/mars-analysis")
def get_mars_analysis():

    try:
        results = load_mars_results()

        return {
            "project": "TERRA-X",
            "analysis": "NASA MOLA global Mars terrain analysis",
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Mars analysis failed: {str(error)}"
        )


@app.get("/jezero-analysis")
def get_jezero_analysis():

    try:
        results = load_jezero_results()

        return {
            "project": "TERRA-X",
            "analysis": "NASA MOLA Jezero regional terrain analysis",
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Jezero analysis failed: {str(error)}"
        )


@app.get("/matched-comparison")
def get_matched_comparison():

    try:
        results = load_matched_results()

        return results

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Matched comparison failed: {str(error)}"
        )


@app.get("/planetary-comparison")
def get_planetary_comparison():

    try:
        earth_results = analyse_elevation()
        mars_results = load_mars_results()
        jezero_results = load_jezero_results()
        matched_results = load_matched_results()

        return {
            "project": "TERRA-X",

            "comparison_type": (
                "Exploratory planetary terrain comparison"
            ),

            "earth": {
                "location": "Atacama Desert region",
                "dataset": "NASA SRTM",
                "resolution": "Approximately 30 metres",
                "results": earth_results
            },

            "mars_global": {
                "location": "Global Mars dataset",
                "dataset": "NASA MOLA MEGDR",
                "resolution": "Approximately 3.7 kilometres",
                "results": mars_results
            },

            "mars_regional": {
                "location": "Jezero Crater surroundings",
                "dataset": "NASA MOLA MEGDR",
                "resolution": "Approximately 3.5 kilometres at Jezero",
                "results": jezero_results
            },

            "resolution_matched_comparison": matched_results,

            "scientific_note": (
                "The resolution-matched comparison aggregates "
                "Earth SRTM elevation data and compares it with "
                "a regional subset of Mars MOLA data. The grids "
                "have similar approximate pixel spacing, but "
                "the instruments, processing methods and "
                "geographic regions differ."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Planetary comparison failed: {str(error)}"
        )


@app.get("/terrain-map")
def get_terrain_map():

    return serve_map("terra_x_terrain_maps.png")


@app.get("/elevation-map")
def get_elevation_map():

    return serve_map("elevation_map.png")


@app.get("/slope-map")
def get_slope_map():

    return serve_map("slope_map.png")


@app.get("/roughness-map")
def get_roughness_map():

    return serve_map("roughness_map.png")


def serve_map(filename):

    map_file = BASE_DIR / filename

    if not map_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found. Generate the map first."
        )

    return FileResponse(
        map_file,
        media_type="image/png"
    )
