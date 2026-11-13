
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Literal
from pathlib import Path

from terrain_engine import analyse_elevation

app = FastAPI(
    title="TERRA-X",
    description="Earth-to-Planet Analogue Intelligence Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

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


@app.get("/")
def home():

    return {
        "project": "TERRA-X",
        "status": "Online"
    }


@app.get("/locations")
def get_locations():

    return {
        "locations": [
            {
                "name": "Atacama Desert",
                "planet": "Earth",
                "terrain": "Desert"
            },
            {
                "name": "Icelandic Highlands",
                "planet": "Earth",
                "terrain": "Volcanic"
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
