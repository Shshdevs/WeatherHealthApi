from __future__ import annotations

from pydantic import BaseModel


class WeatherPoint(BaseModel):
    date: str

    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    surface_pressure: float | None = None
    wind_speed_10m: float | None = None
    precipitation: float | None = None

    weather_code: float | None = None
    cloud_cover: float | None = None
    sunshine_duration: float | None = None


class AirQualityPoint(BaseModel):
    date: str

    alder_pollen: float | None = None
    birch_pollen: float | None = None
    grass_pollen: float | None = None
    mugwort_pollen: float | None = None
    olive_pollen: float | None = None
    ragweed_pollen: float | None = None


class WeatherForecastResult(BaseModel):
    latitude: float
    longitude: float
    elevation: float | None = None
    timezone_offset_seconds: int
    hourly: list[WeatherPoint]


class AirQualityForecastResult(BaseModel):
    latitude: float
    longitude: float
    timezone_offset_seconds: int
    hourly: list[AirQualityPoint]