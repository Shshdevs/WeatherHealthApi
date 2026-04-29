from __future__ import annotations

from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from datetime import datetime, timezone, timedelta
from retry_requests import retry

from application.services.infrastructure.open_meteo.schemas import (
    AirQualityForecastResult,
    AirQualityPoint,
    WeatherForecastResult,
    WeatherPoint,
)


class OpenMeteoClient:
    def __init__(self):
        cache_session = requests_cache.CachedSession(
            "cache/.cache",
            expire_after=3600,
        )

        retry_session = retry(
            cache_session,
            retries=5,
            backoff_factor=0.2,
        )

        self.client = openmeteo_requests.Client(session=retry_session)

        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7,
        past_days: int = 2,
        timezone_name: str = "auto",
    ) -> WeatherForecastResult:
        hourly_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "sunshine_duration",
        ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": hourly_variables,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": timezone_name,
        }

        response = self.client.weather_api(
            self.forecast_url,
            params=params,
        )[0]

        hourly = response.Hourly()

        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        dataframe = pd.DataFrame(
            {
                "date": dates.astype(str),
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                "surface_pressure": hourly.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
                "precipitation": hourly.Variables(4).ValuesAsNumpy(),
                "weather_code": hourly.Variables(5).ValuesAsNumpy(),
                "cloud_cover": hourly.Variables(6).ValuesAsNumpy(),
                "sunshine_duration": hourly.Variables(7).ValuesAsNumpy(),
            }
        )

        hourly_points = [
            WeatherPoint(**row)
            for row in dataframe.to_dict(orient="records")
        ]

        return WeatherForecastResult(
            latitude=response.Latitude(),
            longitude=response.Longitude(),
            elevation=response.Elevation(),
            timezone_offset_seconds=response.UtcOffsetSeconds(),
            hourly=hourly_points,
        )

    def get_air_quality(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 4,
        past_days: int = 2,
        timezone_name: str = "auto",
    ) -> AirQualityForecastResult:
        
        hourly_variables = [
            "alder_pollen",
            "birch_pollen",
            "grass_pollen",
            "mugwort_pollen",
            "olive_pollen",
            "ragweed_pollen",
        ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": hourly_variables,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": timezone_name,
        }

        response = self.client.weather_api(
            self.air_quality_url,
            params=params,
        )[0]

        hourly = response.Hourly()

        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        dataframe = pd.DataFrame(
            {
                "date": dates.astype(str),
                "alder_pollen": hourly.Variables(0).ValuesAsNumpy(),
                "birch_pollen": hourly.Variables(1).ValuesAsNumpy(),
                "grass_pollen": hourly.Variables(2).ValuesAsNumpy(),
                "mugwort_pollen": hourly.Variables(3).ValuesAsNumpy(),
                "olive_pollen": hourly.Variables(4).ValuesAsNumpy(),
                "ragweed_pollen": hourly.Variables(5).ValuesAsNumpy(),
            }
        )

        hourly_points = [
            AirQualityPoint(**row)
            for row in dataframe.to_dict(orient="records")
        ]

        return AirQualityForecastResult(
            latitude=response.Latitude(),
            longitude=response.Longitude(),
            timezone_offset_seconds=response.UtcOffsetSeconds(),
            hourly=hourly_points,
        )

    def build_weather_by_entry_id(
        self,
        diary_entries: list[dict],
        latitude: float,
        longitude: float,
        kp_index_client,
        past_days: int = 1,
    ) -> dict[str, dict]:
        forecast = self.get_forecast(
            latitude=latitude,
            longitude=longitude,
            past_days=past_days,
            forecast_days=1,
        )

        air_quality = self.get_air_quality(
            latitude=latitude,
            longitude=longitude,
            past_days=7,
            forecast_days=1,
        )

        weather_df = self._weather_result_to_dataframe(forecast)
        pollen_df = self._air_quality_result_to_dataframe(air_quality)

        df = self._merge_weather_and_pollen(weather_df, pollen_df)
        df = self._add_delta_features(df)

        result: dict[str, dict[str, Any]] = {}

        for entry in diary_entries:
            entry_id = entry.get("id")
            
            entry_time = self._entry_datetime(entry)

            kp_index = kp_index_client.get_kp_index_at(entry_time)

            matched_row = self._nearest_row(df, entry_time)

            if matched_row is None:
                continue

            result[entry_id] = self._row_to_model_features(
                row=matched_row,
                kp_index=kp_index,
            )

        return result

    def build_prediction_feature_rows(
        self,
        latitude: float,
        longitude: float,
        kp_index_client,
        forecast_days: int = 2,
        timezone_name: str = "auto",
    ) -> list[dict[str, Any]]:

        forecast = self.get_forecast(
            latitude=latitude,
            longitude=longitude,
            past_days=1,
            forecast_days=forecast_days,
            timezone_name=timezone_name,
        )

        air_quality = self.get_air_quality(
            latitude=latitude,
            longitude=longitude,
            past_days=1,
            forecast_days=min(forecast_days, 4),
            timezone_name=timezone_name,
        )

        weather_df = self._weather_result_to_dataframe(forecast)
        pollen_df = self._air_quality_result_to_dataframe(air_quality)

        df = self._merge_weather_and_pollen(weather_df, pollen_df)
        df = self._add_delta_features(df)

        rows = []

        for from_hour, to_hour, time_of_day in [
            (6, 12, "MORNING"),
            (12, 18, "DAY"),
            (18, 23, "EVENING"),
            (23, 6, "NIGHT"),
        ]:
            period_df = self._period_rows(df, from_hour, to_hour)

            if period_df.empty:
                continue

            grouped = period_df.groupby(period_df["date"].dt.date)

            for forecast_date, day_period_df in grouped:
                representative = day_period_df.iloc[len(day_period_df) // 2]

                kp_datetime = representative["date"]

                if kp_datetime.tzinfo is None:
                    kp_datetime = kp_datetime.to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    kp_datetime = kp_datetime.to_pydatetime()

                kp_index = kp_index_client.get_forecast_kp_index_at(kp_datetime)

                row = self._row_to_model_features(
                    row=representative,
                    kp_index=kp_index,
                )

                row.update(
                    {
                        "forecastDate": str(forecast_date),
                        "fromHour": from_hour,
                        "toHour": to_hour,
                        "timeOfDay": time_of_day,
                    }
                )

                rows.append(row)

        return rows

    def _weather_result_to_dataframe(
        self,
        forecast: WeatherForecastResult,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            [point.model_dump() for point in forecast.hourly]
        )

        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    def _air_quality_result_to_dataframe(
        self,
        forecast: AirQualityForecastResult,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            [point.model_dump() for point in forecast.hourly]
        )

        df["date"] = pd.to_datetime(df["date"], utc=True)

        pollen_columns = [
            "alder_pollen",
            "birch_pollen",
            "grass_pollen",
            "mugwort_pollen",
            "olive_pollen",
            "ragweed_pollen",
        ]

        existing_columns = [
            column for column in pollen_columns
            if column in df.columns
        ]

        if existing_columns:
            df["pollen_index"] = df[existing_columns].fillna(0).max(axis=1)
        else:
            df["pollen_index"] = 0

        return df[["date", "pollen_index"]].sort_values("date").reset_index(drop=True)

    def _merge_weather_and_pollen(
        self,
        weather_df: pd.DataFrame,
        pollen_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.merge_asof(
            weather_df.sort_values("date"),
            pollen_df.sort_values("date"),
            on="date",
            direction="nearest",
        ).fillna(0)

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True).copy()

        df["temperature_delta_6h"] = (
            df["temperature_2m"]
            - df["temperature_2m"].shift(6)
        ).fillna(0)

        df["temperature_delta_24h"] = (
            df["temperature_2m"]
            - df["temperature_2m"].shift(24)
        ).fillna(0)

        df["pressure_delta_3h"] = (
            df["surface_pressure"]
            - df["surface_pressure"].shift(3)
        ).fillna(0)

        df["pressure_delta_24h"] = (
            df["surface_pressure"]
            - df["surface_pressure"].shift(24)
        ).fillna(0)

        return df

    def _row_to_model_features(
        self,
        row: pd.Series,
        kp_index: float,
    ) -> dict[str, Any]:
        weather_code = int(
            self._safe_float(
                row.get("weather_code"),
                default=0,
            )
        )

        return {
            "temperature": self._safe_float(row.get("temperature_2m"), 0),
            "temperature_delta_6h": self._safe_float(row.get("temperature_delta_6h"), 0),
            "temperature_delta_24h": self._safe_float(row.get("temperature_delta_24h"), 0),
            "pressure": self._safe_float(row.get("surface_pressure"), 1013),
            "pressure_delta_3h": self._safe_float(row.get("pressure_delta_3h"), 0),
            "pressure_delta_24h": self._safe_float(row.get("pressure_delta_24h"), 0),
            "humidity": self._safe_float(row.get("relative_humidity_2m"), 50),
            "wind_speed": self._safe_float(row.get("wind_speed_10m"), 0),
            "precipitation": self._safe_float(row.get("precipitation"), 0),
            "kp_index": self._safe_float(kp_index, 1),
            "thunderstorm_probability": self._thunderstorm_probability(weather_code),
            "cloud_cover": self._safe_float(row.get("cloud_cover"), 0),
            "sunshine_duration": self._safe_float(row.get("sunshine_duration"), 3600),
            "pollen_index": self._safe_float(row.get("pollen_index"), 0),
        }

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)
    
    def _entry_datetime(self, entry: dict[str, Any]) -> pd.Timestamp:
        value = (
            entry.get("createdAt")
            or f"{entry.get('localDate')}T{entry.get('localTime', '12:00')}"
        )

        return pd.to_datetime(value, utc=True)

    def _nearest_row(
        self,
        df: pd.DataFrame,
        target: pd.Timestamp,
    ) -> pd.Series | None:
        if df.empty:
            return None

        index = (df["date"] - target).abs().idxmin()
        return df.loc[index]

    def _period_rows(
        self,
        df: pd.DataFrame,
        from_hour: int,
        to_hour: int,
    ) -> pd.DataFrame:
        hours = df["date"].dt.hour

        if from_hour < to_hour:
            return df[(hours >= from_hour) & (hours < to_hour)]

        return df[(hours >= from_hour) | (hours < to_hour)]

    def _thunderstorm_probability(self, weather_code: int) -> float:
        if weather_code in {95, 96, 99}:
            return 100.0

        return 0.0