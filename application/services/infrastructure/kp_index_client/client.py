from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import requests


class KpIndexClient:
    def __init__(self):
        self.noaa_url = (
            "https://services.swpc.noaa.gov/products/"
            "noaa-planetary-k-index-forecast.json"
        )

    def get_kp_index_at(self, dt: datetime) -> float:
        return self._get_nearest_kp(
            dt=dt,
            allowed_statuses={"observed", "estimated"},
            max_distance=timedelta(days=10),
        )

    def get_forecast_kp_index_at(self, dt: datetime) -> float:
        return self._get_nearest_kp(
            dt=dt,
            allowed_statuses={"predicted", "estimated"},
            max_distance=timedelta(days=2),
        )

    def _get_nearest_kp(
        self,
        dt: datetime,
        allowed_statuses: set[str],
        max_distance: timedelta,
    ) -> float:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        target = dt.astimezone(timezone.utc)
        rows = self.get_rows()

        candidates = [
            row for row in rows
            if row["status"] in allowed_statuses
        ]

        if not candidates:
            return 2.0

        nearest = min(
            candidates,
            key=lambda row: abs(row["datetime"] - target),
        )

        distance = abs(nearest["datetime"] - target)

        if distance <= max_distance:
            return nearest["kp"]
        past_candidates = [
            row for row in candidates
            if row["datetime"] <= target
        ]

        if past_candidates:
            latest = max(
                past_candidates,
                key=lambda row: row["datetime"],
            )
            return latest["kp"]

        return nearest["kp"]

    def get_rows(self) -> list[dict[str, Any]]:
        response = requests.get(
            self.noaa_url,
            timeout=(2, 10),
            headers={
                "User-Agent": "WeatherHealthApi/1.0",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return []

        result: list[dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            raw_time = item.get("time_tag")
            raw_kp = item.get("kp")
            raw_status = item.get("observed")

            if raw_time is None or raw_kp is None or raw_status is None:
                continue

            try:
                parsed_time = datetime.fromisoformat(raw_time).replace(
                    tzinfo=timezone.utc
                )
                kp = float(raw_kp)
            except (TypeError, ValueError):
                continue

            result.append(
                {
                    "datetime": parsed_time,
                    "kp": kp,
                    "status": str(raw_status).lower(),
                    "noaaScale": item.get("noaa_scale"),
                }
            )

        result.sort(key=lambda row: row["datetime"])
        return result