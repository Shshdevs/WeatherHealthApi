from __future__ import annotations

import httpx


class WeatherPredictionCenterClient:
    def __init__(self):
        self.observed_kp_url = (
            "https://services.swpc.noaa.gov/products/"
            "noaa-planetary-k-index.json"
        )
        self.forecast_kp_url = (
            "https://services.swpc.noaa.gov/products/"
            "noaa-planetary-k-index-forecast.json"
        )

    async def get_latest_kp_index(self) -> float | None:
        data = await self._get_json(self.observed_kp_url)
        return self._extract_latest_kp(data)

    async def get_forecast_kp_index(self) -> float | None:
        data = await self._get_json(self.forecast_kp_url)
        return self._extract_max_kp(data)

    async def get_effective_kp_index(self) -> float:
        forecast_kp = await self.get_forecast_kp_index()

        if forecast_kp is not None:
            return forecast_kp

        latest_kp = await self.get_latest_kp_index()

        if latest_kp is not None:
            return latest_kp

        return 1.0

    async def _get_json(self, url: str) -> list:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def _extract_latest_kp(self, data: list) -> float | None:
        if len(data) <= 1:
            return None

        for row in reversed(data[1:]):
            kp = self._try_parse_kp(row)

            if kp is not None:
                return kp

        return None

    def _extract_max_kp(self, data: list) -> float | None:
        values = []

        for row in data[1:]:
            kp = self._try_parse_kp(row)

            if kp is not None:
                values.append(kp)

        return max(values) if values else None

    def _try_parse_kp(self, row: list) -> float | None:
        if not isinstance(row, list) or len(row) < 2:
            return None

        for item in reversed(row):
            try:
                return float(item)
            except (TypeError, ValueError):
                continue

        return None