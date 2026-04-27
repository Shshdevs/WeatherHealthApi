from __future__ import annotations

import requests


class KpIndexClient:
    def __init__(self):
        self.gfz_nowcast_url = (
            "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_nowcast.txt"
        )

    def get_effective_kp_index(self) -> float:
        kp = self.get_gfz_latest_kp()

        if kp is not None:
            return kp

        raise RuntimeError("Unable to fetch Kp-index from GFZ Potsdam")

    def get_gfz_latest_kp(self) -> float | None:
        text = self._get_text(self.gfz_nowcast_url)
        if not text:
            return None

        rows = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]

        for row in reversed(rows):
            parts = row.split()

            if len(parts) < 8:
                continue

            try:
                return float(parts[7])  # Kp
            except ValueError:
                continue

        return None

    def _get_text(self, url: str) -> str | None:
        try:
            response = requests.get(
                url,
                timeout=(2, 5),
                headers={
                    "User-Agent": "WeatherHealthApi/1.0",
                    "Accept": "text/plain",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException:
            return None