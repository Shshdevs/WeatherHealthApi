WEATHER_FIELDS = [
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "precipitation",
    "kp_index",
    "thunderstorm_probability",
    "cloud_cover",
    "sunshine_duration",
    "pollen_index",
]


def attach_weather_to_predictions(
    predictions: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    return [
        {
            **prediction,
            "weather": {
                field: row.get(field)
                for field in WEATHER_FIELDS
            },
        }
        for prediction, row in zip(predictions, feature_rows)
    ]