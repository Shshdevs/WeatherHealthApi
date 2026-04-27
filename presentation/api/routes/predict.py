from fastapi import APIRouter, HTTPException, Query
from container import container

router = APIRouter(prefix = '/api', tags = ["Predict"])

@router.get("/{user_id}/predict")
def predict(
    user_id: str,
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180)
    ):
    try:
        kp = container.weather_prediction_center_client.get_effective_kp_index()
        feature_rows = container.open_meteo_client.build_prediction_feature_rows(latitude, longitude, kp)
        predictions = container.ml_service.predict_risk(
            user_id=user_id,
            feature_rows = feature_rows
        )
        return {
            "status_code": 200,
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(status_code = 400, detail=str(e))