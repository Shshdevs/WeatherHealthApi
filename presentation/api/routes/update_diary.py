from fastapi import APIRouter, HTTPException, Query
from container import container
from typing import Any

router = APIRouter(prefix = '/api', tags = ['Update'])

@router.post("/{user_id}/diary")
def update_diary(user_id: str, 
    payload: dict[str, Any], 
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180)
):
    try:
        container.firebase_client.create_diary_entry(user_id, payload)
        diary_entries = container.firebase_client.get_diary_entries(user_id)
        kp = container.kp_index_client.get_effective_kp_index()
        weather_by_entry_id = container.open_meteo_client.build_weather_by_entry_id(
            diary_entries,
            latitude,
            longitude,
            kp
        )
        prediction_feature_rows = container.open_meteo_client.build_prediction_feature_rows(latitude, longitude, kp)

        result = container.ml_service.run_user_prediction_pipeline(
            user_id=user_id,
            diary_entries=diary_entries,
            weather_by_entry_id=weather_by_entry_id,
            prediction_feature_rows=prediction_feature_rows,
        )
        container.firebase_client.save_model_meta(user_id, result['model'])
        return {
            "status_code": 200,
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "predictions": result['predictions'],
        }

    except Exception as e:
        raise HTTPException(400, str(e))