from fastapi import APIRouter, HTTPException, Query
from container import container
from presentation.utils.attach_weather_to_predictions import attach_weather_to_predictions

router = APIRouter(prefix="/api", tags=["Predict"])


@router.get("/{user_id}/predict")
def predict(
    user_id: str,
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        user_profile = container.firebase_client.get_user_profile(user_id)
        container.firebase_client.update_user_location(user_id=user_id, latitude=latitude, longitude=longitude)
        
        category = (user_profile or {}).get("healthCategory", "GENERAL")
        meteosensitivity_score = (user_profile or {}).get("meteosensitivity_score", 5)
        age = (user_profile or {}).get("age", 30)

        feature_rows = container.open_meteo_client.build_prediction_feature_rows(
            latitude=latitude,
            longitude=longitude,
            kp_index_client=container.kp_index_client,
            forecast_days = 7
        )

        category_features = container.ml_service._encode_user_category(category)

        feature_rows = [
            {
                **row,
                **category_features,
            }
            for row in feature_rows
        ]

        predictions = container.ml_service.predict_risk(
            user_id=user_id,
            feature_rows=feature_rows,
            meteosensitivity_score = meteosensitivity_score,
            age = age
        )

        return {
            "status_code": 200,
            "userId": user_id,
            "healthCategory": category,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "predictions": attach_weather_to_predictions(predictions, feature_rows),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
