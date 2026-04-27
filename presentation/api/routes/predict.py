from fastapi import APIRouter, HTTPException, Query
from container import container

router = APIRouter(prefix="/api", tags=["Predict"])


@router.get("/{user_id}/predict")
def predict(
    user_id: str,
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        user_profile = container.firebase_client.get_user_profile(user_id)
        category = (user_profile or {}).get("healthCategory", "GENERAL")

        kp = container.kp_index_client.get_effective_kp_index()

        feature_rows = container.open_meteo_client.build_prediction_feature_rows(
            latitude=latitude,
            longitude=longitude,
            kp_index=kp,
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
        )

        return {
            "status_code": 200,
            "userId": user_id,
            "healthCategory": category,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "predictions": predictions,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))