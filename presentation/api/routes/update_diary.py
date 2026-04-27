from typing import Any

from fastapi import APIRouter, HTTPException, Query

from container import container

router = APIRouter(prefix="/api", tags=["Update"])


@router.post("/{user_id}/diary")
def update_diary(
    user_id: str,
    payload: dict[str, Any],
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        created_entry = container.firebase_client.create_diary_entry(
            user_id=user_id,
            payload=payload,
        )

        user_profile = container.firebase_client.get_user_profile(user_id)
        category = (user_profile or {}).get("healthCategory", "GENERAL")

        diary_entries = container.firebase_client.get_diary_entries(user_id)

        kp = container.kp_index_client.get_effective_kp_index()

        weather_by_entry_id = container.open_meteo_client.build_weather_by_entry_id(
            diary_entries=diary_entries,
            latitude=latitude,
            longitude=longitude,
            kp_index=kp,
        )

        prediction_feature_rows = container.open_meteo_client.build_prediction_feature_rows(
            latitude=latitude,
            longitude=longitude,
            kp_index=kp,
        )

        result = container.ml_service.run_user_prediction_pipeline(
            user_id=user_id,
            diary_entries=diary_entries,
            weather_by_entry_id=weather_by_entry_id,
            prediction_feature_rows=prediction_feature_rows,
            user_profile=user_profile,
        )

        container.firebase_client.save_model_meta(
            user_id=user_id,
            meta=result["model"],
        )

        return {
            "status_code": 200,
            "userId": user_id,
            "healthCategory": category,
            "createdEntry": created_entry,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "model": result["model"],
            "predictions": result["predictions"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))