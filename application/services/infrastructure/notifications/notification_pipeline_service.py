from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class NotificationPipelineService:
    def __init__(
        self,
        firebase_client,
        open_meteo_client,
        kp_index_client,
        ml_service,
    ):
        self.firebase_client = firebase_client
        self.open_meteo_client = open_meteo_client
        self.kp_index_client = kp_index_client
        self.ml_service = ml_service

    def run_for_all_users(self) -> dict[str, Any]:
        users = self.firebase_client.get_all_users()

        result = {
            "processed": 0,
            "sent": 0,
            "skipped": 0,
            "errors": [],
        }

        for user in users:
            user_id = user["id"]

            try:
                user_result = self.run_for_user(user_id, user)
                result["processed"] += 1
                result["sent"] += user_result["sent"]
                result["skipped"] += user_result["skipped"]

            except Exception as e:
                result["errors"].append({
                    "userId": user_id,
                    "error": str(e),
                })

        return result

    def run_for_user(self, user_id: str, user_doc: dict[str, Any]) -> dict[str, Any]:
        settings = self.firebase_client.get_user_settings(user_id)

        if not settings.get("push_enabled", True):
            return {"sent": 0, "skipped": 1, "reason": "PUSH_DISABLED"}

        if not self._is_allowed_notification_time(settings):
            return {"sent": 0, "skipped": 1, "reason": "QUIET_HOURS"}

        token = user_doc.get("token")
        latitude = user_doc.get("latitude")
        longitude = user_doc.get("longitude")

        if not token or latitude is None or longitude is None:
            return {"sent": 0, "skipped": 1, "reason": "MISSING_USER_DATA"}

        user_profile = self.firebase_client.get_user_profile(user_id)
        diary_entries = self.firebase_client.get_diary_entries(user_id)

        kp = self.kp_index_client.get_effective_kp_index()

        weather_by_entry_id = self.open_meteo_client.build_weather_by_entry_id(
            diary_entries=diary_entries,
            latitude=latitude,
            longitude=longitude,
            kp_index=kp,
        )

        prediction_feature_rows = self.open_meteo_client.build_prediction_feature_rows(
            latitude=latitude,
            longitude=longitude,
            kp_index=kp,
            forecast_days=2,
        )

        pipeline_result = self.ml_service.run_user_prediction_pipeline(
            user_id=user_id,
            diary_entries=diary_entries,
            weather_by_entry_id=weather_by_entry_id,
            prediction_feature_rows=prediction_feature_rows,
            user_profile=user_profile,
        )

        self.firebase_client.save_model_meta(
            user_id=user_id,
            meta=pipeline_result["model"],
        )

        self.firebase_client.save_predictions(
            user_id=user_id,
            predictions=pipeline_result["predictions"],
        )

        predictions_to_notify = self._filter_predictions(
            predictions=pipeline_result["predictions"],
            settings=settings,
        )

        sent = 0

        for prediction in predictions_to_notify:
            title, body = self._build_notification_text(prediction)

            self.firebase_client.send_fcm_notification(
                token=token,
                title=title,
                body=body,
                data={
                    "type": "WEATHER_HEALTH_RISK",
                    "forecastDate": str(prediction.get("forecastDate")),
                    "riskLevel": str(prediction.get("riskLevel")),
                },
            )

            self.firebase_client.save_notification(
                user_id=user_id,
                notification={
                    "sentAt": datetime.now(timezone.utc).isoformat(),
                    "status": "SENT",
                    "title": title,
                    "body": body,
                    "prediction": prediction,
                    "fcmToken": token,
                },
            )

            sent += 1

        return {
            "sent": sent,
            "skipped": 0 if sent else 1,
        }

    def _filter_predictions(
        self,
        predictions: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        only_high = settings.get("only_high_risk_notifications", False)
        allowed_symptoms = set(settings.get("symptoms_notify_about", []))

        for prediction in predictions:
            if only_high and prediction.get("riskLevel") != "HIGH":
                continue

            predicted_symptoms = set(prediction.get("predictedSymptoms", []))

            if allowed_symptoms and not predicted_symptoms.intersection(allowed_symptoms):
                continue

            if prediction.get("riskLevel") in {"MEDIUM", "HIGH"}:
                result.append(prediction)

        return result

    def _is_allowed_notification_time(self, settings: dict[str, Any]) -> bool:
        allowed = settings.get("allowed_notification_hours", {})
        from_time = allowed.get("from", "00:00")
        to_time = allowed.get("to", "23:00")

        now = datetime.now().time()
        start = datetime.strptime(from_time, "%H:%M").time()
        end = datetime.strptime(to_time, "%H:%M").time()

        if start <= end:
            return start <= now <= end

        return now >= start or now <= end

    def _build_notification_text(self, prediction: dict[str, Any]) -> tuple[str, str]:
        risk_level = prediction.get("riskLevel")
        date = prediction.get("forecastDate")
        symptoms = prediction.get("predictedSymptoms", [])

        title = "Прогноз самочувствия"

        if risk_level == "HIGH":
            title = "Высокий риск ухудшения самочувствия"

        symptoms_text = ", ".join(symptoms[:3]) if symptoms else "неприятные симптомы"

        body = f"{date}: возможны {symptoms_text}. Проверьте рекомендации в приложении."

        return title, body