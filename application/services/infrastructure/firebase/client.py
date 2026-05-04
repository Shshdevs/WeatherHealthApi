from __future__ import annotations

from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore, messaging
from datetime import datetime, timezone


class FirebaseClient:
    def __init__(self, credentials_path: str = "sensitive/serviceAccountKey.json"):
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client(database_id="default")

    def get_all_users(self) -> list[dict[str, Any]]:
        users = []
        for doc in self.db.collection("users").stream():
            data = doc.to_dict() or {}
            data["id"] = doc.id
            users.append(data)
        return users
    
    def get_user_settings(self, user_id: str) -> dict[str, Any]:
        doc = (
            self.db.collection("users")
            .document(user_id)
            .collection("settings")
            .document("main")
            .get()
        )

        if not doc.exists:
            return {
                "push_enabled": True,
                "only_high_risk_notifications": False,
                "symptoms_notify_about": [],
                "allowed_notification_hours": {
                    "from": "00:00",
                    "to": "23:00",
                },
            }

        return doc.to_dict() or {}
    
    def save_notification(self, user_id: str, notification: dict[str, Any]) -> str:
        notification.setdefault("createdAt", firestore.SERVER_TIMESTAMP)

        ref = (
            self.db.collection("users")
            .document(user_id)
            .collection("notifications")
            .document()
        )

        ref.set(notification)
        return ref.id
    
    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        doc = (
            self.db.collection("users")
            .document(user_id)
            .get()
        )

        if not doc.exists:
            return {
                "healthCategory": "GENERAL",
            }

        data = doc.to_dict() or {}

        return {
            "meteosensitivity_score": data.get("meteosensitivity_score", 5),
            "healthCategory": data.get("healthCategory", "GENERAL"),
            "age": self._calculate_age(data.get("birthDate")),
            "token": data.get("token")
        }

    def update_user_location(self, user_id: str, latitude: float, longitude: float):
        self.db.collection("users").document(user_id).update({"latitude": latitude, "longitude": longitude})

    def _calculate_age(self, birth_date) -> int:
        if birth_date is None:
            return 30

        if hasattr(birth_date, "to_datetime"):
            birth_dt = birth_date.to_datetime()
        else:
            birth_dt = birth_date

        today = datetime.now(timezone.utc).date()
        born = birth_dt.date()

        age = today.year - born.year - (
            (today.month, today.day) < (born.month, born.day)
        )

        return max(age, 0)
    
    def get_diary_entries(self, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        query = (
            self.db.collection("users")
            .document(user_id)
            .collection("diary_entries")
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
        )

        if limit is not None:
            query = query.limit(limit)

        entries: list[dict[str, Any]] = []

        for doc in query.stream():
            data = doc.to_dict()
            data["id"] = doc.id
            entries.append(data)

        entries.reverse()
        return entries

    def create_diary_entry(self, user_id: str, entry: dict[str, Any]) -> str:
        entry.setdefault("createdAt", firestore.SERVER_TIMESTAMP)

        ref = (
            self.db.collection("users")
            .document(user_id)
            .collection("diary_entries")
            .document()
        )
        entry["id"] = ref.id

        ref.set(entry)
        return ref.id

    def send_fcm_notification(
        self,
        token: str,
        data: dict[str, str] | None = None,
    ) -> str:
        message = messaging.Message(
            token=token,
            android=messaging.AndroidConfig(
                priority="high"
            ),
            data=data or {},
        )

        return messaging.send(message)
    
    def save_model_meta(self, user_id: str, meta: dict[str, Any]) -> None:
        meta = {
            **meta,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        
        self.db.collection("users") \
            .document(user_id) \
            .collection("ml_model") \
            .document("meta") \
            .set(meta, merge=True)

    def get_model_meta(self, user_id: str) -> dict[str, Any] | None:
        doc = self.db.collection("users") \
            .document(user_id) \
            .collection("ml_model") \
            .document("meta") \
            .get()
        return doc.to_dict() if doc.exists else None

    def save_prediction(
        self,
        user_id: str,
        prediction: dict[str, Any],
    ) -> str:
        prediction.setdefault("createdAt", firestore.SERVER_TIMESTAMP)
        user_ref = self.db.collection("users").document(user_id)

        prediction_ref = (
            user_ref
            .collection("predictions")
            .document()
        )

        batch = self.db.batch()

        batch.set(prediction_ref, prediction)

        risk_reasons = set(prediction.get("riskReasons") or [])

        for reason in risk_reasons:
            stat_ref = (
                user_ref
                .collection("riskReasonStats")
                .document(reason)
            )

            batch.set(
                stat_ref,
                {
                    "reason": reason,
                    "count": firestore.Increment(1),
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

        batch.commit()
        return prediction_ref.id
    
    def save_predictions(
        self,
        user_id: str,
        predictions: list[dict[str, Any]],
    ) -> list[str]:
        ids = []

        for prediction in predictions:
            prediction_id = self.save_prediction(
                user_id=user_id,
                prediction=prediction,
            )
            ids.append(prediction_id)

        return ids