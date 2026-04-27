from __future__ import annotations

from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseClient:
    def __init__(self, credentials_path: str = "sensitive/serviceAccountKey.json"):
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client(database_id="default")

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        doc = (
            self.db.collection("users")
            .document(user_id)
            .collection("profile")
            .document("main")
            .get()
        )
        return doc.to_dict() if doc.exists else None

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

        ref.set(entry)
        return ref.id

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

        ref = (
            self.db.collection("users") 
            .document(user_id) 
            .collection("predictions") 
            .document() 
            )
        ref.set(prediction)
        return ref.id
    
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