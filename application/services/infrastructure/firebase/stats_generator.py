import firebase_admin
from firebase_admin import firestore, credentials
import logging
from typing import Any
from itertools import groupby
from functools import reduce

logger = logging.Logger(__name__)

class StatsGeneratorListener:
    def __init__(self, credentials_path="sensitive/serviceAccountKey.json"):
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client(database_id="default")
        self.diary_watch = None
        self._start_listen_to_diary_entries()
        self._start_listen_to_predictions()

    def stop(self):
        if self.diary_watch:
            self.diary_watch.unsubscribe()
            self.diary_watch = None

        if self.predictions_watch:
            self.predictions_watch.unsubscribe()
            self.predictions_watch = None

    def _start_listen_to_diary_entries(self):
        self.diary_watch = (
            self.db
            .collection_group("diary_entries")
            .on_snapshot(self._on_diary_collection_snapshot)
        )
        logger.info("Entries Listener started", 'Success')

    def _start_listen_to_predictions(self):
        self.predictions_watch = (
            self.db
            .collection_group("predictions")
            .on_snapshot(self._on_predictions_collection_snapshot)
        )
        logger.info("Listener started", 'Success')

    def _on_diary_collection_snapshot(self, collection_snapshot, changes, read_time):
        affected_user_ids: set[str] = set()

        for change in changes:
            doc = change.document
            user_doc = doc.reference.parent.parent
            if user_doc is None:
                continue
            affected_user_ids.add(user_doc.id)

        for user_id in affected_user_ids:
            stats = self._build_user_entries_stats(user_id)
            self.db.collection("users").document(user_id) \
                .collection("stats") \
                .document("main") \
                .set(stats, merge = True)
            
    def _on_predictions_collection_snapshot(self, collection_shaphot, changes, read_time):
        affected_user_ids: set[str] = set()

        for change in changes:
            doc = change.document
            user_doc = doc.reference.parent.parent
            if user_doc is None:
                continue
            affected_user_ids.add(user_doc.id)

        for user_id in affected_user_ids:
            stats = self._build_predictions_stats(user_id)
            
            print(stats)
            self.db.collection("users").document(user_id) \
                .collection("stats") \
                .document("main") \
               .set({
                    "ratedRiskReasons": stats
                }, merge=True)
   
    def _build_predictions_stats(self, user_id: str) -> list[dict[str, Any]]:
        query = self.db.collection("users").document(user_id).collection("predictions")
        entries: list[dict[str, Any]] = []

        for doc in query.stream():
            entries.append(doc.to_dict())

        risk_reasons = reduce(lambda x, y: x + y, [entry.get('riskReasons', []) for entry in entries])
        rated_risk_reasons = [{"count": risk_reasons.count(s), "riskReason": s} for s in set(risk_reasons)]
        return rated_risk_reasons

    def _build_user_entries_stats(self, user_id: str) -> dict[str, Any]:
        query = self.db.collection("users").document(user_id).collection("diary_entries")
        entries: list[dict[str, Any]] = []

        for doc in query.stream():
            entries.append(doc.to_dict())

        entries.sort(key=lambda x: x['localDate'])
        entries_grouped = {local_date: list(entries) for local_date, entries in groupby(entries, key=lambda x: x['localDate'])}
        dated_wellbeing_scores = [{"date": local_date, "score": float(sum([d.get('wellbeingScore', 0) for d in entries_grouped[local_date]]))/len(entries_grouped[local_date])} for local_date in entries_grouped.keys()]
        
        entries_symptoms = [entry.get('symptoms', []) for entry in entries]
        entries_symptoms = reduce(lambda x, y: x + y, entries_symptoms)
        rated_symptoms = [{"count": entries_symptoms.count(s), "symptom": s} for s in set(entries_symptoms)]
        return {
            "datedWellbeingScores": dated_wellbeing_scores,
            "symptomsRated": rated_symptoms
        }
        
if __name__ == "__main__":
    import time
    gen = StatsGeneratorListener()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping listener...", flush=True)
        gen.stop()
