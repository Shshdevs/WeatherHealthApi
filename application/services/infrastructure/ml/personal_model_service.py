from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from application.domain.enums.health import RiskReason, SymptomType, TimeOfDay


class PersonalModelService:
    def __init__(self, models_dir: str = "sensitive/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.feature_columns = [
            "hour",
            "time_of_day_code",

            "category_hypotonic",
            "category_hypertonic",
            "category_joint_disease",
            "category_migraine",
            "category_general",

            "temperature",
            "temperature_delta_6h",
            "temperature_delta_24h",
            "pressure",
            "pressure_delta_3h",
            "pressure_delta_24h",
            "humidity",
            "wind_speed",
            "precipitation",
            "kp_index",
            "thunderstorm_probability",
            "cloud_cover",
            "sunshine_duration",
            "pollen_index",
            "sleep_quality",
            "stress_score",
            "water_liters",
            "caffeine_cups",
        ]

    def _model_path(self, user_id: str) -> Path:
        return self.models_dir / f"{user_id}.joblib"

    def model_exists(self, user_id: str) -> bool:
        return self._model_path(user_id).exists()

    def run_user_prediction_pipeline(
        self,
        user_id: str,
        diary_entries: list[dict[str, Any]],
        weather_by_entry_id: dict[str, dict[str, Any]],
        prediction_feature_rows: list[dict[str, Any]],
        user_profile: dict[str, Any] | None = None,
        min_entries: int = 10,
    ) -> dict[str, Any]:
        
        model_existed_before = self.model_exists(user_id)

        category = (user_profile or {}).get("healthCategory", "GENERAL")

        training_df = self.build_training_dataframe(
            diary_entries=diary_entries,
            weather_by_entry_id=weather_by_entry_id,
            user_category=category,
        )

        if training_df.empty or len(training_df) < min_entries:
            raise ValueError("Empty dataframe")

        model_meta = self.train(
            user_id=user_id,
            training_df=training_df,
            min_entries=min_entries,
        )

        model_meta["status"] = "RETRAINED" if model_existed_before else "CREATED"

        prediction_feature_rows = [
            {
                **row,
                **self._encode_user_category(category),
            }
            for row in prediction_feature_rows
        ]
        predictions = self.predict_risk(
            user_id=user_id,
            feature_rows=prediction_feature_rows,
        )

        return {
            "userId": user_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": model_meta,
            "predictions": predictions,
        }

    def _is_bad_state(
        self,
        entry: dict[str, Any],
        symptoms: list[str],
    ) -> bool:
        wellbeing_score = int(entry.get("wellbeingScore", 5))
        energy_score = int(entry.get("energyScore", 5))
        stress_score = int(entry.get("stressScore", 1))
        sleep_quality = int(entry.get("sleepQuality", 5))

        severe_symptoms = {
            SymptomType.MIGRAINE.value,
            SymptomType.DIZZINESS.value,
            SymptomType.HEART_PALPITATION.value,
            SymptomType.PRESSURE_SPIKE.value,
            SymptomType.LOW_PRESSURE.value,
            SymptomType.BREATHING_DISCOMFORT.value,
            SymptomType.NAUSEA.value,
        }

        moderate_symptoms = {
            SymptomType.HEADACHE.value,
            SymptomType.WEAKNESS.value,
            SymptomType.FATIGUE.value,
            SymptomType.JOINT_PAIN.value,
            SymptomType.BACK_PAIN.value,
            SymptomType.IRRITABILITY.value,
            SymptomType.ANXIETY.value,
            SymptomType.INSOMNIA.value,
            SymptomType.DROWSINESS.value,
        }

        has_severe_symptom = any(
            symptom in severe_symptoms
            for symptom in symptoms
        )

        moderate_symptoms_count = sum(
            symptom in moderate_symptoms
            for symptom in symptoms
        )

        return (
            wellbeing_score <= 2
            or has_severe_symptom
            or moderate_symptoms_count >= 2
            or energy_score <= 2
            or stress_score >= 4
            or sleep_quality <= 2
        )
    def _predict_symptoms_by_reasons(
        self,
        risk_score: float,
        reasons: list[RiskReason],
        row: dict[str, Any],
    ) -> list[str]:
        if risk_score < 0.4:
            return []

        symptoms = set()

        is_hypotonic = int(row.get("category_hypotonic", 0)) == 1
        is_hypertonic = int(row.get("category_hypertonic", 0)) == 1
        is_joint = int(row.get("category_joint_disease", 0)) == 1
        is_migraine = int(row.get("category_migraine", 0)) == 1
        is_general = int(row.get("category_general", 0)) == 1

        pressure_reasons = {
            RiskReason.PRESSURE_DROP,
            RiskReason.PRESSURE_RISE,
            RiskReason.PRESSURE_SWING,
        }

        temp_reasons = {
            RiskReason.TEMPERATURE_DROP,
            RiskReason.TEMPERATURE_RISE,
            RiskReason.TEMPERATURE_SWING,
        }

        if any(reason in reasons for reason in pressure_reasons):
            symptoms.add(SymptomType.HEADACHE.value)

            if is_hypotonic:
                symptoms.add(SymptomType.DIZZINESS.value)
                symptoms.add(SymptomType.WEAKNESS.value)
                symptoms.add(SymptomType.LOW_PRESSURE.value)

            if is_hypertonic:
                symptoms.add(SymptomType.PRESSURE_SPIKE.value)
                symptoms.add(SymptomType.HEART_PALPITATION.value)

            if is_migraine:
                symptoms.add(SymptomType.MIGRAINE.value)
                symptoms.add(SymptomType.NAUSEA.value)

        if RiskReason.GEOMAGNETIC_STORM in reasons:
            symptoms.add(SymptomType.HEADACHE.value)
            symptoms.add(SymptomType.FATIGUE.value)

            if is_hypertonic:
                symptoms.add(SymptomType.PRESSURE_SPIKE.value)

            if is_migraine:
                symptoms.add(SymptomType.MIGRAINE.value)

        if RiskReason.HIGH_HUMIDITY in reasons:
            symptoms.add(SymptomType.WEAKNESS.value)
            symptoms.add(SymptomType.DROWSINESS.value)

            if is_hypotonic:
                symptoms.add(SymptomType.DIZZINESS.value)

            if is_joint:
                symptoms.add(SymptomType.JOINT_PAIN.value)
                symptoms.add(SymptomType.BACK_PAIN.value)

        if RiskReason.PRECIPITATION in reasons:
            if is_joint:
                symptoms.add(SymptomType.JOINT_PAIN.value)
                symptoms.add(SymptomType.BACK_PAIN.value)
            else:
                symptoms.add(SymptomType.FATIGUE.value)

        if RiskReason.STRONG_WIND in reasons:
            symptoms.add(SymptomType.HEADACHE.value)

            if is_migraine:
                symptoms.add(SymptomType.MIGRAINE.value)
                symptoms.add(SymptomType.NAUSEA.value)

        if RiskReason.HEAT_STRESS in reasons:
            symptoms.add(SymptomType.WEAKNESS.value)
            symptoms.add(SymptomType.FATIGUE.value)

            if is_hypertonic:
                symptoms.add(SymptomType.PRESSURE_SPIKE.value)
                symptoms.add(SymptomType.HEART_PALPITATION.value)

        if RiskReason.COLD_STRESS in reasons or RiskReason.TEMPERATURE_DROP in reasons:
            if is_joint:
                symptoms.add(SymptomType.JOINT_PAIN.value)
                symptoms.add(SymptomType.BACK_PAIN.value)
            else:
                symptoms.add(SymptomType.WEAKNESS.value)

        if any(reason in reasons for reason in temp_reasons):
            symptoms.add(SymptomType.FATIGUE.value)

            if is_migraine:
                symptoms.add(SymptomType.HEADACHE.value)

        if RiskReason.LOW_SUNLIGHT in reasons:
            symptoms.add(SymptomType.DROWSINESS.value)
            symptoms.add(SymptomType.FATIGUE.value)

        if RiskReason.HIGH_POLLEN in reasons:
            symptoms.add(SymptomType.BREATHING_DISCOMFORT.value)

        if is_general and not symptoms:
            symptoms.add(SymptomType.FATIGUE.value)
            symptoms.add(SymptomType.WEAKNESS.value)

        if risk_score >= 0.7:
            return list(symptoms)

        soft_symptoms = {
            SymptomType.FATIGUE.value,
            SymptomType.WEAKNESS.value,
            SymptomType.DROWSINESS.value,
            SymptomType.HEADACHE.value,
        }

        return list(symptoms.intersection(soft_symptoms))

    def build_training_dataframe(
        self,
        diary_entries: list[dict[str, Any]],
        weather_by_entry_id: dict[str, dict[str, Any]],
        user_category: str = "GENERAL",
    ) -> pd.DataFrame:
        rows = []

        for entry in diary_entries:
            entry_id = entry.get("id")

            if not entry_id:
                continue

            weather = weather_by_entry_id.get(entry_id)

            if not weather:
                continue

            symptoms = self._normalize_symptoms(entry.get("symptoms", []))

            bad_state = self._is_bad_state(
                entry=entry,
                symptoms=symptoms,
            )

            row = {
                "bad_state": int(bad_state),
                "hour": int(entry.get("hour", 12)),
                "time_of_day_code": self._encode_time_of_day(
                    entry.get("timeOfDay", TimeOfDay.DAY.value)
                ),
                "thunderstorm_probability": float(weather.get("thunderstorm_probability", 0)),
                "cloud_cover": float(weather.get("cloud_cover", 0)),
                "sunshine_duration": float(weather.get("sunshine_duration", 3600)),
                "pollen_index": float(weather.get("pollen_index", 0)),
                "temperature": float(weather.get("temperature", 0)),
                "temperature_delta_6h": float(weather.get("temperature_delta_6h", 0)),
                "temperature_delta_24h": float(weather.get("temperature_delta_24h", 0)),
                "pressure": float(weather.get("pressure", 1013)),
                "pressure_delta_3h": float(weather.get("pressure_delta_3h", 0)),
                "pressure_delta_24h": float(weather.get("pressure_delta_24h", 0)),
                "humidity": float(weather.get("humidity", 50)),
                "wind_speed": float(weather.get("wind_speed", 0)),
                "precipitation": float(weather.get("precipitation", 0)),
                "kp_index": float(weather.get("kp_index", 1)),
                "sleep_quality": float(entry.get("sleepQuality", 3)),
                "stress_score": float(entry.get("stressScore", 3)),
                "water_liters": float(entry.get("waterLiters", 1.0)),
                "caffeine_cups": float(entry.get("caffeineCups", 0)),
            }
            row.update(self._encode_user_category(user_category))
            rows.append(row)

        return pd.DataFrame(rows)

    def train(
        self,
        user_id: str,
        training_df: pd.DataFrame,
        min_entries: int = 10,
    ) -> dict[str, Any]:
        
        if training_df.empty or len(training_df) < min_entries:
            raise ValueError("Empty dataframe")

        missing_columns = [
            column for column in self.feature_columns + ["bad_state"]
            if column not in training_df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        if training_df["bad_state"].nunique() < 2:
            raise ValueError("Need at least 2 classes")

        x = training_df[self.feature_columns]
        y = training_df["bad_state"]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(x, y)

        preds = model.predict(x)
        score = accuracy_score(y, preds)

        model_path = self._model_path(user_id)
        joblib.dump(model, model_path)

        return {
            "modelType": "LogisticRegression",
            "status": "TRAINED",
            "modelPath": str(model_path),
            "entriesCount": len(training_df),
            "lastTrainedAt": datetime.now(timezone.utc).isoformat(),
            "quality": {
                "trainAccuracy": float(score),
            },
        }

    def predict_risk(
        self,
        user_id: str,
        feature_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        model_path = self._model_path(user_id)

        if not model_path.exists():
            raise ValueError("Model not found")

        if not feature_rows:
            return []

        model = joblib.load(model_path)

        df = pd.DataFrame(feature_rows)

        if df.empty:
            return []

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0

        df = df[self.feature_columns].fillna(0)

        probs = model.predict_proba(df)[:, 1]

        result = []

        for row, risk_score in zip(feature_rows, probs):
            reasons = self._detect_risk_reasons(row)

            result.append(
                {
                    "forecastDate": row.get("forecastDate"),
                    "period": {
                        "fromHour": row.get("fromHour"),
                        "toHour": row.get("toHour"),
                        "timeOfDay": row.get("timeOfDay", TimeOfDay.DAY.value),
                    },
                    "riskScore": float(risk_score),
                    "riskLevel": self._risk_level(float(risk_score)),
                    "riskReasons": [
                        reason.value for reason in reasons
                    ],
                    "predictedSymptoms": self._predict_symptoms_by_reasons(
                        risk_score=float(risk_score),
                        reasons=reasons,
                        row=row,
                    ),
                    "source": "ML_MODEL",
                }
            )

        return result

    def _encode_time_of_day(self, value: str) -> int:
        mapping = {
            TimeOfDay.MORNING.value: 0,
            TimeOfDay.DAY.value: 1,
            TimeOfDay.EVENING.value: 2,
            TimeOfDay.NIGHT.value: 3,
        }

        return mapping.get(value, 1)

    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.7:
            return "HIGH"

        if risk_score >= 0.4:
            return "MEDIUM"

        return "LOW"

    def _normalize_symptoms(self, symptoms: list[Any]) -> list[str]:
        result = []

        for item in symptoms:
            if isinstance(item, SymptomType):
                result.append(item.value)
            else:
                result.append(str(item))

        return result

    def _detect_risk_reasons(
        self,
        row: dict[str, Any],
    ) -> list[RiskReason]:
        reasons = []

        pressure_delta_24h = float(row.get("pressure_delta_24h", 0))
        pressure_delta_3h = float(row.get("pressure_delta_3h", 0))

        temp_delta_24h = float(row.get("temperature_delta_24h", 0))
        temp_delta_6h = float(row.get("temperature_delta_6h", 0))

        humidity = float(row.get("humidity", 50))
        wind_speed = float(row.get("wind_speed", 0))
        precipitation = float(row.get("precipitation", 0))
        kp_index = float(row.get("kp_index", 0))
        temperature = float(row.get("temperature", 20))

        thunderstorm_probability = float(row.get("thunderstorm_probability", 0))
        cloud_cover = float(row.get("cloud_cover", 0))
        sunshine_duration = float(row.get("sunshine_duration", 3600))
        pollen_index = float(row.get("pollen_index", 0))

        is_hypotonic = int(row.get("category_hypotonic", 0)) == 1
        is_hypertonic = int(row.get("category_hypertonic", 0)) == 1
        is_joint = int(row.get("category_joint_disease", 0)) == 1
        is_migraine = int(row.get("category_migraine", 0)) == 1

        def add(reason: RiskReason) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if pressure_delta_24h <= -7 or pressure_delta_3h <= -4:
            add(RiskReason.PRESSURE_DROP)

        if pressure_delta_24h >= 7 or pressure_delta_3h >= 4:
            add(RiskReason.PRESSURE_RISE)

        if abs(pressure_delta_24h) >= 7 or abs(pressure_delta_3h) >= 4:
            add(RiskReason.PRESSURE_SWING)

        if temp_delta_24h <= -7 or temp_delta_6h <= -5:
            add(RiskReason.TEMPERATURE_DROP)

        if temp_delta_24h >= 7 or temp_delta_6h >= 5:
            add(RiskReason.TEMPERATURE_RISE)

        if abs(temp_delta_24h) >= 7 or abs(temp_delta_6h) >= 5:
            add(RiskReason.TEMPERATURE_SWING)

        if humidity >= 80:
            add(RiskReason.HIGH_HUMIDITY)

        if humidity <= 25:
            add(RiskReason.LOW_HUMIDITY)

        if wind_speed >= 10:
            add(RiskReason.STRONG_WIND)

        if precipitation > 0:
            add(RiskReason.PRECIPITATION)

        if thunderstorm_probability >= 50:
            add(RiskReason.THUNDERSTORM)

        if kp_index >= 5:
            add(RiskReason.GEOMAGNETIC_STORM)

        if cloud_cover >= 85 or sunshine_duration <= 1800:
            add(RiskReason.LOW_SUNLIGHT)

        if temperature >= 30:
            add(RiskReason.HEAT_STRESS)

        if temperature <= -10:
            add(RiskReason.COLD_STRESS)

        if pollen_index >= 3:
            add(RiskReason.HIGH_POLLEN)

        if is_hypotonic:
            if pressure_delta_24h <= -5 or pressure_delta_3h <= -3:
                add(RiskReason.PRESSURE_DROP)

            if humidity >= 75:
                add(RiskReason.HIGH_HUMIDITY)

            if cloud_cover >= 75 or sunshine_duration <= 2400:
                add(RiskReason.LOW_SUNLIGHT)

        if is_hypertonic:
            if abs(pressure_delta_24h) >= 5 or abs(pressure_delta_3h) >= 3:
                add(RiskReason.PRESSURE_SWING)

            if temperature >= 27:
                add(RiskReason.HEAT_STRESS)

            if kp_index >= 4:
                add(RiskReason.GEOMAGNETIC_STORM)

        if is_joint:
            if humidity >= 70:
                add(RiskReason.HIGH_HUMIDITY)

            if temp_delta_24h <= -5 or temp_delta_6h <= -3:
                add(RiskReason.TEMPERATURE_DROP)

            if precipitation > 0:
                add(RiskReason.PRECIPITATION)

            if temperature <= 5:
                add(RiskReason.COLD_STRESS)

        if is_migraine:
            if wind_speed >= 7:
                add(RiskReason.STRONG_WIND)

            if abs(pressure_delta_24h) >= 5 or abs(pressure_delta_3h) >= 3:
                add(RiskReason.PRESSURE_SWING)

            if kp_index >= 4:
                add(RiskReason.GEOMAGNETIC_STORM)

            if thunderstorm_probability >= 40:
                add(RiskReason.THUNDERSTORM)

        return reasons
    
    def _encode_user_category(self, category: str | None) -> dict[str, int]:
        category = category or "GENERAL"

        return {
            "category_hypotonic": int(category == "HYPOTONIC"),
            "category_hypertonic": int(category == "HYPERTONIC"),
            "category_joint_disease": int(category == "JOINT_DISEASE"),
            "category_migraine": int(category == "MIGRAINE"),
            "category_general": int(category == "GENERAL"),
        }