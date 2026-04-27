from application.services.infrastructure.open_meteo.client import OpenMeteoClient
from application.services.infrastructure.firebase.client import FirebaseClient
from application.services.infrastructure.ml.personal_model_service import PersonalModelService
from application.services.infrastructure.weather_prediction_center.client import WeatherPredictionCenterClient

class Container:
    def __init__(self):
        self.open_meteo_client = OpenMeteoClient()
        self.firebase_client = FirebaseClient()
        self.ml_service = PersonalModelService()
        self.weather_prediction_center_client = WeatherPredictionCenterClient()

container = Container()