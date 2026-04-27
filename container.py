from application.services.infrastructure.open_meteo.client import OpenMeteoClient
from application.services.infrastructure.firebase.client import FirebaseClient
from application.services.infrastructure.ml.personal_model_service import PersonalModelService
from application.services.infrastructure.kp_index_client.client import KpIndexClient

class Container:
    def __init__(self):
        self.open_meteo_client = OpenMeteoClient()
        self.firebase_client = FirebaseClient()
        self.ml_service = PersonalModelService()
        self.kp_index_client = KpIndexClient()

container = Container()