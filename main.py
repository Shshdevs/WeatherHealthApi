from presentation.api.app import create_app

app = create_app()

# uvicorn main:app --reload --host 127.0.0.1 --port 4000