from fastapi import FastAPI
from presentation.api.routes.predict import router as predict_router
from presentation.api.routes.update_diary import router as update_diary_router
from presentation.api.routes.health import router as health_router

app = FastAPI(title="Weather Health API")

app.include_router(predict_router)
app.include_router(update_diary_router)
app.include_router(health_router)