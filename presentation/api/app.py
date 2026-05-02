import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from presentation.api.middleware import register_middlewares
from presentation.api.routes.predict import router as predict_router
from presentation.api.routes.update_diary import router as update_diary_router
from presentation.api.routes.health import router as health_router
from apscheduler.schedulers.background import BackgroundScheduler
from container import container


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        scheduler.shutdown()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(
        title="Weather Health API",
        version="1.0.0",
        description="API для прогноза самочувствия метеозависимых пользователей",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_middlewares(app)

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(update_diary_router)

    register_error_handlers(app)

    scheduler = BackgroundScheduler(timezone="Europe/Moscow")

    scheduler.add_job(
        container.notification_pipeline_service.run_for_all_users,
        trigger="cron",
        hour="7,10,19,22",
        minute=0,
        id="notification_pipeline",
        replace_existing=True,
    )

    scheduler.start()
    
    return app


def register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(400)
    async def bad_request(request: Request, exc):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(exc)},
        )

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Endpoint not found"},
        )

    @app.exception_handler(500)
    async def internal_error(request: Request, exc):
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )