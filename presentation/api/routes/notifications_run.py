from fastapi import APIRouter
from container import container

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.post("/notifications/run")
def run_notifications_pipeline():
    return container.notification_pipeline_service.run_for_all_users()