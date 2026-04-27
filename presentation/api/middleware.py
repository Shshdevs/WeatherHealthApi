import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        logger.info(
            "%s %s %s %s %.3fs",
            request.client.host if request.client else "-",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        return response


def register_middlewares(app):
    app.add_middleware(RequestLoggingMiddleware)