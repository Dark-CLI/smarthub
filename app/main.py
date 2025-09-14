from fastapi import FastAPI
from app.routes_chat import router as chat_router
from app.routes_admin import router as admin_router
from utils.logging import configure_logging

def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="SmartHub", version="0.1.0")
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
    app.include_router(admin_router, tags=["admin"])
    return app

app = create_app()
