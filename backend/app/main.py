"""KMRL Document Intelligence — FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import PROJECT_ROOT, settings
from .database import init_db
from .routers import documents, system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Condense, contextualise and route KMRL's document flow: upload, "
        "OCR/extraction, AI analysis, and a searchable document library."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)


@app.get(f"{settings.api_prefix}", include_in_schema=False)
def api_root() -> JSONResponse:
    return JSONResponse(
        {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                f"{settings.api_prefix}/health",
                f"{settings.api_prefix}/documents",
                f"{settings.api_prefix}/documents/search",
                f"{settings.api_prefix}/documents/facets",
                f"{settings.api_prefix}/documents/filters",
                f"{settings.api_prefix}/documents/stats",
                f"{settings.api_prefix}/documents/upload",
            ],
        }
    )


# --- serve the built single-page app (when `npm run build` has been run) ----
if FRONTEND_DIST.exists():
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")

else:  # pragma: no cover - developer convenience

    @app.get("/", include_in_schema=False)
    def root_placeholder() -> JSONResponse:
        return JSONResponse(
            {
                "message": (
                    "Frontend build not found. Run `npm install && npm run build` in "
                    "./frontend, or start the Vite dev server with `npm run dev`."
                ),
                "api": f"{settings.api_prefix}",
                "docs": "/docs",
            }
        )
