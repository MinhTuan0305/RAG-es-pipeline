"""
Write the API's OpenAPI schema to frontend/openapi.json (no server, no models needed),
for generating the frontend's TypeScript types:

    python backend/scripts/export_openapi.py
    cd frontend && npm run gen:api
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))  # make package "app" (backend/app) importable

from app.main import create_app  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "frontend" / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
