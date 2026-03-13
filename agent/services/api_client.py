"""HTTP client wrapping all 24 backend endpoints the agent uses.

Every method maps to an actual backend route verified against:
  - src/api/v1/routes.py (prefix registration)
  - Individual router files (exact path + method)
  - ResponseModel(data=..., message=...) wrapper

All endpoints are under /api/v1 prefix. The backend APIPREFIXES are:
  /auth, /users, /experiment-types, /kpis, /countries, /zones,
  /business-functions, /use-cases, /experiments
"""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx

from src.modules.agent.config import (
    API_BASE_URL,
    API_TIMEOUT,
    TASK_POLL_INTERVAL,
    TASK_POLL_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)


class APIClient:
    """Async HTTP client for the TestOps backend API.

    All methods:
    - Accept an auth_token for Bearer authentication
    - Return the unwrapped 'data' field from ResponseModel
    - Raise on HTTP errors with context
    """

    def __init__(self, auth_token: str, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.get(self._url(path), headers=self.headers, params=params)
            resp.raise_for_status()
            body = resp.json()
            return body.get("data", body)

    async def _post(self, path: str, json_data: Any = None, params: Optional[Dict] = None) -> Any:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.post(
                self._url(path), headers=self.headers, json=json_data, params=params
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("data", body)

    async def _put(self, path: str, json_data: Any = None) -> Any:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.put(self._url(path), headers=self.headers, json=json_data)
            resp.raise_for_status()
            body = resp.json()
            return body.get("data", body)

    async def _post_multipart(self, path: str, file_bytes: bytes, filename: str,
                               form_data: Optional[Dict] = None) -> Any:
        headers = {"Authorization": self.headers["Authorization"]}
        files = {"file": (filename, BytesIO(file_bytes), "application/octet-stream")}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                self._url(path), headers=headers, files=files, data=form_data or {}
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("data", body)

    # ────────────────────────────────────────────────────────────
    # 1-2. AUTH & USER
    # Routes: GET /auth/me, GET /users/{literal}
    # ────────────────────────────────────────────────────────────

    async def get_current_user(self) -> Dict:
        """GET /auth/me → user profile with user_literal."""
        return await self._get("/auth/me")

    async def get_user_details(self, user_literal: str) -> Dict:
        """GET /users/{literal} → full user with zones, countries, business_functions."""
        return await self._get(f"/users/{user_literal}")

    # ────────────────────────────────────────────────────────────
    # 3-8. REFERENCE DATA
    # Routes: GET /experiment-types, /kpis, /countries, /zones,
    #         /business-functions, /use-cases
    # ────────────────────────────────────────────────────────────

    async def list_experiment_types(self) -> List[Dict]:
        """GET /experiment-types → list of {title, experiment_type_literal}."""
        result = await self._get("/experiment-types")
        return result if isinstance(result, list) else result.get("result", [])

    async def list_kpis(self) -> List[Dict]:
        """GET /kpis → list of {name, type, kpi_literal}."""
        result = await self._get("/kpis")
        return result if isinstance(result, list) else result.get("result", [])

    async def list_countries(self) -> List[Dict]:
        """GET /countries → list of {title, country_code, currency_code, country_literal, zone}."""
        result = await self._get("/countries")
        return result if isinstance(result, list) else result.get("result", [])

    async def list_zones(self) -> List[Dict]:
        """GET /zones → list of {title, zone_literal}."""
        result = await self._get("/zones")
        return result if isinstance(result, list) else result.get("result", [])

    async def list_business_functions(self) -> List[Dict]:
        """GET /business-functions → list of {title, business_function_literal}."""
        result = await self._get("/business-functions")
        return result if isinstance(result, list) else result.get("result", [])

    async def list_use_cases(self) -> List[Dict]:
        """GET /use-cases → list of {title, use_case_literal, business_functions}."""
        result = await self._get("/use-cases")
        return result if isinstance(result, list) else result.get("result", [])

    # ────────────────────────────────────────────────────────────
    # 9-12. EXPERIMENTS CRUD
    # Routes: POST /experiments/get-all-experiments,
    #         POST /experiments, PUT /experiments/{literal},
    #         GET /experiments/{literal}
    # ────────────────────────────────────────────────────────────

    async def list_experiments(self, filters: Optional[Dict] = None,
                                page: int = 1, per_page: int = 100) -> Dict:
        """POST /experiments/get-all-experiments → paginated list."""
        return await self._post(
            "/experiments/get-all-experiments",
            json_data=filters,
            params={"page": page, "per_page": per_page, "sort_by": "-created_at"},
        )

    async def create_experiment(self, payload: Dict) -> Dict:
        """POST /experiments → created experiment with experiment_literal."""
        return await self._post("/experiments", json_data=payload)

    async def update_experiment(self, literal: str, payload: Dict) -> Dict:
        """PUT /experiments/{literal} → updated experiment."""
        return await self._put(f"/experiments/{literal}", json_data=payload)

    async def get_experiment(self, literal: str) -> Dict:
        """GET /experiments/{literal} → full experiment details."""
        return await self._get(f"/experiments/{literal}")

    # ────────────────────────────────────────────────────────────
    # 13-15. FILE OPERATIONS
    # Routes: POST /experiments/upload_file/{literal},
    #         POST /experiments/find-data-quality/{literal},
    #         GET /experiments/get-columns-and-preview-data/{literal}
    # ────────────────────────────────────────────────────────────

    async def upload_file(self, literal: str, file_bytes: bytes,
                          filename: str, upload_type: str = "design") -> Dict:
        """POST /experiments/upload_file/{literal} → file upload result."""
        return await self._post_multipart(
            f"/experiments/upload_file/{literal}",
            file_bytes=file_bytes,
            filename=filename,
            form_data={"upload_type": upload_type},
        )

    async def find_data_quality(self, literal: str, payload: Dict) -> Dict:
        """POST /experiments/find-data-quality/{literal} → quality report."""
        return await self._post(f"/experiments/find-data-quality/{literal}", json_data=payload)

    async def get_columns_and_preview(self, literal: str,
                                       table_type: str = "Design") -> Dict:
        """GET /experiments/get-columns-and-preview-data/{literal} → columns + preview."""
        return await self._get(
            f"/experiments/get-columns-and-preview-data/{literal}",
            params={"table_type": table_type},
        )

    # ────────────────────────────────────────────────────────────
    # 16-17. DATA PROCESSING & TASK STATUS
    # Routes: GET /experiments/process-validated-data/{literal}/{upload_type},
    #         GET /experiments/task/status/{task_id}
    # ────────────────────────────────────────────────────────────

    async def process_validated_data(self, literal: str,
                                      upload_type: str = "design") -> Dict:
        """GET /experiments/process-validated-data/{literal}/{upload_type} → {task_id}."""
        return await self._get(f"/experiments/process-validated-data/{literal}/{upload_type}")

    async def get_task_status(self, task_id: str) -> Dict:
        """GET /experiments/task/status/{task_id} → {task_id, status, result}."""
        return await self._get(f"/experiments/task/status/{task_id}")

    async def poll_task_until_done(self, task_id: str) -> Dict:
        """Poll task status until SUCCESS/FAILURE. Returns final status dict."""
        for attempt in range(TASK_POLL_MAX_ATTEMPTS):
            result = await self.get_task_status(task_id)
            status = result.get("status", "UNKNOWN") if isinstance(result, dict) else str(result)
            logger.info(f"Task {task_id} poll #{attempt+1}: {status}")
            if status in ("SUCCESS", "FAILURE", "REVOKED"):
                return result if isinstance(result, dict) else {"status": status}
            await asyncio.sleep(TASK_POLL_INTERVAL)
        return {"status": "TIMEOUT", "task_id": task_id}

    # ────────────────────────────────────────────────────────────
    # 18-20. RCT
    # Routes: GET /experiments/rct/{literal},
    #         GET /experiments/rct/result/{literal},
    #         GET /experiments/rct/download-link/{literal}
    # ────────────────────────────────────────────────────────────

    async def execute_rct(self, literal: str) -> Dict:
        """GET /experiments/rct/{literal} → {task_id}."""
        return await self._get(f"/experiments/rct/{literal}")

    async def get_rct_result(self, literal: str) -> Dict:
        """GET /experiments/rct/result/{literal} → RCT result data."""
        return await self._get(f"/experiments/rct/result/{literal}")

    async def get_rct_download_link(self, literal: str) -> Dict:
        """GET /experiments/rct/download-link/{literal} → {download_link}."""
        return await self._get(f"/experiments/rct/download-link/{literal}")

    # ────────────────────────────────────────────────────────────
    # 21-22. MEASUREMENT
    # Routes: GET /experiments/run-aks-jobs/{literal},
    #         GET /experiments/measurement-design-merge-overview/{literal}
    # ────────────────────────────────────────────────────────────

    async def run_aks_jobs(self, literal: str) -> Dict:
        """GET /experiments/run-aks-jobs/{literal} → {task_id}."""
        return await self._get(f"/experiments/run-aks-jobs/{literal}")

    async def get_merge_overview(self, literal: str) -> Dict:
        """GET /experiments/measurement-design-merge-overview/{literal} → merge data."""
        return await self._get(f"/experiments/measurement-design-merge-overview/{literal}")

    # ────────────────────────────────────────────────────────────
    # 23-24. SAMPLE SIZE & KPI MEASUREMENT
    # Routes: POST /experiments/estimate-sample-size,
    #         GET /experiments/design-kpi-measurement/{literal}
    # ────────────────────────────────────────────────────────────

    async def estimate_sample_size(self, payload: Dict) -> Dict:
        """POST /experiments/estimate-sample-size → {min_test_sample_size, min_control_sample_size}."""
        return await self._post("/experiments/estimate-sample-size", json_data=payload)

    async def get_design_kpi_measurement(self, literal: str) -> Dict:
        """GET /experiments/design-kpi-measurement/{literal} → {mean, deviation, length, min, max}."""
        return await self._get(f"/experiments/design-kpi-measurement/{literal}")