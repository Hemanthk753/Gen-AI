"""FastAPI router for the AI agent chat endpoint.

Registers POST /api/v1/genai/agent/chat
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File, Form
from typing import Optional

from src.modules.agent.models.schemas import AgentChatRequest, AgentChatResponse
from src.modules.agent.orchestrator import handle_message
from src.modules.agent.services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/agent/chat",
    name="AI Agent Chat",
    response_model=AgentChatResponse,
    description="Send a message to the AI experiment assistant",
)
async def agent_chat(
    request: AgentChatRequest,
    authorization: str = Header(..., description="Bearer token"),
) -> AgentChatResponse:
    """Main chat endpoint for the AI agent.

    The frontend sends user messages here. The agent:
    1. Loads user context (profile, reference data, experiment history)
    2. Routes to the correct flow (design, measurement, update)
    3. Calls specialist GPT models as needed
    4. Calls backend APIs to create/update experiments
    5. Returns structured responses for the frontend to render
    """
    auth_token = authorization.replace("Bearer ", "").strip()
    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    return await handle_message(request, auth_token=auth_token)


@router.post(
    "/agent/upload",
    name="AI Agent File Upload",
    response_model=dict,
    description="Upload a file for the current agent conversation",
)
async def agent_upload(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    authorization: str = Header(..., description="Bearer token"),
    upload_type: str = Form("design"),
) -> dict:
    """Upload a file during an agent conversation.

    This delegates to POST /experiments/upload_file/{literal}.
    The frontend must send the file here along with the conversation_id.
    The agent looks up the experiment_literal from conversation state.
    """
    from src.modules.agent.orchestrator import _states

    auth_token = authorization.replace("Bearer ", "").strip()
    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    state = _states.get(conversation_id)
    if not state or not state.collected.experiment_literal:
        raise HTTPException(status_code=400, detail="No active experiment in this conversation")

    literal = state.collected.experiment_literal
    api = APIClient(auth_token=auth_token)

    file_bytes = await file.read()
    result = await api.upload_file(
        literal=literal,
        file_bytes=file_bytes,
        filename=file.filename or "upload.csv",
        upload_type=upload_type,
    )

    # Store file info in state
    state.collected.file_name = file.filename
    state.collected.upload_type = upload_type
    state.awaiting_upload = False
    state.current_step = 5  # Move to data processing step

    return {"status": "uploaded", "filename": file.filename, "result": result}