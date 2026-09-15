"""
Governance tools for ib_supervisor_agent.

IMPORTANT:
- Audit logging and human approval workflow integrations are placeholders.
- Configure the internal APIs before production deployment.
- An unavailable audit or approval system must never be interpreted
  as successful governance.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from ibm_watsonx_orchestrate.agent_builder.tools import tool


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INTERNAL_AUDIT_LOG_BASE = os.environ.get(
    "INTERNAL_AUDIT_LOG_BASE_URL",
    "",
).strip().rstrip("/")

INTERNAL_APPROVAL_WORKFLOW_BASE = os.environ.get(
    "INTERNAL_APPROVAL_WORKFLOW_BASE_URL",
    "",
).strip().rstrip("/")

REQUEST_TIMEOUT_SECONDS = 15

ALLOWED_EVENT_TYPES = {
    "request_received",
    "collaborator_invoked",
    "response_delivered",
    "response_blocked_pending_approval",
}

ALLOWED_APPROVAL_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "unavailable",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _error(
    code: str,
    message: str,
    *,
    details: Any = None,
) -> dict:
    """
    Return a structured, non-sensitive error response.
    """
    result = {
        "ok": False,
        "error_code": code,
        "message": message,
        "timestamp_utc": _utc_now_iso(),
    }

    if details is not None:
        result["details"] = details

    return result


def _success(
    data: dict,
    *,
    warning: str = "",
) -> dict:
    """
    Return a structured success response.
    """
    result = {
        "ok": True,
        "timestamp_utc": _utc_now_iso(),
        **data,
    }

    if warning:
        result["warning"] = warning

    return result


def _require_text(
    value: str,
    field_name: str,
    *,
    max_length: int = 2000,
) -> str | None:
    """
    Validate a required text field.
    Returns an error message or None when valid.
    """
    if not isinstance(value, str):
        return f"{field_name} must be a string."

    value = value.strip()

    if not value:
        return f"{field_name} must not be empty."

    if len(value) > max_length:
        return (
            f"{field_name} exceeds the maximum allowed length "
            f"of {max_length} characters."
        )

    return None


def _require_bool(value: bool, field_name: str) -> str | None:
    """
    Validate a strict boolean.
    """
    if not isinstance(value, bool):
        return f"{field_name} must be a boolean."

    return None


def _safe_response_json(response: requests.Response) -> dict | list:
    """
    Parse a JSON response safely.
    """
    try:
        payload = response.json()
    except ValueError:
        raise ValueError("The internal service returned invalid JSON.")

    if not isinstance(payload, (dict, list)):
        raise ValueError(
            "The internal service returned an unsupported JSON structure."
        )

    return payload


# ---------------------------------------------------------------------
# Tool 1: Audit logging
# ---------------------------------------------------------------------

@tool
def log_audit_event(
    event_type: str,
    actor: str,
    summary: str,
    related_agent: str = "",
) -> dict:
    """
    Create an audit event for Supervisor governance activity.

    Valid event types:
    - request_received
    - collaborator_invoked
    - response_delivered
    - response_blocked_pending_approval

    If the internal audit service is not configured, a local event is
    returned with an explicit warning. This is NOT durable audit logging.
    """

    validation_error = _require_text(
        event_type,
        "event_type",
        max_length=100,
    )
    if validation_error:
        return _error("INVALID_EVENT_TYPE", validation_error)

    if event_type not in ALLOWED_EVENT_TYPES:
        return _error(
            "UNSUPPORTED_EVENT_TYPE",
            (
                f"event_type must be one of: "
                f"{', '.join(sorted(ALLOWED_EVENT_TYPES))}."
            ),
        )

    validation_error = _require_text(
        actor,
        "actor",
        max_length=300,
    )
    if validation_error:
        return _error("INVALID_ACTOR", validation_error)

    validation_error = _require_text(
        summary,
        "summary",
        max_length=4000,
    )
    if validation_error:
        return _error("INVALID_SUMMARY", validation_error)

    if not isinstance(related_agent, str):
        return _error(
            "INVALID_RELATED_AGENT",
            "related_agent must be a string.",
        )

    related_agent = related_agent.strip()

    if len(related_agent) > 300:
        return _error(
            "INVALID_RELATED_AGENT",
            "related_agent exceeds 300 characters.",
        )

    record = {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now_iso(),
        "event_type": event_type,
        "actor": actor.strip(),
        "related_agent": related_agent,
        "summary": summary.strip(),
        "source": "ib_supervisor_agent",
    }

    # Local fallback: explicit governance warning.
    if not INTERNAL_AUDIT_LOG_BASE:
        return _success(
            {
                "audit_persisted": False,
                "audit_record": record,
            },
            warning=(
                "INTERNAL_AUDIT_LOG_BASE_URL is not configured. "
                "This event was generated locally and was NOT persisted "
                "to a durable audit system. Production deployment requires "
                "a real audit-log integration."
            ),
        )

    try:
        response = requests.post(
            f"{INTERNAL_AUDIT_LOG_BASE}/events",
            json=record,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _success(
            {
                "audit_persisted": False,
                "audit_record": record,
            },
            warning=(
                "The audit service could not be reached. "
                "The event was not durably persisted. "
                f"Technical detail: {type(exc).__name__}."
            ),
        )

    if not 200 <= response.status_code < 300:
        return _success(
            {
                "audit_persisted": False,
                "audit_record": record,
                "http_status": response.status_code,
            },
            warning=(
                "The audit service returned a non-success status. "
                "The event was not confirmed as durably persisted."
            ),
        )

    return _success(
        {
            "audit_persisted": True,
            "audit_record": record,
            "http_status": response.status_code,
        },
    )


# ---------------------------------------------------------------------
# Tool 2: Human approval request
# ---------------------------------------------------------------------

@tool
def request_human_approval(
    request_summary: str,
    sensitivity_reason: str,
    requested_by: str,
) -> dict:
    """
    Submit a sensitive output for human approval.

    The Supervisor must treat all statuses other than exactly
    "approved" as not approved.

    Possible statuses:
    - pending
    - approved
    - rejected
    - unavailable
    """

    validation_error = _require_text(
        request_summary,
        "request_summary",
        max_length=4000,
    )
    if validation_error:
        return _error("INVALID_REQUEST_SUMMARY", validation_error)

    validation_error = _require_text(
        sensitivity_reason,
        "sensitivity_reason",
        max_length=500,
    )
    if validation_error:
        return _error("INVALID_SENSITIVITY_REASON", validation_error)

    validation_error = _require_text(
        requested_by,
        "requested_by",
        max_length=300,
    )
    if validation_error:
        return _error("INVALID_REQUESTED_BY", validation_error)

    # Fail closed if approval workflow is not configured.
    if not INTERNAL_APPROVAL_WORKFLOW_BASE:
        return _success(
            {
                "approval_required": True,
                "request_id": None,
                "status": "unavailable",
                "approved_for_delivery": False,
            },
            warning=(
                "INTERNAL_APPROVAL_WORKFLOW_BASE_URL is not configured. "
                "This request has NOT been approved. "
                "The sensitive output must remain blocked."
            ),
        )

    payload = {
        "request_id": str(uuid.uuid4()),
        "request_summary": request_summary.strip(),
        "sensitivity_reason": sensitivity_reason.strip(),
        "requested_by": requested_by.strip(),
        "requested_at_utc": _utc_now_iso(),
        "source": "ib_supervisor_agent",
    }

    try:
        response = requests.post(
            f"{INTERNAL_APPROVAL_WORKFLOW_BASE}/requests",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _success(
            {
                "approval_required": True,
                "request_id": payload["request_id"],
                "status": "unavailable",
                "approved_for_delivery": False,
            },
            warning=(
                "The approval workflow could not be reached. "
                "The sensitive output remains blocked. "
                f"Technical detail: {type(exc).__name__}."
            ),
        )

    if not 200 <= response.status_code < 300:
        return _success(
            {
                "approval_required": True,
                "request_id": payload["request_id"],
                "status": "unavailable",
                "approved_for_delivery": False,
                "http_status": response.status_code,
            },
            warning=(
                "The approval workflow returned a non-success status. "
                "Approval was not confirmed. The sensitive output remains blocked."
            ),
        )

    try:
        provider_payload = _safe_response_json(response)
    except ValueError as exc:
        return _success(
            {
                "approval_required": True,
                "request_id": payload["request_id"],
                "status": "unavailable",
                "approved_for_delivery": False,
            },
            warning=str(exc),
        )

    if not isinstance(provider_payload, dict):
        return _success(
            {
                "approval_required": True,
                "request_id": payload["request_id"],
                "status": "unavailable",
                "approved_for_delivery": False,
            },
            warning=(
                "The approval workflow returned a JSON list instead of "
                "an approval object. Approval was not confirmed."
            ),
        )

    raw_status = provider_payload.get("status", "unavailable")

    if not isinstance(raw_status, str):
        raw_status = "unavailable"

    status = raw_status.strip().lower()

    if status not in ALLOWED_APPROVAL_STATUSES:
        status = "unavailable"

    return _success(
        {
            "approval_required": True,
            "request_id": provider_payload.get(
                "request_id",
                payload["request_id"],
            ),
            "status": status,
            "approved_for_delivery": status == "approved",
            "provider_response": provider_payload,
        },
    )


# ---------------------------------------------------------------------
# Tool 3: Pure approval-policy logic
# ---------------------------------------------------------------------

@tool
def check_required_approval(
    involves_trade_commitment: bool = False,
    involves_valuation_finalization: bool = False,
    involves_compliance_finding: bool = False,
    involves_risk_limit_exception: bool = False,
) -> dict:
    """
    Determine whether human approval is required.

    Approval is required when at least one of these four triggers is true:

    1. Trade commitment or execution instruction.
    2. Final valuation or investment recommendation.
    3. Compliance or AML finding.
    4. Risk-limit or concentration-limit exception/waiver.

    This function does not call an external system.
    """

    boolean_inputs = {
        "involves_trade_commitment": involves_trade_commitment,
        "involves_valuation_finalization": involves_valuation_finalization,
        "involves_compliance_finding": involves_compliance_finding,
        "involves_risk_limit_exception": involves_risk_limit_exception,
    }

    for field_name, value in boolean_inputs.items():
        validation_error = _require_bool(value, field_name)

        if validation_error:
            return _error(
                "INVALID_APPROVAL_FLAG",
                validation_error,
            )

    triggers = []

    if involves_trade_commitment:
        triggers.append("trade_commitment")

    if involves_valuation_finalization:
        triggers.append("valuation_finalization")

    if involves_compliance_finding:
        triggers.append("compliance_finding")

    if involves_risk_limit_exception:
        triggers.append("risk_limit_exception")

    approval_required = bool(triggers)

    return _success(
        {
            "approval_required": approval_required,
            "approved_for_delivery": not approval_required,
            "triggers": triggers,
            "governance_decision": (
                "HUMAN_APPROVAL_REQUIRED"
                if approval_required
                else "NO_RULE_3_TRIGGER_DETECTED"
            ),
        },
    )