import math
import os
import statistics
from datetime import datetime, timezone
from typing import Any

import requests
from ibm_watsonx_orchestrate.agent_builder.tools import tool


# ============================================================
# Configuration
# ============================================================

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Configure this only after connecting the firm's real system.
INTERNAL_RISK_SYSTEM_BASE_URL = os.environ.get(
    "INTERNAL_RISK_SYSTEM_BASE_URL",
    "",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 15
MIN_HISTORICAL_OBSERVATIONS = 30

SUPPORTED_CONFIDENCE_LEVELS = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.999: 3.0902,
}

SUPPORTED_FRED_SERIES = {
    "VIXCLS": "CBOE Volatility Index",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
}


# ============================================================
# Helpers
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(
    code: str,
    message: str,
    **details: Any,
) -> dict:
    response = {
        "status": "error",
        "error_code": code,
        "message": message,
        "as_of_utc": _utc_now_iso(),
    }

    if details:
        response["details"] = details

    return response


def _success(**data: Any) -> dict:
    return {
        "status": "success",
        "as_of_utc": _utc_now_iso(),
        **data,
    }


def _validate_non_empty_string(
    value: Any,
    field_name: str,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty string."

    return None


def _validate_probability(
    value: Any,
    field_name: str = "confidence_level",
) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{field_name} must be numeric."

    if not 0 < float(value) < 1:
        return f"{field_name} must be between 0 and 1."

    return None


def _validate_positive_number(
    value: Any,
    field_name: str,
) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{field_name} must be numeric."

    if not math.isfinite(float(value)) or float(value) <= 0:
        return f"{field_name} must be a finite number greater than zero."

    return None


def _validate_non_negative_number(
    value: Any,
    field_name: str,
) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{field_name} must be numeric."

    if not math.isfinite(float(value)) or float(value) < 0:
        return f"{field_name} must be a finite number greater than or equal to zero."

    return None


def _validate_holding_period(
    holding_period_days: Any,
) -> str | None:
    if (
        not isinstance(holding_period_days, int)
        or isinstance(holding_period_days, bool)
        or holding_period_days <= 0
    ):
        return "holding_period_days must be a positive integer."

    return None


def _validate_confidence_and_holding_period(
    confidence_level: float,
    holding_period_days: int,
) -> dict | None:
    confidence_error = _validate_probability(confidence_level)

    if confidence_error:
        return _error(
            "INVALID_CONFIDENCE_LEVEL",
            confidence_error,
        )

    holding_error = _validate_holding_period(holding_period_days)

    if holding_error:
        return _error(
            "INVALID_HOLDING_PERIOD",
            holding_error,
        )

    return None


def _validate_position(
    position: Any,
    index: int,
) -> str | None:
    if not isinstance(position, dict):
        return f"Position at index {index} must be an object."

    for field in ("instrument", "sector", "market_value"):
        if field not in position:
            return (
                f"Position at index {index} is missing required field "
                f"'{field}'."
            )

    if not isinstance(position["instrument"], str) or not position["instrument"].strip():
        return f"Position at index {index} has an invalid instrument."

    if not isinstance(position["sector"], str) or not position["sector"].strip():
        return f"Position at index {index} has an invalid sector."

    market_value_error = _validate_non_negative_number(
        position["market_value"],
        f"positions[{index}].market_value",
    )

    if market_value_error:
        return market_value_error

    return None


def _validate_positions(
    positions: Any,
) -> dict | None:
    if not isinstance(positions, list):
        return _error(
            "INVALID_POSITIONS",
            "positions must be a list of position objects.",
        )

    if not positions:
        return _error(
            "EMPTY_POSITIONS",
            "positions is empty; no stress test can be performed.",
        )

    for index, position in enumerate(positions):
        validation_error = _validate_position(position, index)

        if validation_error:
            return _error(
                "INVALID_POSITION",
                validation_error,
            )

    return None


def _fred_api_error(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    return _error(
        "FRED_API_ERROR",
        "FRED returned an unsuccessful response.",
        http_status=response.status_code,
        provider_error=payload.get("error_message"),
    )


# ============================================================
# Tool 1 — Internal portfolio positions
# ============================================================

@tool
def get_portfolio_positions(book_id: str) -> dict:
    """
    Retrieve current positions for a firm's internal book.

    This tool is intentionally fail-closed. It does not fabricate or
    reconstruct positions when the internal risk system is not connected.
    """

    validation_error = _validate_non_empty_string(book_id, "book_id")

    if validation_error:
        return _error(
            "INVALID_BOOK_ID",
            validation_error,
        )

    if not INTERNAL_RISK_SYSTEM_BASE_URL:
        return _error(
            "INTERNAL_RISK_SYSTEM_NOT_CONFIGURED",
            (
                "INTERNAL_RISK_SYSTEM_BASE_URL is not configured. "
                "Connect this tool to the firm's actual position/risk "
                "system before using portfolio-dependent calculations. "
                "No positions were fabricated or assumed."
            ),
            book_id=book_id,
        )

    endpoint = (
        f"{INTERNAL_RISK_SYSTEM_BASE_URL}/books/"
        f"{book_id}/positions"
    )

    try:
        response = requests.get(
            endpoint,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _error(
            "INTERNAL_RISK_SYSTEM_REQUEST_FAILED",
            "Could not reach the internal risk system.",
            book_id=book_id,
            reason=str(exc),
        )

    if not response.ok:
        return _error(
            "INTERNAL_RISK_SYSTEM_HTTP_ERROR",
            "The internal risk system returned an unsuccessful response.",
            book_id=book_id,
            http_status=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError:
        return _error(
            "INVALID_INTERNAL_RISK_RESPONSE",
            "The internal risk system returned invalid JSON.",
            book_id=book_id,
        )

    if not isinstance(payload, dict):
        return _error(
            "INVALID_INTERNAL_RISK_RESPONSE",
            "The internal risk system response must be a JSON object.",
            book_id=book_id,
        )

    return _success(
        book_id=book_id,
        source="internal_position_risk_system",
        verified=True,
        data=payload,
    )


# ============================================================
# Tool 2 — Historical simulation VaR
# ============================================================

@tool
def calculate_historical_var(
    daily_pnl_history: list[float],
    confidence_level: float = 0.99,
    holding_period_days: int = 1,
) -> dict:
    """
    Calculate historical-simulation VaR from historical daily P&L.

    P&L values should use the convention:
    - positive = profit
    - negative = loss

    VaR is returned as a positive loss amount.
    """

    if not isinstance(daily_pnl_history, list):
        return _error(
            "INVALID_PNL_HISTORY",
            "daily_pnl_history must be a list of numeric P&L values.",
        )

    if not daily_pnl_history:
        return _error(
            "EMPTY_PNL_HISTORY",
            "daily_pnl_history is empty; VaR cannot be calculated.",
        )

    if len(daily_pnl_history) < MIN_HISTORICAL_OBSERVATIONS:
        return _error(
            "INSUFFICIENT_PNL_HISTORY",
            (
                "Insufficient historical observations for a meaningful "
                "historical VaR estimate."
            ),
            observations=len(daily_pnl_history),
            minimum_required=MIN_HISTORICAL_OBSERVATIONS,
        )

    for index, pnl in enumerate(daily_pnl_history):
        if (
            not isinstance(pnl, (int, float))
            or isinstance(pnl, bool)
            or not math.isfinite(float(pnl))
        ):
            return _error(
                "INVALID_PNL_VALUE",
                f"daily_pnl_history[{index}] must be a finite number.",
            )

    validation_error = _validate_confidence_and_holding_period(
        confidence_level,
        holding_period_days,
    )

    if validation_error:
        return validation_error

    sorted_pnl = sorted(float(value) for value in daily_pnl_history)
    sample_size = len(sorted_pnl)

    raw_index = (1.0 - confidence_level) * sample_size
    index = max(
        0,
        min(
            int(math.floor(raw_index)),
            sample_size - 1,
        ),
    )

    selected_pnl = sorted_pnl[index]
    var_1day = max(0.0, -selected_pnl)
    scaling_factor = math.sqrt(holding_period_days)
    var_scaled = var_1day * scaling_factor

    return _success(
        method="historical_simulation",
        confidence_level=confidence_level,
        holding_period_days=holding_period_days,
        sample_size=sample_size,
        selected_empirical_pnl=selected_pnl,
        var_1day=var_1day,
        var_scaled_to_holding_period=var_scaled,
        units="same_currency_units_as_input_pnl",
        caveats=[
            (
                "Historical VaR reflects only scenarios represented in "
                "the supplied historical sample."
            ),
            (
                "It does not describe losses beyond the selected "
                "confidence level."
            ),
            (
                "Square-root-of-time scaling is an approximation and "
                "may be inappropriate for nonlinear or path-dependent "
                "portfolios."
            ),
            (
                "The quality of the result depends on the quality, "
                "length, and representativeness of the P&L history."
            ),
        ],
    )


# ============================================================
# Tool 3 — Parametric VaR
# ============================================================

@tool
def calculate_parametric_var(
    portfolio_value: float,
    daily_volatility_pct: float,
    confidence_level: float = 0.99,
    holding_period_days: int = 1,
) -> dict:
    """
    Calculate variance-covariance/parametric VaR.

    daily_volatility_pct must be supplied as a decimal:
    0.012 means 1.2% daily volatility.
    """

    portfolio_error = _validate_positive_number(
        portfolio_value,
        "portfolio_value",
    )

    if portfolio_error:
        return _error(
            "INVALID_PORTFOLIO_VALUE",
            portfolio_error,
        )

    volatility_error = _validate_non_negative_number(
        daily_volatility_pct,
        "daily_volatility_pct",
    )

    if volatility_error:
        return _error(
            "INVALID_DAILY_VOLATILITY",
            volatility_error,
        )

    validation_error = _validate_confidence_and_holding_period(
        confidence_level,
        holding_period_days,
    )

    if validation_error:
        return validation_error

    rounded_confidence = round(float(confidence_level), 3)
    z_score = SUPPORTED_CONFIDENCE_LEVELS.get(rounded_confidence)

    if z_score is None:
        return _error(
            "UNSUPPORTED_CONFIDENCE_LEVEL",
            (
                "No standard z-score is configured for this confidence "
                "level."
            ),
            requested_confidence_level=confidence_level,
            supported_confidence_levels=sorted(
                SUPPORTED_CONFIDENCE_LEVELS.keys()
            ),
        )

    var_1day = (
        float(portfolio_value)
        * float(daily_volatility_pct)
        * z_score
    )

    var_scaled = var_1day * math.sqrt(holding_period_days)

    return _success(
        method="parametric_variance_covariance",
        confidence_level=confidence_level,
        holding_period_days=holding_period_days,
        portfolio_value=portfolio_value,
        daily_volatility_pct=daily_volatility_pct,
        z_score=z_score,
        var_1day=var_1day,
        var_scaled_to_holding_period=var_scaled,
        units="same_currency_units_as_portfolio_value",
        caveats=[
            (
                "This method assumes normally distributed returns or "
                "a normal approximation."
            ),
            (
                "It assumes the volatility estimate is appropriate for "
                "the selected portfolio and period."
            ),
            (
                "It may understate risk during stressed markets with "
                "fat-tailed or skewed returns."
            ),
            (
                "VaR is not a hard ceiling on possible loss."
            ),
            (
                "Square-root-of-time scaling is an approximation."
            ),
        ],
    )


# ============================================================
# Tool 4 — Stress testing
# ============================================================

@tool
def run_stress_test(
    positions: list[dict],
    shock_pct_by_sector: dict[str, float],
) -> dict:
    """
    Apply caller-supplied sector shocks to positions.

    Example:
    {
        "technology": -0.20,
        "energy": -0.10
    }

    A missing sector shock is not silently treated as a real scenario.
    It is listed under sectors_with_no_shock_defined.
    """

    positions_error = _validate_positions(positions)

    if positions_error:
        return positions_error

    if not isinstance(shock_pct_by_sector, dict):
        return _error(
            "INVALID_SHOCK_MAPPING",
            "shock_pct_by_sector must be a mapping of sector to decimal shock.",
        )

    for sector, shock in shock_pct_by_sector.items():
        if not isinstance(sector, str) or not sector.strip():
            return _error(
                "INVALID_SHOCK_SECTOR",
                "Every shock mapping key must be a non-empty sector string.",
            )

        if (
            not isinstance(shock, (int, float))
            or isinstance(shock, bool)
            or not math.isfinite(float(shock))
        ):
            return _error(
                "INVALID_SHOCK_VALUE",
                f"Shock for sector '{sector}' must be a finite number.",
            )

        if abs(float(shock)) > 1:
            return _error(
                "UNREALISTIC_SHOCK_VALUE",
                (
                    f"Shock for sector '{sector}' is outside the supported "
                    "range of -100% to +100%."
                ),
                shock=shock,
            )

    position_level_results = []
    unshocked_sectors = set()
    total_pnl_impact = 0.0

    for position in positions:
        instrument = position["instrument"]
        sector = position["sector"]
        market_value = float(position["market_value"])

        shock = shock_pct_by_sector.get(sector)

        if shock is None:
            unshocked_sectors.add(sector)
            pnl_impact = 0.0
        else:
            pnl_impact = market_value * float(shock)

        total_pnl_impact += pnl_impact

        position_level_results.append(
            {
                "instrument": instrument,
                "sector": sector,
                "market_value": market_value,
                "shock_applied_pct": shock,
                "pnl_impact": pnl_impact,
                "shock_status": (
                    "applied"
                    if shock is not None
                    else "no_shock_defined"
                ),
            }
        )

    return _success(
        scenario_type="sector_shock",
        shock_pct_by_sector=shock_pct_by_sector,
        position_level_impact=position_level_results,
        total_pnl_impact=total_pnl_impact,
        sectors_with_no_shock_defined=sorted(unshocked_sectors),
        caveats=[
            (
                "Only caller-supplied shocks were applied."
            ),
            (
                "No correlation, liquidity, second-order, or "
                "cross-sector contagion effects were modeled."
            ),
            (
                "A sector without a supplied shock was not assumed "
                "to have zero economic risk; it was only left unshocked "
                "for this scenario."
            ),
        ],
    )


# ============================================================
# Tool 5 — Concentration limit checker
# ============================================================

@tool
def check_concentration_limit(
    exposure_by_bucket: dict[str, float],
    total_portfolio_value: float,
    limit_pct: float,
) -> dict:
    """
    Check bucket exposures against a hard, pre-approved limit.

    limit_pct is a decimal:
    0.10 means a 10% concentration limit.

    The agent reports breaches and near-breaches only.
    It never changes or approves the limit.
    """

    if not isinstance(exposure_by_bucket, dict):
        return _error(
            "INVALID_EXPOSURE_MAPPING",
            "exposure_by_bucket must be a mapping of bucket to exposure.",
        )

    portfolio_error = _validate_positive_number(
        total_portfolio_value,
        "total_portfolio_value",
    )

    if portfolio_error:
        return _error(
            "INVALID_TOTAL_PORTFOLIO_VALUE",
            portfolio_error,
        )

    limit_error = _validate_probability(
        limit_pct,
        "limit_pct",
    )

    if limit_error:
        return _error(
            "INVALID_LIMIT_PCT",
            limit_error,
        )

    for bucket, exposure in exposure_by_bucket.items():
        if not isinstance(bucket, str) or not bucket.strip():
            return _error(
                "INVALID_BUCKET",
                "Every exposure bucket must be a non-empty string.",
            )

        exposure_error = _validate_non_negative_number(
            exposure,
            f"exposure_by_bucket[{bucket}]",
        )

        if exposure_error:
            return _error(
                "INVALID_EXPOSURE",
                exposure_error,
            )

    results = []

    for bucket, exposure in exposure_by_bucket.items():
        concentration_pct = (
            float(exposure) / float(total_portfolio_value)
        )

        breached = concentration_pct >= float(limit_pct)
        near_limit = (
            not breached
            and concentration_pct >= float(limit_pct) * 0.90
        )

        if breached:
            status = "BREACHED"
        elif near_limit:
            status = "NEAR_LIMIT"
        else:
            status = "WITHIN_LIMIT"

        results.append(
            {
                "bucket": bucket,
                "exposure": exposure,
                "concentration_pct": concentration_pct,
                "limit_pct": limit_pct,
                "threshold_basis": "approved_pre_set_limit",
                "breached": breached,
                "near_limit": near_limit,
                "status": status,
            }
        )

    breached_buckets = [
        item["bucket"]
        for item in results
        if item["breached"]
    ]

    near_limit_buckets = [
        item["bucket"]
        for item in results
        if item["near_limit"]
    ]

    return _success(
        results=results,
        breached_buckets=breached_buckets,
        near_limit_buckets=near_limit_buckets,
        human_review_required=bool(
            breached_buckets or near_limit_buckets
        ),
        authority_note=(
            "This tool reports against the supplied limit. "
            "It does not change, waive, approve, or reinterpret the limit."
        ),
    )


# ============================================================
# Tool 6 — FRED macro/volatility indicator
# ============================================================

@tool
def get_macro_indicator(
    series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> dict:
    """
    Retrieve an official FRED macro or volatility series.

    Common series:
    - VIXCLS
    - BAMLH0A0HYM2
    - DGS10
    - DGS2
    """

    validation_error = _validate_non_empty_string(
        series_id,
        "series_id",
    )

    if validation_error:
        return _error(
            "INVALID_SERIES_ID",
            validation_error,
        )

    normalized_series_id = series_id.strip().upper()

    if not FRED_API_KEY:
        return _error(
            "FRED_API_KEY_NOT_CONFIGURED",
            (
                "FRED_API_KEY is not configured. Register for an official "
                "FRED API key and configure it before requesting macro data."
            ),
            series_id=normalized_series_id,
        )

    for field_name, value in (
        ("observation_start", observation_start),
        ("observation_end", observation_end),
    ):
        if value is not None:
            if not isinstance(value, str):
                return _error(
                    "INVALID_DATE",
                    f"{field_name} must use YYYY-MM-DD format.",
                )

            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return _error(
                    "INVALID_DATE",
                    f"{field_name} must use YYYY-MM-DD format.",
                )

    params = {
        "series_id": normalized_series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    if observation_start:
        params["observation_start"] = observation_start

    if observation_end:
        params["observation_end"] = observation_end

    try:
        response = requests.get(
            FRED_BASE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _error(
            "FRED_REQUEST_FAILED",
            "Could not reach the FRED API.",
            series_id=normalized_series_id,
            reason=str(exc),
        )

    if not response.ok:
        return _fred_api_error(response)

    try:
        payload = response.json()
    except ValueError:
        return _error(
            "INVALID_FRED_RESPONSE",
            "FRED returned invalid JSON.",
            series_id=normalized_series_id,
        )

    if payload.get("error_code") or payload.get("error_message"):
        return _error(
            "FRED_PROVIDER_ERROR",
            "FRED returned an API error.",
            series_id=normalized_series_id,
            provider_error=payload.get("error_message"),
        )

    raw_observations = payload.get("observations", [])

    if not isinstance(raw_observations, list) or not raw_observations:
        return _error(
            "NO_FRED_OBSERVATIONS",
            f"No observations returned for series '{normalized_series_id}'.",
            series_id=normalized_series_id,
        )

    observations = []
    missing_value_dates = []

    for observation in raw_observations:
        date_value = observation.get("date")
        raw_value = observation.get("value")

        if raw_value in (None, "", "."):
            missing_value_dates.append(date_value)
            continue

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            return _error(
                "INVALID_FRED_OBSERVATION",
                "FRED returned a non-numeric observation value.",
                series_id=normalized_series_id,
                date=date_value,
                value=raw_value,
            )

        observations.append(
            {
                "date": date_value,
                "value": numeric_value,
            }
        )

    if not observations:
        return _error(
            "NO_VALID_FRED_OBSERVATIONS",
            (
                f"FRED returned no valid numeric observations for "
                f"series '{normalized_series_id}'."
            ),
            series_id=normalized_series_id,
            missing_value_dates=missing_value_dates,
        )

    latest_observation = observations[-1]

    return _success(
        series_id=normalized_series_id,
        series_name=SUPPORTED_FRED_SERIES.get(
            normalized_series_id,
            "FRED series",
        ),
        source="Federal Reserve Economic Data (FRED)",
        official_source=True,
        observations=observations,
        most_recent_observation=latest_observation,
        missing_value_dates=missing_value_dates,
        source_url=FRED_BASE_URL,
        staleness_note=(
            "The caller must compare most_recent_observation.date "
            "with the current date and determine whether the data is stale."
        ),
    )