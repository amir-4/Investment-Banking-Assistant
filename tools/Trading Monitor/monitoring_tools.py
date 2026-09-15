"""
Trading Monitor Agent tools.

IMPORTANT:
- These tools are read-only.
- They do not place, cancel, resize, reroute, release, or modify orders.
- Alpha Vantage is suitable only for prototype/testing usage.
- It must not be treated as an authoritative low-latency production market-data feed.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import requests

from ibm_watsonx_orchestrate.agent_builder.tools import tool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

REQUEST_TIMEOUT_SECONDS = 15

ALLOWED_INTRADAY_INTERVALS = {
    "1min",
    "5min",
    "15min",
    "30min",
    "60min",
}


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _error(
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    """
    Return a consistent structured error.

    The tools intentionally return errors instead of substituting fake values
    such as zero for missing market data.
    """
    result: dict[str, Any] = {
        "status": "error",
        "error_code": code,
        "error": message,
        "observation_timestamp": _utc_now_iso(),
    }

    result.update(details)
    return result


def _success(
    data: dict[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    """Return a consistent structured success response."""
    result: dict[str, Any] = {
        "status": "success",
        "observation_timestamp": _utc_now_iso(),
    }

    result.update(data)
    result.update(metadata)

    return result


def _require_api_key() -> dict[str, Any] | None:
    """Validate that the Alpha Vantage API key exists."""
    if not ALPHA_VANTAGE_API_KEY:
        return _error(
            "MISSING_API_KEY",
            (
                "ALPHA_VANTAGE_API_KEY is not configured. "
                "Market data cannot be verified."
            ),
            data_verified=False,
            market_data_verified=False,
        )

    return None


def _validate_symbol(symbol: str) -> str | None:
    """Validate and normalize a market symbol."""
    if not isinstance(symbol, str):
        return "Symbol must be a string."

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        return "Symbol cannot be empty."

    if len(normalized_symbol) > 20:
        return "Symbol is too long."

    return None


def _safe_float(
    value: Any,
    field_name: str,
) -> tuple[float | None, str | None]:
    """
    Convert a value to float without silently converting missing data to zero.
    """
    if value is None:
        return None, f"Missing required numeric field: {field_name}."

    if isinstance(value, str) and not value.strip():
        return None, f"Empty required numeric field: {field_name}."

    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        return None, f"Invalid numeric value for field: {field_name}."

    if not math.isfinite(parsed_value):
        return None, f"Non-finite numeric value for field: {field_name}."

    return parsed_value, None


def _safe_positive_float(
    value: Any,
    field_name: str,
) -> tuple[float | None, str | None]:
    """Convert a value to a strictly positive float."""
    parsed_value, error_message = _safe_float(value, field_name)

    if error_message:
        return None, error_message

    if parsed_value is None or parsed_value <= 0:
        return None, f"{field_name} must be greater than zero."

    return parsed_value, None


def _safe_non_negative_float(
    value: Any,
    field_name: str,
) -> tuple[float | None, str | None]:
    """Convert a value to a non-negative float."""
    parsed_value, error_message = _safe_float(value, field_name)

    if error_message:
        return None, error_message

    if parsed_value is None or parsed_value < 0:
        return None, f"{field_name} cannot be negative."

    return parsed_value, None


def _parse_iso_datetime(
    value: Any,
    field_name: str,
) -> tuple[datetime | None, str | None]:
    """
    Parse an ISO-8601 datetime.

    A timezone is required to avoid ambiguous scheduling calculations.
    """
    if not isinstance(value, str) or not value.strip():
        return None, f"{field_name} must be a non-empty ISO-8601 string."

    try:
        parsed_datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, f"{field_name} is not a valid ISO-8601 datetime."

    if parsed_datetime.tzinfo is None:
        return None, f"{field_name} must include a timezone."

    return parsed_datetime, None


def _extract_alpha_vantage_error(data: Any) -> str | None:
    """Extract Alpha Vantage API error or rate-limit messages."""
    if not isinstance(data, dict):
        return "Provider returned a non-object response."

    if data.get("Error Message"):
        return str(data["Error Message"])

    if data.get("Note"):
        return str(data["Note"])

    if data.get("Information"):
        return str(data["Information"])

    return None


def _request_alpha_vantage(
    params: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Execute a read-only Alpha Vantage request.

    Returns:
        (data, error)
    """
    api_key_error = _require_api_key()

    if api_key_error:
        return None, api_key_error

    request_params = dict(params)
    request_params["apikey"] = ALPHA_VANTAGE_API_KEY

    try:
        response = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params=request_params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.Timeout:
        return None, _error(
            "MARKET_DATA_TIMEOUT",
            "Market-data provider request timed out.",
            data_verified=False,
            market_data_verified=False,
        )

    except requests.RequestException as exc:
        return None, _error(
            "MARKET_DATA_REQUEST_FAILED",
            f"Market-data provider request failed: {exc}",
            data_verified=False,
            market_data_verified=False,
        )

    try:
        data = response.json()
    except ValueError:
        return None, _error(
            "INVALID_PROVIDER_RESPONSE",
            "Market-data provider returned invalid JSON.",
            data_verified=False,
            market_data_verified=False,
        )

    provider_error = _extract_alpha_vantage_error(data)

    if provider_error:
        return None, _error(
            "PROVIDER_ERROR",
            provider_error,
            data_verified=False,
            market_data_verified=False,
        )

    return data, None


def _validate_fill(fill: Any, index: int) -> tuple[dict[str, float] | None, str | None]:
    """Validate one execution fill."""
    if not isinstance(fill, dict):
        return None, f"Fill at index {index} must be an object."

    price, price_error = _safe_positive_float(
        fill.get("price"),
        f"fills[{index}].price",
    )

    if price_error:
        return None, price_error

    quantity, quantity_error = _safe_positive_float(
        fill.get("quantity"),
        f"fills[{index}].quantity",
    )

    if quantity_error:
        return None, quantity_error

    return {
        "price": float(price),
        "quantity": float(quantity),
    }, None


def _validate_market_bar(
    bar: Any,
    index: int,
) -> tuple[dict[str, float] | None, str | None]:
    """Validate one market bar."""
    if not isinstance(bar, dict):
        return None, f"Market bar at index {index} must be an object."

    price, price_error = _safe_positive_float(
        bar.get("price"),
        f"market_bars[{index}].price",
    )

    if price_error:
        return None, price_error

    volume, volume_error = _safe_positive_float(
        bar.get("volume"),
        f"market_bars[{index}].volume",
    )

    if volume_error:
        return None, volume_error

    return {
        "price": float(price),
        "volume": float(volume),
    }, None


# ---------------------------------------------------------------------------
# Market-data tools
# ---------------------------------------------------------------------------

@tool
def get_quote(symbol: str) -> dict[str, Any]:
    """
    Retrieve a quote for a symbol.

    This tool is intended for prototype/testing use only.

    It fails closed when:
    - API key is missing
    - provider returns an error
    - required quote fields are missing
    - numeric values are malformed
    - provider data cannot be verified

    It never converts missing values to zero.
    """
    symbol_error = _validate_symbol(symbol)

    if symbol_error:
        return _error(
            "INVALID_SYMBOL",
            symbol_error,
            data_verified=False,
            market_data_verified=False,
        )

    normalized_symbol = symbol.strip().upper()

    data, request_error = _request_alpha_vantage(
        {
            "function": "GLOBAL_QUOTE",
            "symbol": normalized_symbol,
        }
    )

    if request_error:
        return request_error

    if not data:
        return _error(
            "EMPTY_PROVIDER_RESPONSE",
            "Provider returned an empty response.",
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    quote = data.get("Global Quote")

    if not isinstance(quote, dict) or not quote:
        return _error(
            "MISSING_QUOTE_DATA",
            "Provider did not return Global Quote data.",
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    required_fields = {
        "05. price": "price",
        "06. volume": "volume",
        "08. previous close": "previous_close",
        "09. change": "change",
        "10. change percent": "change_percent",
        "07. latest trading day": "latest_trading_day",
    }

    missing_fields = [
        provider_field
        for provider_field in required_fields
        if provider_field not in quote
        or quote.get(provider_field) in (None, "")
    ]

    if missing_fields:
        return _error(
            "INCOMPLETE_QUOTE_DATA",
            (
                "Required quote fields are missing. "
                "No substitute values were used."
            ),
            symbol=normalized_symbol,
            missing_fields=missing_fields,
            data_verified=False,
            market_data_verified=False,
        )

    price, price_error = _safe_positive_float(
        quote.get("05. price"),
        "price",
    )

    if price_error:
        return _error(
            "INVALID_QUOTE_PRICE",
            price_error,
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    volume, volume_error = _safe_non_negative_float(
        quote.get("06. volume"),
        "volume",
    )

    if volume_error:
        return _error(
            "INVALID_QUOTE_VOLUME",
            volume_error,
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    previous_close, previous_close_error = _safe_positive_float(
        quote.get("08. previous close"),
        "previous_close",
    )

    if previous_close_error:
        return _error(
            "INVALID_PREVIOUS_CLOSE",
            previous_close_error,
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    change, change_error = _safe_float(
        quote.get("09. change"),
        "change",
    )

    if change_error:
        return _error(
            "INVALID_QUOTE_CHANGE",
            change_error,
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    change_percent_raw = str(quote.get("10. change percent", "")).strip()
    change_percent_clean = change_percent_raw.replace("%", "").strip()

    change_percent, change_percent_error = _safe_float(
        change_percent_clean,
        "change_percent",
    )

    if change_percent_error:
        return _error(
            "INVALID_CHANGE_PERCENT",
            change_percent_error,
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    latest_trading_day = str(quote.get("07. latest trading day")).strip()

    if not latest_trading_day:
        return _error(
            "MISSING_TRADING_DAY",
            "Latest trading day is missing.",
            symbol=normalized_symbol,
            data_verified=False,
            market_data_verified=False,
        )

    return _success(
        {
            "symbol": normalized_symbol,
            "price": price,
            "volume": volume,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "latest_trading_day": latest_trading_day,
        },
        data_source="Alpha Vantage",
        data_status="provider_response_received",
        delayed_or_realtime_status="not independently verified",
        freshness_status="not independently verified",
        market_data_verified=False,
        prototype_only=True,
        limitation=(
            "Alpha Vantage data freshness, latency, and real-time status "
            "must be independently verified before any production use."
        ),
    )


@tool
def get_intraday_series(
    symbol: str,
    interval: str = "5min",
) -> dict[str, Any]:
    """
    Retrieve intraday OHLCV data.

    Supported intervals:
    - 1min
    - 5min
    - 15min
    - 30min
    - 60min

    The tool fails closed for incomplete or malformed bars.
    """
    symbol_error = _validate_symbol(symbol)

    if symbol_error:
        return _error(
            "INVALID_SYMBOL",
            symbol_error,
            data_verified=False,
            market_data_verified=False,
        )

    if not isinstance(interval, str):
        return _error(
            "INVALID_INTERVAL",
            "Interval must be a string.",
            data_verified=False,
            market_data_verified=False,
        )

    normalized_interval = interval.strip().lower()

    if normalized_interval not in ALLOWED_INTRADAY_INTERVALS:
        return _error(
            "UNSUPPORTED_INTERVAL",
            (
                "Unsupported interval. Allowed intervals are: "
                f"{sorted(ALLOWED_INTRADAY_INTERVALS)}"
            ),
            allowed_intervals=sorted(ALLOWED_INTRADAY_INTERVALS),
            data_verified=False,
            market_data_verified=False,
        )

    normalized_symbol = symbol.strip().upper()

    data, request_error = _request_alpha_vantage(
        {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "outputsize": "compact",
        }
    )

    if request_error:
        return request_error

    series_key = f"Time Series ({normalized_interval})"
    raw_series = data.get(series_key) if data else None

    if not isinstance(raw_series, dict) or not raw_series:
        return _error(
            "MISSING_INTRADAY_DATA",
            f"Provider did not return {series_key}.",
            symbol=normalized_symbol,
            interval=normalized_interval,
            data_verified=False,
            market_data_verified=False,
        )

    required_fields = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    }

    parsed_series: dict[str, dict[str, float]] = {}
    invalid_bars: list[dict[str, Any]] = []

    for timestamp, raw_bar in raw_series.items():
        if not isinstance(raw_bar, dict):
            invalid_bars.append(
                {
                    "timestamp": timestamp,
                    "reason": "Bar is not an object.",
                }
            )
            continue

        missing_fields = [
            provider_field
            for provider_field in required_fields
            if provider_field not in raw_bar
            or raw_bar.get(provider_field) in (None, "")
        ]

        if missing_fields:
            invalid_bars.append(
                {
                    "timestamp": timestamp,
                    "reason": "Missing required fields.",
                    "missing_fields": missing_fields,
                }
            )
            continue

        open_price, open_error = _safe_positive_float(
            raw_bar.get("1. open"),
            f"{timestamp}.open",
        )
        high_price, high_error = _safe_positive_float(
            raw_bar.get("2. high"),
            f"{timestamp}.high",
        )
        low_price, low_error = _safe_positive_float(
            raw_bar.get("3. low"),
            f"{timestamp}.low",
        )
        close_price, close_error = _safe_positive_float(
            raw_bar.get("4. close"),
            f"{timestamp}.close",
        )
        volume, volume_error = _safe_non_negative_float(
            raw_bar.get("5. volume"),
            f"{timestamp}.volume",
        )

        errors = [
            error
            for error in (
                open_error,
                high_error,
                low_error,
                close_error,
                volume_error,
            )
            if error
        ]

        if errors:
            invalid_bars.append(
                {
                    "timestamp": timestamp,
                    "reason": errors,
                }
            )
            continue

        if high_price < max(open_price, close_price, low_price):
            invalid_bars.append(
                {
                    "timestamp": timestamp,
                    "reason": "High price is inconsistent with OHLC values.",
                }
            )
            continue

        if low_price > min(open_price, close_price, high_price):
            invalid_bars.append(
                {
                    "timestamp": timestamp,
                    "reason": "Low price is inconsistent with OHLC values.",
                }
            )
            continue

        parsed_series[timestamp] = {
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }

    if invalid_bars:
        return _error(
            "INVALID_INTRADAY_DATA",
            (
                "One or more intraday bars were missing, malformed, "
                "or internally inconsistent."
            ),
            symbol=normalized_symbol,
            interval=normalized_interval,
            invalid_bars=invalid_bars[:20],
            invalid_bar_count=len(invalid_bars),
            data_verified=False,
            market_data_verified=False,
        )

    if not parsed_series:
        return _error(
            "NO_VALID_INTRADAY_BARS",
            "No valid intraday bars were returned.",
            symbol=normalized_symbol,
            interval=normalized_interval,
            data_verified=False,
            market_data_verified=False,
        )

    return _success(
        {
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "bars": parsed_series,
            "bar_count": len(parsed_series),
        },
        data_source="Alpha Vantage",
        data_status="provider_response_received",
        delayed_or_realtime_status="not independently verified",
        freshness_status="not independently verified",
        market_data_verified=False,
        prototype_only=True,
        limitation=(
            "The tool does not independently verify exchange timestamps, "
            "latency, feed entitlement, or real-time status."
        ),
    )


# ---------------------------------------------------------------------------
# Execution analytics tools
# ---------------------------------------------------------------------------

@tool
def calculate_vwap_progress(
    fills: list[dict[str, Any]],
    market_bars: list[dict[str, Any]],
    side: str | None = None,
) -> dict[str, Any]:
    """
    Calculate execution VWAP versus market VWAP.

    Important:
    - This is a mathematical comparison only.
    - It does not include fees, commissions, spread, or market impact.
    - If side is provided, the tool gives a directional interpretation.
    - If side is omitted, the variance is reported neutrally.
    """
    if not isinstance(fills, list) or not fills:
        return _error(
            "INVALID_FILLS",
            "fills must be a non-empty list.",
        )

    if not isinstance(market_bars, list) or not market_bars:
        return _error(
            "INVALID_MARKET_BARS",
            "market_bars must be a non-empty list.",
        )

    normalized_side: str | None = None

    if side is not None:
        if not isinstance(side, str):
            return _error(
                "INVALID_SIDE",
                "side must be either BUY, SELL, or omitted.",
            )

        normalized_side = side.strip().upper()

        if normalized_side not in {"BUY", "SELL"}:
            return _error(
                "INVALID_SIDE",
                "side must be either BUY, SELL, or omitted.",
            )

    validated_fills: list[dict[str, float]] = []

    for index, fill in enumerate(fills):
        validated_fill, validation_error = _validate_fill(fill, index)

        if validation_error:
            return _error(
                "INVALID_FILL",
                validation_error,
                invalid_index=index,
            )

        validated_fills.append(validated_fill)

    validated_market_bars: list[dict[str, float]] = []

    for index, bar in enumerate(market_bars):
        validated_bar, validation_error = _validate_market_bar(bar, index)

        if validation_error:
            return _error(
                "INVALID_MARKET_BAR",
                validation_error,
                invalid_index=index,
            )

        validated_market_bars.append(validated_bar)

    filled_quantity = sum(
        fill["quantity"]
        for fill in validated_fills
    )

    market_volume = sum(
        bar["volume"]
        for bar in validated_market_bars
    )

    if filled_quantity <= 0:
        return _error(
            "ZERO_FILLED_QUANTITY",
            "Total filled quantity must be greater than zero.",
        )

    if market_volume <= 0:
        return _error(
            "ZERO_MARKET_VOLUME",
            "Total market volume must be greater than zero.",
        )

    own_vwap = (
        sum(fill["price"] * fill["quantity"] for fill in validated_fills)
        / filled_quantity
    )

    market_vwap = (
        sum(bar["price"] * bar["volume"] for bar in validated_market_bars)
        / market_volume
    )

    if market_vwap <= 0:
        return _error(
            "INVALID_MARKET_VWAP",
            "Market VWAP must be greater than zero.",
        )

    variance_vs_market = (own_vwap - market_vwap) / market_vwap

    directional_interpretation = (
        "neutral_comparison_only"
    )

    if normalized_side == "BUY":
        if variance_vs_market > 0:
            directional_interpretation = (
                "unfavorable_for_buy_order: execution_price_above_market_vwap"
            )
        elif variance_vs_market < 0:
            directional_interpretation = (
                "favorable_for_buy_order: execution_price_below_market_vwap"
            )
        else:
            directional_interpretation = "at_market_vwap"

    elif normalized_side == "SELL":
        if variance_vs_market > 0:
            directional_interpretation = (
                "favorable_for_sell_order: execution_price_above_market_vwap"
            )
        elif variance_vs_market < 0:
            directional_interpretation = (
                "unfavorable_for_sell_order: execution_price_below_market_vwap"
            )
        else:
            directional_interpretation = "at_market_vwap"

    return _success(
        {
            "own_vwap": round(own_vwap, 8),
            "market_vwap": round(market_vwap, 8),
            "variance_vs_market": round(variance_vs_market, 8),
            "variance_percent": round(variance_vs_market * 100, 6),
            "filled_quantity": filled_quantity,
            "market_volume": market_volume,
            "side": normalized_side,
            "directional_interpretation": directional_interpretation,
        },
        calculation="VWAP comparison",
        fees_included=False,
        commissions_included=False,
        spread_included=False,
        market_impact_included=False,
        limitation=(
            "This calculation excludes fees, commissions, spread, "
            "market impact, and implementation-shortfall components."
        ),
    )


@tool
def calculate_twap_schedule(
    total_size: float,
    start_time: str,
    end_time: str,
    interval_minutes: int,
) -> dict[str, Any]:
    """
    Calculate a deterministic TWAP schedule.

    The number of intervals uses math.ceil() so the requested time window
    is not under-covered by rounding down.
    """
    total_size_value, size_error = _safe_positive_float(
        total_size,
        "total_size",
    )

    if size_error:
        return _error(
            "INVALID_TOTAL_SIZE",
            size_error,
        )

    start_datetime, start_error = _parse_iso_datetime(
        start_time,
        "start_time",
    )

    if start_error:
        return _error(
            "INVALID_START_TIME",
            start_error,
        )

    end_datetime, end_error = _parse_iso_datetime(
        end_time,
        "end_time",
    )

    if end_error:
        return _error(
            "INVALID_END_TIME",
            end_error,
        )

    if end_datetime <= start_datetime:
        return _error(
            "INVALID_TIME_WINDOW",
            "end_time must be later than start_time.",
        )

    if not isinstance(interval_minutes, int) or isinstance(
        interval_minutes,
        bool,
    ):
        return _error(
            "INVALID_INTERVAL_MINUTES",
            "interval_minutes must be an integer.",
        )

    if interval_minutes <= 0:
        return _error(
            "INVALID_INTERVAL_MINUTES",
            "interval_minutes must be greater than zero.",
        )

    duration_seconds = (
        end_datetime - start_datetime
    ).total_seconds()

    duration_minutes = duration_seconds / 60

    number_of_intervals = max(
        1,
        math.ceil(duration_minutes / interval_minutes),
    )

    slice_size = total_size_value / number_of_intervals

    return _success(
        {
            "total_size": total_size_value,
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat(),
            "duration_minutes": duration_minutes,
            "interval_minutes": interval_minutes,
            "number_of_intervals": number_of_intervals,
            "slice_size": slice_size,
        },
        calculation="TWAP schedule calculation",
        execution_authorization=False,
        order_submission=False,
        limitation=(
            "This tool calculates a theoretical schedule only. "
            "It does not submit or modify any order."
        ),
    )


# ---------------------------------------------------------------------------
# Market-status and risk-control tools
# ---------------------------------------------------------------------------

@tool
def check_luld_band(
    reference_price: float,
    last_price: float,
    tier: int,
    price_above_3: bool,
    is_final_25_min: bool = False,
) -> dict[str, Any]:
    """
    Perform a mathematical LULD price-band comparison.

    IMPORTANT:
    This tool does NOT independently confirm:
    - official Limit State
    - official Straddle State
    - official trading pause
    - official exchange status

    Official status requires an authoritative market-status source.
    """
    reference_price_value, reference_error = _safe_positive_float(
        reference_price,
        "reference_price",
    )

    if reference_error:
        return _error(
            "INVALID_REFERENCE_PRICE",
            reference_error,
        )

    last_price_value, last_price_error = _safe_positive_float(
        last_price,
        "last_price",
    )

    if last_price_error:
        return _error(
            "INVALID_LAST_PRICE",
            last_price_error,
        )

    if tier not in {1, 2}:
        return _error(
            "INVALID_LULD_TIER",
            "tier must be either 1 or 2.",
        )

    if not isinstance(price_above_3, bool):
        return _error(
            "INVALID_PRICE_BUCKET",
            "price_above_3 must be boolean.",
        )

    if not isinstance(is_final_25_min, bool):
        return _error(
            "INVALID_SESSION_FLAG",
            "is_final_25_min must be boolean.",
        )

    if tier == 1:
        band_percentage = 0.05
    else:
        band_percentage = 0.10 if price_above_3 else 0.20

    if is_final_25_min and (
        tier == 1 or (tier == 2 and not price_above_3)
    ):
        band_percentage *= 2

    lower_band = reference_price_value * (1 - band_percentage)
    upper_band = reference_price_value * (1 + band_percentage)

    within_band = (
        lower_band <= last_price_value <= upper_band
    )

    return _success(
        {
            "reference_price": reference_price_value,
            "last_price": last_price_value,
            "tier": tier,
            "price_above_3": price_above_3,
            "is_final_25_min": is_final_25_min,
            "band_percentage": band_percentage,
            "band_percentage_display": f"{band_percentage * 100:.2f}%",
            "lower_band": lower_band,
            "upper_band": upper_band,
            "within_band": within_band,
            "official_halt_verified": False,
            "official_limit_state_verified": False,
            "official_straddle_state_verified": False,
        },
        calculation="Mathematical LULD band comparison only",
        market_status_source=None,
        limitation=(
            "A mathematical band comparison does not independently confirm "
            "an official Limit State, Straddle State, trading pause, or "
            "exchange halt. Verify official status using an authoritative "
            "market-status source."
        ),
    )


@tool
def check_risk_threshold(
    current_exposure: float,
    pre_set_limit: float,
    metric_name: str,
) -> dict[str, Any]:
    """
    Compare one exposure metric against one configured limit.

    This is only a deterministic threshold check.

    It is NOT a complete implementation of SEC Rule 15c3-5.
    """
    exposure_value, exposure_error = _safe_non_negative_float(
        current_exposure,
        "current_exposure",
    )

    if exposure_error:
        return _error(
            "INVALID_CURRENT_EXPOSURE",
            exposure_error,
        )

    limit_value, limit_error = _safe_positive_float(
        pre_set_limit,
        "pre_set_limit",
    )

    if limit_error:
        return _error(
            "INVALID_PRE_SET_LIMIT",
            limit_error,
        )

    if not isinstance(metric_name, str) or not metric_name.strip():
        return _error(
            "INVALID_METRIC_NAME",
            "metric_name must be a non-empty string.",
        )

    normalized_metric_name = metric_name.strip()

    utilization = exposure_value / limit_value
    utilization_percentage = utilization * 100

    breached = utilization >= 1.0
    near_limit = 0.90 <= utilization < 1.0

    if breached:
        status = "BREACHED"
    elif near_limit:
        status = "NEAR_LIMIT"
    else:
        status = "WITHIN_LIMIT"

    return _success(
        {
            "metric_name": normalized_metric_name,
            "current_exposure": exposure_value,
            "pre_set_limit": limit_value,
            "utilization": utilization,
            "utilization_percentage": utilization_percentage,
            "breached": breached,
            "near_limit": near_limit,
            "status": status,
        },
        calculation="Single configured threshold comparison",
        rule_15c3_5_complete_implementation=False,
        execution_authorization=False,
        order_modification_authorization=False,
        limitation=(
            "This tool checks only one configured threshold. "
            "It is not a complete implementation of SEC Rule 15c3-5 "
            "or a complete market-access risk-control framework."
        ),
    )