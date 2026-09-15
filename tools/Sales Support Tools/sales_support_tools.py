"""
Sales Support Agent tools.

Purpose:
- Prepare internal meeting briefs for human salespeople.
- Retrieve approved client portfolio information.
- Retrieve company fundamentals.
- Retrieve recent news and sentiment.
- Highlight materially negative news prominently.

IMPORTANT:
- These tools do not contact clients.
- These tools do not send emails.
- These tools do not create client-facing communications.
- These tools do not recommend trades.
- Client holdings must never be fabricated.
- Alpha Vantage is suitable for prototype/pre-meeting research only.
- Alpha Vantage must not be treated as a low-latency production market-data feed.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import requests

from ibm_watsonx_orchestrate.agent_builder.tools import tool


# ============================================================================
# Configuration
# ============================================================================

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

ALPHA_VANTAGE_API_KEY = os.environ.get(
    "ALPHA_VANTAGE_API_KEY",
    "",
).strip()

# This must be replaced with the firm's actual internal CRM/book-of-record API.
INTERNAL_CRM_BASE = os.environ.get(
    "INTERNAL_CRM_BASE_URL",
    "",
).strip().rstrip("/")

REQUEST_TIMEOUT_SECONDS = 15

MAX_NEWS_LIMIT = 1000

SUPPORTED_NEWS_TOPICS = {
    "blockchain",
    "earnings",
    "ipo",
    "mergers_and_acquisitions",
    "financial_markets",
    "economy_fiscal",
    "economy_monetary",
    "economy_macro",
    "energy_transportation",
    "finance",
    "life_sciences",
    "manufacturing",
    "real_estate",
    "retail_wholesale",
    "technology",
}


# ============================================================================
# Internal helper functions
# ============================================================================

def _utc_now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _error(
    error_code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    """
    Return a consistent structured error.

    Errors are explicit and fail closed. No missing information is replaced
    with fabricated holdings, prices, dates, or sentiment values.
    """
    result: dict[str, Any] = {
        "status": "error",
        "error_code": error_code,
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


def _validate_client_id(client_id: str) -> str | None:
    """Validate a client identifier."""
    if not isinstance(client_id, str):
        return "client_id must be a string."

    if not client_id.strip():
        return "client_id cannot be empty."

    if len(client_id.strip()) > 256:
        return "client_id is too long."

    return None


def _validate_symbol(symbol: str) -> str | None:
    """Validate a ticker symbol."""
    if not isinstance(symbol, str):
        return "symbol must be a string."

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        return "symbol cannot be empty."

    if len(normalized_symbol) > 20:
        return "symbol is too long."

    return None


def _normalize_symbol(symbol: str) -> str:
    """Normalize a ticker symbol."""
    return symbol.strip().upper()


def _require_alpha_vantage_key() -> dict[str, Any] | None:
    """Return an explicit error when the market-data API key is missing."""
    if not ALPHA_VANTAGE_API_KEY:
        return _error(
            "MISSING_ALPHA_VANTAGE_API_KEY",
            (
                "ALPHA_VANTAGE_API_KEY is not configured. "
                "Market research cannot be retrieved."
            ),
            data_verified=False,
            market_data_verified=False,
        )

    return None


def _extract_provider_error(data: Any) -> str | None:
    """
    Extract common Alpha Vantage provider errors and rate-limit messages.
    """
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
        data, error
    """
    api_key_error = _require_alpha_vantage_key()

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
            "PROVIDER_TIMEOUT",
            "Alpha Vantage request timed out.",
            data_verified=False,
            market_data_verified=False,
        )

    except requests.RequestException as exc:
        return None, _error(
            "PROVIDER_REQUEST_FAILED",
            f"Alpha Vantage request failed: {exc}",
            data_verified=False,
            market_data_verified=False,
        )

    try:
        data = response.json()
    except ValueError:
        return None, _error(
            "INVALID_PROVIDER_JSON",
            "Alpha Vantage returned invalid JSON.",
            data_verified=False,
            market_data_verified=False,
        )

    provider_error = _extract_provider_error(data)

    if provider_error:
        return None, _error(
            "ALPHA_VANTAGE_PROVIDER_ERROR",
            provider_error,
            data_verified=False,
            market_data_verified=False,
        )

    return data, None


def _clean_optional_text(value: Any) -> str | None:
    """Return normalized text or None for missing values."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _clean_optional_number(value: Any) -> float | None:
    """
    Convert an optional numeric value.

    Missing or malformed values remain None. They are never converted to zero.
    """
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed_value):
        return None

    return parsed_value


def _normalize_sentiment_label(label: Any) -> str:
    """Normalize a sentiment label."""
    if label is None:
        return ""

    return str(label).strip().lower()


def _is_materially_negative_article(article: dict[str, Any]) -> bool:
    """
    Determine whether an article should be prominently flagged.

    This is a conservative rule-based screening layer. Final materiality
    judgment remains with the human salesperson/research professional.
    """
    label = _normalize_sentiment_label(
        article.get("overall_sentiment_label")
    )

    score = _clean_optional_number(
        article.get("overall_sentiment_score")
    )

    negative_labels = {
        "bearish",
        "somewhat-bearish",
        "negative",
        "very bearish",
        "strongly bearish",
    }

    if label in negative_labels:
        return True

    if "bearish" in label or "negative" in label:
        return True

    # Alpha Vantage sentiment scores are generally between -1 and +1.
    # A score at or below -0.35 is treated as a screening flag only.
    if score is not None and score <= -0.35:
        return True

    return False


def _article_relevant_to_ticker(
    article: dict[str, Any],
    ticker: str,
) -> bool:
    """
    Check whether an article has relevance to a particular ticker.

    The article may contain Alpha Vantage ticker_sentiment entries.
    """
    normalized_ticker = ticker.strip().upper()

    ticker_sentiment = article.get("ticker_sentiment", [])

    if not isinstance(ticker_sentiment, list):
        return False

    for item in ticker_sentiment:
        if not isinstance(item, dict):
            continue

        article_ticker = (
            item.get("ticker")
            or item.get("symbol")
            or item.get("ticker_symbol")
        )

        if article_ticker:
            if str(article_ticker).strip().upper() == normalized_ticker:
                return True

    return False


def _validate_news_limit(limit: int) -> str | None:
    """Validate the requested news limit."""
    if not isinstance(limit, int) or isinstance(limit, bool):
        return "limit must be an integer."

    if limit < 1 or limit > MAX_NEWS_LIMIT:
        return f"limit must be between 1 and {MAX_NEWS_LIMIT}."

    return None


def _validate_time_from(time_from: str | None) -> str | None:
    """
    Validate Alpha Vantage NEWS_SENTIMENT time_from format.

    Expected format:
        YYYYMMDDTHHMM
    """
    if time_from is None:
        return None

    if not isinstance(time_from, str):
        return "time_from must be a string."

    value = time_from.strip()

    if not value:
        return "time_from cannot be empty when provided."

    try:
        datetime.strptime(value, "%Y%m%dT%H%M")
    except ValueError:
        return (
            "time_from must use the format YYYYMMDDTHHMM, "
            "for example 20260914T0900."
        )

    return None


def _validate_topics(topics: str | None) -> str | None:
    """Validate Alpha Vantage topic filters."""
    if topics is None:
        return None

    if not isinstance(topics, str):
        return "topics must be a string."

    topic_values = [
        topic.strip().lower()
        for topic in topics.split(",")
        if topic.strip()
    ]

    if not topic_values:
        return "topics cannot be empty when provided."

    unsupported_topics = [
        topic
        for topic in topic_values
        if topic not in SUPPORTED_NEWS_TOPICS
    ]

    if unsupported_topics:
        return _error(
            "UNSUPPORTED_NEWS_TOPIC",
            "One or more news topics are unsupported.",
            unsupported_topics=unsupported_topics,
            supported_topics=sorted(SUPPORTED_NEWS_TOPICS),
        ).get("error")

    return None


# ============================================================================
# CRM / client portfolio tool
# ============================================================================

@tool
def get_client_portfolio_snapshot(
    client_id: str,
) -> dict[str, Any]:
    """
    Retrieve a client's current known holdings from the firm's internal CRM
    or book-of-record system.

    IMPORTANT:
    This function is intentionally a safe placeholder.

    There is no public API for a firm's private client holdings. This tool
    must be connected to the firm's authenticated internal system, such as:
    - Salesforce
    - Internal CRM
    - Portfolio accounting system
    - Order/position management system
    - Client book-of-record platform

    Until INTERNAL_CRM_BASE_URL is configured and the endpoint is implemented,
    this tool returns an explicit error.

    Expected endpoint:
        GET {INTERNAL_CRM_BASE_URL}/clients/{client_id}/holdings

    Expected response shape:
    {
        "client_id": "CLIENT-001",
        "as_of": "2026-09-14T10:30:00Z",
        "holdings": [
            {
                "ticker": "AAPL",
                "quantity": 1000,
                "market_value": 225000.00
            }
        ]
    }

    No holdings are fabricated or reconstructed from assumptions.
    """
    client_id_error = _validate_client_id(client_id)

    if client_id_error:
        return _error(
            "INVALID_CLIENT_ID",
            client_id_error,
            portfolio_verified=False,
        )

    normalized_client_id = client_id.strip()

    if not INTERNAL_CRM_BASE:
        return _error(
            "INTERNAL_CRM_NOT_CONFIGURED",
            (
                "INTERNAL_CRM_BASE_URL is not configured. "
                "The client portfolio tool is only a placeholder and must "
                "be connected to the firm's actual authenticated CRM or "
                "book-of-record system. No client holdings can be fabricated "
                "or assumed."
            ),
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
            production_ready=False,
        )

    endpoint = (
        f"{INTERNAL_CRM_BASE}/clients/"
        f"{normalized_client_id}/holdings"
    )

    try:
        response = requests.get(
            endpoint,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.Timeout:
        return _error(
            "INTERNAL_CRM_TIMEOUT",
            "The internal CRM request timed out.",
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
        )

    except requests.RequestException as exc:
        return _error(
            "INTERNAL_CRM_REQUEST_FAILED",
            f"The internal CRM request failed: {exc}",
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
        )

    try:
        data = response.json()
    except ValueError:
        return _error(
            "INVALID_INTERNAL_CRM_RESPONSE",
            "The internal CRM returned invalid JSON.",
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
        )

    if not isinstance(data, dict):
        return _error(
            "INVALID_PORTFOLIO_RESPONSE",
            "The internal CRM response must be a JSON object.",
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
        )

    returned_client_id = data.get("client_id")

    if returned_client_id is not None:
        if str(returned_client_id).strip() != normalized_client_id:
            return _error(
                "CLIENT_ID_MISMATCH",
                (
                    "The CRM response client_id does not match the "
                    "requested client_id."
                ),
                requested_client_id=normalized_client_id,
                returned_client_id=returned_client_id,
                portfolio_verified=False,
                holdings_available=False,
            )

    as_of = data.get("as_of")

    if not as_of:
        return _error(
            "MISSING_PORTFOLIO_AS_OF",
            (
                "The CRM response does not contain an as_of timestamp. "
                "The portfolio snapshot cannot be treated as current."
            ),
            client_id=normalized_client_id,
            portfolio_verified=False,
            holdings_available=False,
        )

    holdings = data.get("holdings")

    if holdings is None:
        return _error(
            "MISSING_HOLDINGS_FIELD",
            (
                "The CRM response does not contain a holdings field. "
                "No holdings will be assumed."
            ),
            client_id=normalized_client_id,
            as_of=as_of,
            portfolio_verified=False,
            holdings_available=False,
        )

    if not isinstance(holdings, list):
        return _error(
            "INVALID_HOLDINGS_FIELD",
            "The holdings field must be a list.",
            client_id=normalized_client_id,
            as_of=as_of,
            portfolio_verified=False,
            holdings_available=False,
        )

    if not holdings:
        return _error(
            "NO_HOLDINGS_RETURNED",
            (
                "The CRM returned no holdings for this client. "
                "No holdings will be fabricated or inferred."
            ),
            client_id=normalized_client_id,
            as_of=as_of,
            holdings=[],
            portfolio_verified=True,
            holdings_available=False,
        )

    validated_holdings: list[dict[str, Any]] = []

    for index, holding in enumerate(holdings):
        if not isinstance(holding, dict):
            return _error(
                "INVALID_HOLDING_RECORD",
                f"Holding at index {index} must be an object.",
                client_id=normalized_client_id,
                as_of=as_of,
                invalid_index=index,
                portfolio_verified=False,
            )

        ticker = holding.get("ticker")

        if not isinstance(ticker, str) or not ticker.strip():
            return _error(
                "MISSING_HOLDING_TICKER",
                f"Holding at index {index} has no valid ticker.",
                client_id=normalized_client_id,
                as_of=as_of,
                invalid_index=index,
                portfolio_verified=False,
            )

        quantity = _clean_optional_number(holding.get("quantity"))
        market_value = _clean_optional_number(
            holding.get("market_value")
        )

        if quantity is None:
            return _error(
                "INVALID_HOLDING_QUANTITY",
                (
                    f"Holding at index {index} has a missing or invalid "
                    "quantity."
                ),
                client_id=normalized_client_id,
                as_of=as_of,
                invalid_index=index,
                portfolio_verified=False,
            )

        if market_value is None:
            return _error(
                "INVALID_HOLDING_MARKET_VALUE",
                (
                    f"Holding at index {index} has a missing or invalid "
                    "market_value."
                ),
                client_id=normalized_client_id,
                as_of=as_of,
                invalid_index=index,
                portfolio_verified=False,
            )

        validated_holdings.append(
            {
                "ticker": ticker.strip().upper(),
                "quantity": quantity,
                "market_value": market_value,
            }
        )

    return _success(
        {
            "client_id": normalized_client_id,
            "as_of": as_of,
            "holdings": validated_holdings,
            "holding_count": len(validated_holdings),
        },
        data_source="Internal CRM / book-of-record",
        portfolio_verified=True,
        holdings_available=True,
        production_ready=True,
        limitation=(
            "The endpoint and authentication must be implemented according "
            "to the firm's internal CRM security and access-control model."
        ),
    )


# ============================================================================
# Company fundamentals tool
# ============================================================================

@tool
def get_company_overview(
    symbol: str,
) -> dict[str, Any]:
    """
    Retrieve company fundamentals using Alpha Vantage OVERVIEW.

    Returns:
    - Sector
    - Industry
    - Market capitalization
    - P/E ratio
    - Dividend yield
    - 52-week high
    - 52-week low

    Missing values remain None. They are never silently converted to zero.
    """
    symbol_error = _validate_symbol(symbol)

    if symbol_error:
        return _error(
            "INVALID_SYMBOL",
            symbol_error,
            fundamentals_verified=False,
        )

    normalized_symbol = _normalize_symbol(symbol)

    data, request_error = _request_alpha_vantage(
        {
            "function": "OVERVIEW",
            "symbol": normalized_symbol,
        }
    )

    if request_error:
        return request_error

    if not data:
        return _error(
            "EMPTY_OVERVIEW_RESPONSE",
            "Alpha Vantage returned an empty overview response.",
            symbol=normalized_symbol,
            fundamentals_verified=False,
        )

    provider_symbol = data.get("Symbol")

    if not provider_symbol:
        return _error(
            "NO_OVERVIEW_DATA",
            (
                f"No company overview data returned for "
                f"'{normalized_symbol}'."
            ),
            symbol=normalized_symbol,
            fundamentals_verified=False,
        )

    provider_symbol_normalized = str(provider_symbol).strip().upper()

    if provider_symbol_normalized != normalized_symbol:
        return _error(
            "OVERVIEW_SYMBOL_MISMATCH",
            (
                "The provider returned a different symbol than requested."
            ),
            requested_symbol=normalized_symbol,
            returned_symbol=provider_symbol_normalized,
            fundamentals_verified=False,
        )

    overview = {
        "symbol": provider_symbol_normalized,
        "asset_type": _clean_optional_text(data.get("AssetType")),
        "name": _clean_optional_text(data.get("Name")),
        "description": _clean_optional_text(data.get("Description")),
        "exchange": _clean_optional_text(data.get("Exchange")),
        "currency": _clean_optional_text(data.get("Currency")),
        "country": _clean_optional_text(data.get("Country")),
        "sector": _clean_optional_text(data.get("Sector")),
        "industry": _clean_optional_text(data.get("Industry")),
        "market_cap": _clean_optional_number(
            data.get("MarketCapitalization")
        ),
        "ebitda": _clean_optional_number(data.get("EBITDA")),
        "pe_ratio": _clean_optional_number(data.get("PERatio")),
        "peg_ratio": _clean_optional_number(data.get("PEGRatio")),
        "book_value": _clean_optional_number(data.get("BookValue")),
        "dividend_per_share": _clean_optional_number(
            data.get("DividendPerShare")
        ),
        "dividend_yield": _clean_optional_number(
            data.get("DividendYield")
        ),
        "eps": _clean_optional_number(data.get("EPS")),
        "revenue_per_share_ttm": _clean_optional_number(
            data.get("RevenuePerShareTTM")
        ),
        "profit_margin": _clean_optional_number(
            data.get("ProfitMargin")
        ),
        "operating_margin_ttm": _clean_optional_number(
            data.get("OperatingMarginTTM")
        ),
        "return_on_assets_ttm": _clean_optional_number(
            data.get("ReturnOnAssetsTTM")
        ),
        "return_on_equity_ttm": _clean_optional_number(
            data.get("ReturnOnEquityTTM")
        ),
        "revenue_ttm": _clean_optional_number(
            data.get("RevenueTTM")
        ),
        "gross_profit_ttm": _clean_optional_number(
            data.get("GrossProfitTTM")
        ),
        "diluted_eps_ttm": _clean_optional_number(
            data.get("DilutedEPSTTM")
        ),
        "quarterly_earnings_growth_yoy": _clean_optional_number(
            data.get("QuarterlyEarningsGrowthYOY")
        ),
        "quarterly_revenue_growth_yoy": _clean_optional_number(
            data.get("QuarterlyRevenueGrowthYOY")
        ),
        "analyst_target_price": _clean_optional_number(
            data.get("AnalystTargetPrice")
        ),
        "52_week_high": _clean_optional_number(
            data.get("52WeekHigh")
        ),
        "52_week_low": _clean_optional_number(
            data.get("52WeekLow")
        ),
        "50_day_moving_average": _clean_optional_number(
            data.get("50DayMovingAverage")
        ),
        "200_day_moving_average": _clean_optional_number(
            data.get("200DayMovingAverage")
        ),
        "shares_outstanding": _clean_optional_number(
            data.get("SharesOutstanding")
        ),
        "shares_float": _clean_optional_number(
            data.get("SharesFloat")
        ),
        "last_split_factor": _clean_optional_text(
            data.get("LastSplitFactor")
        ),
        "last_split_date": _clean_optional_text(
            data.get("LastSplitDate")
        ),
    }

    return _success(
        {
            "symbol": normalized_symbol,
            "overview": overview,
        },
        data_source="Alpha Vantage OVERVIEW",
        fundamentals_verified=True,
        data_as_of="provider-reported; exact filing/as-of date not supplied",
        delayed_or_realtime_status="not applicable to company fundamentals",
        limitation=(
            "Alpha Vantage OVERVIEW fields may reflect different reporting "
            "periods. Do not describe all fields as being from one identical "
            "filing date unless independently verified."
        ),
    )


# ============================================================================
# News and sentiment tool
# ============================================================================

@tool
def get_sector_news_sentiment(
    tickers: list[str] | None = None,
    topics: str | None = None,
    time_from: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Retrieve recent news and sentiment from Alpha Vantage NEWS_SENTIMENT.

    Args:
        tickers:
            Optional list of ticker symbols, for example ["AAPL", "MSFT"].

        topics:
            Optional comma-separated topics, for example:
            "technology,earnings"

        time_from:
            Optional start time in YYYYMMDDTHHMM format.

        limit:
            Number of articles from 1 to 1000.

    Returns:
        Structured article list with:
        - title
        - URL
        - publication time
        - source
        - overall sentiment label
        - overall sentiment score
        - ticker sentiment
        - negative-news screening flag

    Important:
    Negative-news screening is a prioritization mechanism, not a final
    legal, investment, or materiality determination.
    """
    limit_error = _validate_news_limit(limit)

    if limit_error:
        return _error(
            "INVALID_NEWS_LIMIT",
            limit_error,
        )

    topics_error = _validate_topics(topics)

    if topics_error:
        return _error(
            "INVALID_NEWS_TOPICS",
            topics_error,
        )

    time_from_error = _validate_time_from(time_from)

    if time_from_error:
        return _error(
            "INVALID_TIME_FROM",
            time_from_error,
        )

    normalized_tickers: list[str] = []

    if tickers is not None:
        if not isinstance(tickers, list):
            return _error(
                "INVALID_TICKERS",
                "tickers must be a list of ticker symbols.",
            )

        if not tickers:
            return _error(
                "INVALID_TICKERS",
                "tickers cannot be an empty list.",
            )

        for index, ticker in enumerate(tickers):
            ticker_error = _validate_symbol(ticker)

            if ticker_error:
                return _error(
                    "INVALID_TICKER",
                    ticker_error,
                    invalid_index=index,
                )

            normalized_tickers.append(_normalize_symbol(ticker))

        # Remove duplicates while preserving order.
        normalized_tickers = list(dict.fromkeys(normalized_tickers))

    params: dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "limit": limit,
        "sort": "LATEST",
    }

    if normalized_tickers:
        params["tickers"] = ",".join(normalized_tickers)

    if topics:
        params["topics"] = ",".join(
            topic.strip().lower()
            for topic in topics.split(",")
            if topic.strip()
        )

    if time_from:
        params["time_from"] = time_from.strip()

    data, request_error = _request_alpha_vantage(params)

    if request_error:
        return request_error

    if not data:
        return _error(
            "EMPTY_NEWS_RESPONSE",
            "Alpha Vantage returned an empty news response.",
            news_verified=False,
        )

    feed = data.get("feed")

    if feed is None:
        return _error(
            "MISSING_NEWS_FEED",
            "Alpha Vantage response does not contain a feed field.",
            news_verified=False,
        )

    if not isinstance(feed, list):
        return _error(
            "INVALID_NEWS_FEED",
            "Alpha Vantage news feed must be a list.",
            news_verified=False,
        )

    if not feed:
        return _success(
            {
                "articles": [],
                "article_count": 0,
                "negative_article_count": 0,
            },
            data_source="Alpha Vantage NEWS_SENTIMENT",
            news_verified=True,
            note=(
                "No articles returned for the supplied filters. "
                "No news item was fabricated."
            ),
        )

    articles: list[dict[str, Any]] = []
    malformed_articles: list[dict[str, Any]] = []

    for index, raw_article in enumerate(feed):
        if not isinstance(raw_article, dict):
            malformed_articles.append(
                {
                    "index": index,
                    "reason": "Article is not an object.",
                }
            )
            continue

        title = _clean_optional_text(raw_article.get("title"))
        url = _clean_optional_text(raw_article.get("url"))
        time_published = _clean_optional_text(
            raw_article.get("time_published")
        )
        source = _clean_optional_text(raw_article.get("source"))

        if not title or not time_published:
            malformed_articles.append(
                {
                    "index": index,
                    "reason": (
                        "Article is missing title or publication timestamp."
                    ),
                }
            )
            continue

        overall_sentiment_label = _clean_optional_text(
            raw_article.get("overall_sentiment_label")
        )

        overall_sentiment_score = _clean_optional_number(
            raw_article.get("overall_sentiment_score")
        )

        ticker_sentiment = raw_article.get("ticker_sentiment", [])

        if not isinstance(ticker_sentiment, list):
            ticker_sentiment = []

        article = {
            "title": title,
            "url": url,
            "time_published": time_published,
            "source": source,
            "authors": raw_article.get("authors", []),
            "summary": _clean_optional_text(raw_article.get("summary")),
            "banner_image": _clean_optional_text(
                raw_article.get("banner_image")
            ),
            "overall_sentiment_label": overall_sentiment_label,
            "overall_sentiment_score": overall_sentiment_score,
            "ticker_sentiment": ticker_sentiment,
            "negative_news_screening_flag": False,
            "negative_news_screening_reason": None,
        }

        is_negative = _is_materially_negative_article(article)

        if is_negative:
            article["negative_news_screening_flag"] = True

            reasons: list[str] = []

            label = _normalize_sentiment_label(
                overall_sentiment_label
            )

            if "bearish" in label or "negative" in label:
                reasons.append(
                    f"negative sentiment label: {overall_sentiment_label}"
                )

            if (
                overall_sentiment_score is not None
                and overall_sentiment_score <= -0.35
            ):
                reasons.append(
                    "sentiment score at or below -0.35"
                )

            article["negative_news_screening_reason"] = "; ".join(
                reasons
            ) or "negative-news screening rule triggered"

        articles.append(article)

    # Keep negative articles first so they cannot be buried.
    articles.sort(
        key=lambda item: (
            not item["negative_news_screening_flag"],
            item.get("time_published") or "",
        )
    )

    negative_articles = [
        article
        for article in articles
        if article["negative_news_screening_flag"]
    ]

    result = _success(
        {
            "articles": articles,
            "article_count": len(articles),
            "negative_article_count": len(negative_articles),
            "negative_articles": negative_articles,
        },
        data_source="Alpha Vantage NEWS_SENTIMENT",
        news_verified=True,
        filters={
            "tickers": normalized_tickers,
            "topics": topics,
            "time_from": time_from,
            "limit": limit,
        },
        malformed_article_count=len(malformed_articles),
        malformed_articles=malformed_articles[:20],
        limitation=(
            "Negative-news screening is rule-based and does not independently "
            "determine legal materiality, investment significance, or whether "
            "an article is confirmed. Human review is required."
        ),
    )

    return result


# ============================================================================
# Meeting brief aggregation tool
# ============================================================================

@tool
def build_meeting_brief(
    client_id: str,
    portfolio_snapshot: dict[str, Any],
    fundamentals_by_ticker: dict[str, Any],
    news_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble an internal client-meeting brief.

    Sections:
    - Portfolio snapshot
    - Fundamentals context
    - Flagged negative news
    - Other news

    The tool:
    - refuses to create a brief from missing portfolio data
    - refuses to fabricate holdings
    - places negative news first and separately
    - preserves source and timestamp fields when supplied
    - does not generate client-facing language
    - does not recommend a trade
    """
    client_id_error = _validate_client_id(client_id)

    if client_id_error:
        return _error(
            "INVALID_CLIENT_ID",
            client_id_error,
        )

    if not isinstance(portfolio_snapshot, dict):
        return _error(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "portfolio_snapshot must be an object.",
        )

    if not isinstance(fundamentals_by_ticker, dict):
        return _error(
            "INVALID_FUNDAMENTALS",
            "fundamentals_by_ticker must be an object.",
        )

    if not isinstance(news_articles, list):
        return _error(
            "INVALID_NEWS_ARTICLES",
            "news_articles must be a list.",
        )

    portfolio_error = portfolio_snapshot.get("error")

    if portfolio_error:
        return _error(
            "PORTFOLIO_DATA_UNAVAILABLE",
            (
                "The client portfolio could not be verified. "
                "The meeting brief will not fabricate holdings."
            ),
            client_id=client_id.strip(),
            portfolio_error=portfolio_error,
            portfolio_verified=False,
            brief_created=False,
        )

    portfolio_status = portfolio_snapshot.get("status")

    if portfolio_status == "error":
        return _error(
            "PORTFOLIO_DATA_UNAVAILABLE",
            (
                "The client portfolio tool returned an error. "
                "The meeting brief will not fabricate holdings."
            ),
            client_id=client_id.strip(),
            portfolio_snapshot=portfolio_snapshot,
            portfolio_verified=False,
            brief_created=False,
        )

    holdings = portfolio_snapshot.get("holdings")

    if holdings is None:
        return _error(
            "MISSING_PORTFOLIO_HOLDINGS",
            (
                "The portfolio snapshot contains no holdings field. "
                "The meeting brief cannot be safely created."
            ),
            client_id=client_id.strip(),
            portfolio_verified=False,
            brief_created=False,
        )

    if not isinstance(holdings, list):
        return _error(
            "INVALID_PORTFOLIO_HOLDINGS",
            "The portfolio holdings field must be a list.",
            client_id=client_id.strip(),
            portfolio_verified=False,
            brief_created=False,
        )

    if not holdings:
        return _error(
            "EMPTY_CLIENT_PORTFOLIO",
            (
                "The portfolio snapshot contains no holdings. "
                "No holding will be fabricated or inferred."
            ),
            client_id=client_id.strip(),
            portfolio_snapshot=portfolio_snapshot,
            portfolio_verified=True,
            holdings_available=False,
            brief_created=False,
        )

    flagged_negative_news: list[dict[str, Any]] = []
    other_news: list[dict[str, Any]] = []
    malformed_news: list[dict[str, Any]] = []

    for index, article in enumerate(news_articles):
        if not isinstance(article, dict):
            malformed_news.append(
                {
                    "index": index,
                    "reason": "Article is not an object.",
                }
            )
            continue

        # Prefer the screening flag generated by get_sector_news_sentiment.
        # If it is absent, independently apply the same conservative rule.
        explicitly_flagged = article.get(
            "negative_news_screening_flag"
        )

        if explicitly_flagged is True:
            is_negative = True
        else:
            is_negative = _is_materially_negative_article(article)

        if is_negative:
            flagged_negative_news.append(article)
        else:
            other_news.append(article)

    # Sort flagged news first by publication timestamp descending where
    # available. The negative section itself is always before other news.
    flagged_negative_news.sort(
        key=lambda article: article.get("time_published") or "",
        reverse=True,
    )

    other_news.sort(
        key=lambda article: article.get("time_published") or "",
        reverse=True,
    )

    source_timestamp_requirements = {
        "portfolio_as_of": portfolio_snapshot.get("as_of"),
        "fundamentals_source": "Provided by upstream fundamentals tool",
        "news_source": "Provided by upstream news tool",
    }

    return _success(
        {
            "client_id": client_id.strip(),
            "portfolio_snapshot": portfolio_snapshot,
            "fundamentals_context": fundamentals_by_ticker,
            "flagged_negative_news": flagged_negative_news,
            "other_news": other_news,
            "summary_counts": {
                "holding_count": len(holdings),
                "fundamentals_count": len(fundamentals_by_ticker),
                "flagged_negative_news_count": len(
                    flagged_negative_news
                ),
                "other_news_count": len(other_news),
                "malformed_news_count": len(malformed_news),
            },
            "source_timestamp_requirements": source_timestamp_requirements,
            "limitations": [
                (
                    "This is an internal-use meeting brief for human review."
                ),
                (
                    "Negative-news screening is not a final materiality "
                    "or investment judgment."
                ),
                (
                    "Fundamentals may have different reporting dates by field."
                ),
                (
                    "No client-facing communication was drafted or sent."
                ),
                (
                    "No specific trade recommendation was generated."
                ),
            ],
            "human_review_required": True,
            "client_facing_document": False,
            "client_contacted": False,
            "trade_recommendation_generated": False,
        },
        brief_type="Internal sales meeting preparation",
        production_ready=bool(
            portfolio_snapshot.get("portfolio_verified") is True
        ),
    )