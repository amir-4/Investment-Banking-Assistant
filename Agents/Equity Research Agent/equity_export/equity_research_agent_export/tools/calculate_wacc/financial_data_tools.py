"""
Tools for financial_analyst_agent.

Data tools call the free, official SEC EDGAR XBRL API (data.sec.gov) —
no API key required, but a compliant User-Agent header and rate-limit
discipline are required per SEC guidance.
Docs: https://www.sec.gov/edgar/sec-api-documentation

Calculation tools implement the formulas in
knowledge/valuation_methodology.md and knowledge/financial_ratios_reference.md
exactly — do not change a formula here without updating that file too.
"""
import requests
from ibm_watsonx_orchestrate.agent_builder.tools import tool

SEC_BASE = "https://data.sec.gov"

HEADERS = {
    "User-Agent": "InvestmentBankingAssistant/1.0 contact@yourcompany.com",
    "Accept-Encoding": "gzip, deflate",
}


def _sec_get(url: str):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )
        return response

    except requests.RequestException as exc:
        return {
            "success": False,
            "error_type": "REQUEST_EXCEPTION",
            "message": str(exc),
            "url": url,
        }


@tool
def get_company_cik(ticker: str) -> dict:
    """
    Resolve a stock ticker to its SEC CIK.
    """

    url = "https://www.sec.gov/files/company_tickers.json"

    result = _sec_get(url)

    # Network-level error
    if isinstance(result, dict):
        return result

    # SEC HTTP error
    if result.status_code != 200:
        return {
            "success": False,
            "error_type": "SEC_HTTP_ERROR",
            "status_code": result.status_code,
            "message": result.text[:1000],
            "url": url,
        }

    # JSON parsing error
    try:
        data = result.json()

    except ValueError as exc:
        return {
            "success": False,
            "error_type": "INVALID_JSON",
            "message": str(exc),
        }

    ticker_upper = ticker.strip().upper()

    # Find ticker
    for entry in data.values():

        if entry.get("ticker") == ticker_upper:

            return {
                "success": True,
                "ticker": ticker_upper,
                "cik": str(entry["cik_str"]).zfill(10),
                "title": entry["title"],
            }

    # Ticker not found
    return {
        "success": False,
        "error_type": "TICKER_NOT_FOUND",
        "ticker": ticker_upper,
        "message": f"No SEC CIK found for ticker '{ticker_upper}'.",
    }

@tool
def get_company_filings(cik: str) -> dict:
    """
    Retrieve recent SEC filing metadata for a company.

    This tool is used to identify the latest authoritative SEC filings
    before any financial analysis is performed.

    Args:
        cik: 10-digit zero-padded SEC CIK number.

    Returns:
        A structured response containing the latest 10-K, 10-Q, and 8-K
        filings plus a recent filing list.
    """

    url = f"{SEC_BASE}/submissions/CIK{cik}.json"

    result = _sec_get(url)

    if isinstance(result, dict):
        return result

    if result.status_code != 200:
        return {
            "success": False,
            "error_type": "SEC_HTTP_ERROR",
            "status_code": result.status_code,
            "message": result.text[:1000],
            "url": url,
        }

    try:
        data = result.json()
    except ValueError as exc:
        return {
            "success": False,
            "error_type": "INVALID_JSON",
            "message": str(exc),
            "url": url,
        }

    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    primary_descriptions = recent.get("primaryDocDescription", [])
    is_xbrl = recent.get("isXBRL", [])
    is_inline_xbrl = recent.get("isInlineXBRL", [])

    filings = []

    for i, form in enumerate(forms):
        if form not in {"10-K", "10-Q", "8-K"}:
            continue

        accession = accession_numbers[i]
        accession_no_hyphens = accession.replace("-", "")

        primary_document = (
            primary_documents[i]
            if i < len(primary_documents)
            else None
        )

        filing_url = None

        if primary_document:
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/"
                f"{accession_no_hyphens}/"
                f"{primary_document}"
            )

        filings.append({
            "form": form,
            "filing_date": filing_dates[i],
            "report_date": report_dates[i],
            "accession_number": accession,
            "primary_document": primary_document,
            "primary_document_description": (
                primary_descriptions[i]
                if i < len(primary_descriptions)
                else None
            ),
            "is_xbrl": (
                is_xbrl[i]
                if i < len(is_xbrl)
                else None
            ),
            "is_inline_xbrl": (
                is_inline_xbrl[i]
                if i < len(is_inline_xbrl)
                else None
            ),
            "filing_url": filing_url,
        })

    latest_10k = next(
        (f for f in filings if f["form"] == "10-K"),
        None
    )

    latest_10q = next(
        (f for f in filings if f["form"] == "10-Q"),
        None
    )

    latest_8k = next(
        (f for f in filings if f["form"] == "8-K"),
        None
    )

    return {
        "success": True,
        "cik": str(cik).zfill(10),
        "company_name": data.get("name"),
        "tickers": data.get("tickers", []),
        "latest_10k": latest_10k,
        "latest_10q": latest_10q,
        "latest_8k": latest_8k,
        "recent_filings": filings[:20],
        "source": "SEC EDGAR submissions API",
        "source_url": url,
    }


@tool
def get_company_facts(cik: str) -> dict:
    """
    Retrieve every XBRL-tagged financial fact ever reported by a company.

    Args:
        cik: 10-digit zero-padded SEC CIK number (use get_company_cik first).

    Returns:
        Full companyfacts JSON payload from SEC EDGAR.
    """
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


@tool
def get_company_concept(cik: str, tag: str, taxonomy: str = "us-gaap") -> dict:
    """
    Retrieve the full historical series for one financial line item.

    Args:
        cik: 10-digit zero-padded SEC CIK number.
        tag: XBRL tag, e.g. "Revenues", "NetIncomeLoss", "Assets",
             "StockholdersEquity", "Liabilities". Not every company uses
             the same tag for the same concept — if this returns no
             data, check get_company_facts for the tag actually used.
        taxonomy: XBRL taxonomy, default "us-gaap".

    Returns:
        JSON payload with the tag's full reporting history across filings.
    """
    url = f"{SEC_BASE}/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


@tool
def calculate_financial_ratios(
    current_assets: float, current_liabilities: float, inventory: float,
    cash: float, total_debt: float, shareholders_equity: float,
    total_assets: float, revenue: float, net_income: float,
    ebit: float, interest_expense: float,
) -> dict:
    """
    Calculate the standard liquidity, profitability, and leverage ratios
    defined in knowledge/financial_ratios_reference.md, from figures
    already pulled from a company's filings.

    All arguments are the underlying financial statement figures for a
    single fiscal period, in the same currency units.

    Returns:
        dict of computed ratios for that period.
    """
    return {
        "current_ratio": current_assets / current_liabilities,
        "quick_ratio": (current_assets - inventory) / current_liabilities,
        "cash_ratio": cash / current_liabilities,
        "net_margin": net_income / revenue,
        "operating_margin": ebit / revenue,
        "roa": net_income / total_assets,
        "roe": net_income / shareholders_equity,
        "debt_to_equity": total_debt / shareholders_equity,
        "interest_coverage": ebit / interest_expense if interest_expense else None,
    }


@tool
def calculate_wacc(
    market_value_equity: float, market_value_debt: float,
    cost_of_equity: float, pre_tax_cost_of_debt: float,
    tax_rate: float, market_value_preferred: float = 0.0,
    cost_of_preferred: float = 0.0,
) -> dict:
    """
    Calculate WACC per the standard formula in
    knowledge/valuation_methodology.md section 3.

    Args:
        market_value_equity: Market value of common equity (E).
        market_value_debt: Market value of total debt (D).
        cost_of_equity: Re, as a decimal (e.g. 0.10 for 10%).
        pre_tax_cost_of_debt: Rd before tax, as a decimal.
        tax_rate: Effective tax rate, as a decimal.
        market_value_preferred: Market value of preferred stock (P), default 0.
        cost_of_preferred: Rp, as a decimal, default 0.

    Returns:
        dict with the WACC and the weight of each capital component, so
        the analyst can see the inputs behind the number, not just the
        result.
    """
    total = market_value_equity + market_value_debt + market_value_preferred
    we = market_value_equity / total
    wd = market_value_debt / total
    wp = market_value_preferred / total
    wacc = (
        we * cost_of_equity
        + wd * pre_tax_cost_of_debt * (1 - tax_rate)
        + wp * cost_of_preferred
    )
    return {
        "wacc": wacc,
        "weight_equity": we,
        "weight_debt": wd,
        "weight_preferred": wp,
    }


@tool
def calculate_dcf(
    projected_fcff: list[float], wacc: float, terminal_growth_rate: float,
    net_debt: float,
) -> dict:
    """
    Calculate implied firm and equity value via DCF, per
    knowledge/valuation_methodology.md section 2.

    Args:
        projected_fcff: Projected free cash flow to the firm for each
            future period (e.g. 5 values for a 5-year projection). Must
            be supplied by the analyst/agent from a built model — this
            tool does not generate projections itself.
        wacc: Discount rate as a decimal (see calculate_wacc).
        terminal_growth_rate: Long-run perpetuity growth rate (g), as a
            decimal. Should be flagged to the user if it looks unusually
            high relative to typical long-run economic growth.
        net_debt: Market value of total debt minus cash and equivalents,
            used to bridge from firm value to equity value.

    Returns:
        dict with discounted cash flows, terminal value (shown
        separately, per methodology notes), firm value, and equity value.
    """
    pv_stage = sum(
        cf / (1 + wacc) ** (t + 1) for t, cf in enumerate(projected_fcff)
    )
    n = len(projected_fcff)
    terminal_fcff = projected_fcff[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcff / (wacc - terminal_growth_rate)
    pv_terminal_value = terminal_value / (1 + wacc) ** n

    firm_value = pv_stage + pv_terminal_value
    equity_value = firm_value - net_debt

    return {
        "pv_of_projection_period_cash_flows": pv_stage,
        "terminal_value_undiscounted": terminal_value,
        "pv_of_terminal_value": pv_terminal_value,
        "terminal_value_pct_of_firm_value": pv_terminal_value / firm_value,
        "firm_value": firm_value,
        "equity_value": equity_value,
    }


@tool
def calculate_comparable_multiples(
    target_metric: dict, peer_multiples: dict[str, list[float]],
) -> dict:
    """
    Derive an implied valuation range for the target company by applying
    peer trading multiples, per knowledge/valuation_methodology.md section 4.

    Args:
        target_metric: The target company's own metrics to which
            multiples will be applied, e.g. {"ebitda": 500_000_000,
            "revenue": 2_000_000_000}.
        peer_multiples: For each multiple type (e.g. "ev_ebitda",
            "ev_revenue"), a list of the peer set's individual multiple
            values (not pre-averaged), so the agent can report both
            median and mean for the analyst to compare.

    Returns:
        dict of implied values per multiple type, using both the peer
        median and mean, so the range — not a single point — is visible.
    """
    import statistics

    results = {}
    metric_map = {"ev_ebitda": "ebitda", "ev_revenue": "revenue"}
    for multiple_name, values in peer_multiples.items():
        metric_key = metric_map.get(multiple_name)
        if metric_key is None or metric_key not in target_metric:
            continue
        median_mult = statistics.median(values)
        mean_mult = statistics.mean(values)
        base = target_metric[metric_key]
        results[multiple_name] = {
            "peer_median_multiple": median_mult,
            "peer_mean_multiple": mean_mult,
            "implied_value_at_median": base * median_mult,
            "implied_value_at_mean": base * mean_mult,
        }
    return results
