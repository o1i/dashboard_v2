from typing import Union, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def get_benchmark_portfolio(
    benchmark_frame: pd.DataFrame, benchmark: str = "SAA"
) -> pd.DataFrame:
    """
    Extracts the chosen portfolio from the benchmark files.
    All required fields are already populated
    Returns only rows with non-zero weight
    :param benchmark_frame: benchmark|isin|weight
    :param benchmark: name of the benchmark (must be in benchmark column)
    :return: isin|weight
    """
    return (
        benchmark_frame.loc[
            benchmark_frame["benchmark"] == benchmark, ["isin", "weight"]
        ]
        .copy()
        .assign(value=lambda x: x["weight"])
    )


def get_holdings_portfolio(
    holdings_frame: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    as_of: str,
) -> pd.DataFrame:
    """
    Calculates portfolio weights from holdings and prices at given date.

    :param holdings_frame: date|isin|count
    :param prices: isin|date|price
    :param universe: isin|currency
    :param as_of: YYYY-mm-aa
    :return: isin|weight|amount
    """

    def get_date(df: pd.DataFrame) -> pd.DataFrame:
        assert "date" in df and any(df["date"] == as_of)
        return df[df["date"] == as_of]

    ccy = universe.loc[universe["bucket"] == "Currencies", "isin"]
    info = (
        get_date(holdings_frame)[["isin", "count"]]
        .merge(get_date(prices)[["isin", "price"]])
        .merge(universe[["isin", "currency"]])
        .merge(
            get_date(prices[prices["isin"].isin(ccy)])[["isin", "price"]].rename(
                columns={"isin": "currency", "price": "fx_rate"}
            )
        )
        .assign(value=lambda x: x["count"] * x["price"] * x["fx_rate"])
        .assign(weight=lambda x: x["value"] / x["value"].sum() * 100)
    )
    assert round(info["weight"].sum(), 3) == 100
    return info[["isin", "weight", "value"]]


def portfolio_total_performance(
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    start: str,
    end: Optional[str] = None,
    level: Union[str, list] = "subbucket",
) -> pd.DataFrame:
    """
    Calculates portfolio weights and returns to be used in Return attribution.

    If the end date is None, the next higher date in prices is taken.

    Throws:
    - ValueError when there is no higher date than start_date in prices
    - ValueError when there are securities in the benchmarks that do not have a price
        for the given dates

    :param universe:bucket|subbucket|isin|currency
    :param portfolio: isin|weight
    :param prices: isin|date|price
    :param start: needs to be a pricing date in iso-format
    :param end: needs to be a pricing date or none
    :param level: "bucket" or "subbucket" denoting the reporting level
    :return: <level>|weight|return
    """
    assert abs(portfolio["weight"].sum() - 100) < 0.001, "Portfolio w don't add to 100"
    zero_weights = portfolio.loc[~(portfolio["weight"] > 0), "isin"].values
    assert len(zero_weights) == 0, f"Zero portfolio weights in {tuple(zero_weights)}"
    if end is None:
        try:
            end = min(prices.loc[prices["date"] > start, "date"])
        except ValueError as e:
            raise ValueError(f"No valid price date after {start}") from e
    if isinstance(level, str):
        level = [level]

    returns = (
        prices[prices["date"] == start][["isin", "price"]]
        .merge(
            prices[prices["date"] == end][["isin", "price"]],
            on="isin",
            how="inner",
            suffixes=("_start", "_end"),
        )
        .assign(local_return=lambda x: x["price_end"] / x["price_start"] - 1)
    )
    pos_without_returns = set(portfolio["isin"]) - set(returns["isin"])
    assert (
        not pos_without_returns
    ), f"Portfolio pos. without returns: {tuple(pos_without_returns)}"
    pos_not_in_universe = set(portfolio["isin"]) - set(universe["isin"])
    assert (
        not pos_not_in_universe
    ), f"Portfolio pos. not in universe: {tuple(pos_not_in_universe)}"

    ccy = set(universe[universe["bucket"] == "Currencies"]["isin"].unique())

    info = (
        portfolio.merge(universe)
        .merge(returns[["isin", "local_return"]])
        .merge(
            returns.loc[returns["isin"].isin(ccy), ["isin", "local_return"]].rename(
                columns={"isin": "currency", "local_return": "curr_return"}
            ),
        )
        .assign(
            total_return=lambda x: (1 + x["local_return"]) * (1 + x["curr_return"]) - 1
        )
    )
    # Remove double counting currencies
    ind_curr = info["isin"].isin(ccy)
    info.loc[ind_curr, "total_return"] = info.loc[ind_curr, "local_return"]
    assert len(info) == len(portfolio), "Incomplete data"

    info["weighted_total_return"] = info["weight"] * info["total_return"]
    info["cash_return"] = info["value"] * info["total_return"]
    grouped = info.groupby(level, dropna=False)
    aggregated = pd.DataFrame(
        {
            "return": (
                grouped["weighted_total_return"].sum() / grouped["weight"].sum()
            ),
            "weight": np.round(grouped["weight"].sum(), 2),
            "cash_return": np.round(grouped["cash_return"].sum(), 2),
        }
    ).reset_index()
    aggregated["weight"] *= 100 / np.round(aggregated["weight"].sum(), 4)
    assert (
        round(aggregated["weight"].sum(), 3) == 100
    ), "Adj. weights don't add up to 100"
    return aggregated


def comparison(realised: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates excess return attribution
    - Assumes the same columns in both frames
    - Assumes that the the rows contain no duplicates (ignoring return and weight col)
    - Calculates 3 measures
        - Allocation: (W_r - W_b) * (R_b- RB)
        - Selection: W_b * (R_r - R_b)
        - Interaction: (W_r - W-b) * (R_r - R_b)
    - If the realised portfolio does not have an entry (W_r = 0) then the whole difference
        sshould be allocation effect -> assume R_r == R_b
    - If the realised portfolio has an entry not in the Benchmark (W_b = 0), then it is
        compared to the averaged benchmark return (R_b := sum(W_b*R_b)/sum(W_b)) and
        W_b := 0

    :param realised: <id_columns>|return|weight
    :param benchmark: <id_columns>|return|weight
    :return: <id_coulmns>|allocation|selection
    """
    assert sorted(realised.columns) == sorted(benchmark.columns)
    realised.columns = realised.columns.str.replace("weight", "w")
    realised.columns = realised.columns.str.replace("return", "r")
    benchmark.columns = benchmark.columns.str.replace("weight", "w")
    benchmark.columns = benchmark.columns.str.replace("return", "r")
    index_cols = [c for c in realised.columns if c not in ("r", "w")]
    realised["w"] /= 100
    benchmark["w"] /= 100
    realised["r"] -= 1
    benchmark["r"] -= 1
    both = realised.merge(benchmark, on=index_cols, how="outer", suffixes=("_r", "_b"))
    # Fill missing values
    both["w_r"] = both["w_r"].fillna(0)
    both["w_b"] = both["w_b"].fillna(0)
    ind_not_in_r = both["r_r"].isna()
    both.loc[ind_not_in_r, "r_r"] = both.loc[ind_not_in_r, "r_b"]
    ind_not_in_b = both["r_b"].isna()
    both.loc[ind_not_in_b, "r_b"] = (both["r_b"] * both["w_b"]).sum() / both[
        "w_b"
    ].sum()

    benchmark_return = (both["w_b"] * both["r_b"]).sum() / both["w_b"].sum()

    both["allocation"] = (both["w_r"] - both["w_b"]) * (both["r_b"] - benchmark_return)
    both["selection"] = both["w_b"] * (both["r_r"] - both["r_b"])
    both["interaction"] = (both["w_r"] - both["w_b"]) * (both["r_r"] - both["r_b"])
    both["total"] = both["allocation"] + both["selection"] + both["interaction"]
    return both[index_cols + ["allocation", "selection", "interaction", "total"]]


def get_prices(
    universe: pd.DataFrame, portfolio: pd.DataFrame, manual_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Assemble all relevant prices.

    1. Creates list of dates for which prices are needed.
    2. Sources prices from yahoo finance
    3. Completes with prices from manual_prices

    If Prices of the dates in question are not available, the last available price
    is taken instead.

    Throws errors if:
    - Price neither on yahoo finance or in manual prices

    Args:
        universe: isin|yf
        portfolio: date
        manual_prices: date|isin|price

    Returns:
        date|isin|price
    """
    used_dates = sorted(list(portfolio["date"].unique()))
    start_date = min(used_dates)
    symbols = list(universe["yf"].dropna().unique())
    proxy_symbols = list(universe["yf_proxy"].dropna().unique())
    yf_prices = yf.download(
        symbols + proxy_symbols,
        interval="1d",
        start=start_date,
    )["Close"]
    yf_prices.index = [str(i)[:10] for i in yf_prices.index]
    missing_dates = set(used_dates) - set(yf_prices.index)
    all_dates = (
        yf_prices.reindex(yf_prices.index.values.tolist() + list(missing_dates))
        .sort_index()
        .ffill()
    )
    still_missing_dates = set(used_dates) - set(all_dates.index)
    assert not still_missing_dates, f"Missing dates {still_missing_dates}"
    used_prices = all_dates.loc[used_dates].copy()

    def use_proxies(prices: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
        """Fill nans before first value using proxy."""
        proxies = univ[["yf", "yf_proxy"]].dropna()
        for symbol, proxy in zip(proxies["yf"], proxies["yf_proxy"]):
            first_value = np.where(prices[symbol].notna())[0][0]
            scale = prices[proxy] / prices[proxy].iloc[first_value]
            filled = np.where(
                prices[symbol].notna(),
                prices[symbol],
                prices[symbol].iloc[first_value] * scale,
            )
            prices[symbol] = filled
        return prices

    proxied_prices = use_proxies(used_prices.copy(), universe)[symbols]
    assert proxied_prices.isna().sum().sum() == 0, "Missing prices"

    name_translations = universe.set_index("yf")["isin"].to_dict()
    proxied_prices.columns = [name_translations[c] for c in proxied_prices]

    def add_manual_price(existing: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
        """Adds manual prices to downloaded prices."""
        ordered = man.sort_values(["isin", "date"])
        completed = existing.copy()
        for _, (date, isin, price) in ordered.iterrows():
            try:
                next_date = next(iter(dt for dt in used_dates if dt >= date))
            except StopIteration:
                next_date = max(used_dates)
            completed.loc[next_date, isin] = price
        completed = completed.ffill()
        assert completed.isna().sum().sum() == 0
        return completed

    filled = (
        add_manual_price(existing=proxied_prices, man=manual_prices)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    long = filled.melt(id_vars="date", var_name="isin", value_name="price").sort_values(
        ["date", "isin"]
    )
    return long
