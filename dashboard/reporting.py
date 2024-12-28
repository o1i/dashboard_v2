from functools import partial
from pathlib import Path

import colour
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.reporting import (
    get_benchmark_portfolio,
    get_holdings_portfolio,
    portfolio_total_performance,
    get_prices,
)

# __file__ = r"/Users/oliverburkhard/PycharmProjects/dashboard/dashboard/reporting.py"
app = Dash(__name__)


def desaturate(s: str, frac: float) -> str:
    c = colour.Color(s)
    lum = c.get_luminance()
    c.set_luminance(lum * (1 - frac))
    return c.get_hex()


COLS = {
    "bg": "#333339",
    "text": "#abaaaa",
    "gray1": "#9999aa",
    "gray2": "#666677",
}

GRAPH_LAYOUT = {
    "font_family": "Helvetica",
    "font_color": COLS["text"],
    "paper_bgcolor": COLS["bg"],
    "plot_bgcolor": COLS["bg"],
    "margin": {"b": 0, "t": 0, "l": 0, "r": 0},
}

GRAPH_CONFIG = {"displayModeBar": False}

RAW = Path(__file__).parents[0] / "data"
benchmarks = pd.read_csv(RAW / "benchmarks.csv")
universe = pd.read_csv(RAW / "universe.csv")
manual_prices = pd.read_csv(RAW / "manual_prices.csv")
holdings_frame = pd.read_csv(RAW / "portfolio.csv")
prices = get_prices(
    universe=universe, portfolio=holdings_frame, manual_prices=manual_prices
)
colours = pd.read_csv(RAW / "colours.csv")
colours["colour_desat"] = colours["colour"].map(partial(desaturate, frac=0.3))
dates = sorted(list(holdings_frame["date"].unique()))

HEIGHT = "80%"


app.layout = html.Div(
    [
        html.H1("Portfolio Dashboard"),
        html.Div(
            [
                html.Div(
                    [
                        "Start: ",
                        date_start := dcc.Dropdown(
                            dates[:-1],
                            dates[-2],
                            id="start",
                            className="inputField",
                            style={"color": "abaaaa", "baackground": "#333339"},
                        ),
                    ],
                    className="input-part",
                ),
                html.Div(
                    [
                        "End: ",
                        date_end := dcc.Dropdown(
                            dates[1:], dates[-1], id="end", className="inputField"
                        ),
                    ],
                    className="input-part",
                ),
            ],
            id="user-input",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Evolution"),
                        dcc.Graph(
                            id="evolution",
                            config=GRAPH_CONFIG,
                            style={"height": HEIGHT, "width": "100%"},
                        ),
                    ],
                    className="grid-item",
                    id="evolutionContainer",
                ),
                html.Div(
                    [
                        html.H2("Cash Return"),
                        dcc.Graph(
                            id="cash_return",
                            config=GRAPH_CONFIG,
                            style={"height": HEIGHT, "width": "100%"},
                        ),
                    ],
                    className="grid-item",
                ),
                html.Div(
                    [
                        html.H2("Return"),
                        dcc.Graph(
                            id="return",
                            config=GRAPH_CONFIG,
                            style={"height": HEIGHT, "width": "100%"},
                        ),
                    ],
                    className="grid-item",
                ),
                html.Div(
                    [
                        html.H2("Deviation"),
                        dcc.Graph(
                            id="deviation",
                            config=GRAPH_CONFIG,
                            style={"height": HEIGHT, "width": "100%"},
                        ),
                    ],
                    className="grid-item",
                ),
                html.Div(
                    [
                        html.H2("Performance attribution"),
                        dcc.Graph(
                            id="attribution",
                            config=GRAPH_CONFIG,
                            style={"height": HEIGHT, "width": "100%"},
                        ),
                    ],
                    className="grid-item",
                    id="attributionContainer",
                ),
            ],
            id="grid_container",
        ),
    ]
)


@app.callback(
    Output("evolution", "figure"),
    Input(date_start, "value"),
    Input(date_end, "value"),
)
def evolution(start, end):
    evolution = (
        pd.concat(
            [
                get_holdings_portfolio(holdings_frame, prices, universe, date).assign(
                    date=date
                )
                for date in dates
            ]
        )
        .merge(universe[["isin", "desc", "bucket", "subbucket"]], how="left")
        .merge(colours, how="left")
        .sort_values(["date", "bucket", "value"], ascending=False)
    )
    evolution["value"] = evolution["value"].map(lambda x: round(x / 100) * 100)
    evolution["weight"] /= 100

    fig = px.area(
        evolution,
        x="date",
        y="value",
        line_group="desc",
        color="desc",
        color_discrete_map=dict(zip(evolution["desc"], evolution["colour"])),
        hover_name="desc",
        hover_data={"weight": ":.1%", "value": True, "desc": False, "date": False},
    )

    fig.update_layout(legend_title_text="Security", legend_traceorder="reversed")
    fig.update_xaxes(
        title_text="Date",
        tickvals=dates,
        tickmode="array",
    )
    fig.update_yaxes(
        title_text="Value",
        # showticklabels=False,
    )
    fig.update_layout(GRAPH_LAYOUT)
    return fig


@app.callback(
    Output("cash_return", "figure"),
    Input(date_start, "value"),
    Input(date_end, "value"),
)
def update_cash_return(start, end):
    portfolio = get_holdings_portfolio(holdings_frame, prices, universe, start)
    returns = (
        portfolio_total_performance(
            universe, portfolio, prices, start, end, level=["bucket", "subbucket"]
        )
        .sort_values("cash_return", ascending=False)
        .merge(colours, how="left")
    )
    returns["label"] = returns["bucket"] + ", " + returns["subbucket"]
    fig = px.bar(
        returns,
        x="label",
        y="cash_return",
        color="label",
        color_discrete_map=dict(zip(returns["label"], returns["colour"])),
    )
    fig.update_layout(GRAPH_LAYOUT)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Cash return", nticks=4)
    fig.update_xaxes(title="Asset class")
    return fig


@app.callback(
    Output("return", "figure"),
    Input(date_start, "value"),
    Input(date_end, "value"),
)
def update_return(start, end):
    portfolio = get_holdings_portfolio(holdings_frame, prices, universe, start)
    returns = (
        portfolio_total_performance(
            universe, portfolio, prices, start, end, level=["bucket", "subbucket"]
        )
        .sort_values("cash_return", ascending=False)
        .merge(colours, how="left")
    )
    returns["label"] = returns["bucket"] + ", " + returns["subbucket"]
    fig = px.bar(
        returns,
        x="label",
        y="return",
        color="label",
        color_discrete_map=dict(zip(returns["label"], returns["colour"])),
    )
    fig.update_layout(GRAPH_LAYOUT)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Return", nticks=4, tickformat=".0%")
    fig.update_xaxes(title="Asset class")
    return fig


@app.callback(
    Output("deviation", "figure"),
    Input(date_end, "value"),
)
def update_deviation(end):
    current_weights = (
        get_holdings_portfolio(holdings_frame, prices, universe, end)
        .merge(universe, on="isin", how="left")
        .merge(colours, how="left")
        .groupby(["bucket", "subbucket", "colour"])["value"]
        .sum()
    )
    target_weights = (
        get_benchmark_portfolio(benchmarks, "SAA")
        .merge(universe, on="isin", how="left")
        .merge(colours, how="left")
        .groupby(["bucket", "subbucket", "colour"])["value"]
        .sum()
    )

    def norm(s: pd.Series) -> pd.Series:
        return s / s.sum()

    deviation = (
        (
            current_weights.transform(norm).subtract(
                target_weights.transform(norm), fill_value=0
            )
            * (current_weights.sum())
        )
        .reset_index()
        .sort_values(["bucket", "subbucket"])
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=deviation["value"],
            y=deviation["bucket"] + ", " + deviation["subbucket"],
            marker={"color": deviation["colour"]},
            orientation="h",
            showlegend=False,
        )
    )
    fig.update_layout(GRAPH_LAYOUT)
    return fig


@app.callback(
    Output("attribution", "figure"),
    Input(date_start, "value"),
    Input(date_end, "value"),
)
def update_attribution(start, end):
    portfolio = get_holdings_portfolio(holdings_frame, prices, universe, start)
    portfolio_returns = portfolio_total_performance(
        universe, portfolio, prices, start, end, level=["bucket", "subbucket"]
    ).drop(columns="cash_return")
    benchmark = get_benchmark_portfolio(benchmarks, "SAA")
    benchmark_returns = portfolio_total_performance(
        universe, benchmark, prices, start, end, level=["bucket", "subbucket"]
    ).drop(columns="cash_return")
    attrib = portfolio_returns.merge(
        benchmark_returns,
        on=["bucket", "subbucket"],
        how="outer",
        suffixes=["_p", "_b"],
    )
    ind_no_b = attrib["return_b"].isna()
    attrib.loc[ind_no_b, "return_b"] = attrib.loc[ind_no_b, "return_p"]
    ind_no_p = attrib["return_p"].isna()
    attrib.loc[ind_no_p, "return_p"] = attrib.loc[ind_no_p, "return_b"]
    attrib.fillna(0, inplace=True)

    # Brinson-Fachler subtracts avg benchmark portfolio (other than BHB attribution)
    r_b = (attrib["weight_b"] * attrib["return_b"]).sum() / attrib["weight_b"].sum()

    attrib["allocation"] = (attrib["weight_p"] - attrib["weight_b"]) * (
        attrib["return_b"] - r_b
    )
    attrib["selection"] = attrib["weight_b"] * (attrib["return_p"] - attrib["return_b"])
    attrib["interaction"] = (attrib["weight_p"] - attrib["weight_b"]) * (
        attrib["return_p"] - attrib["return_b"]
    )
    attrib["total"] = (
        attrib["weight_p"] * attrib["return_p"]
        - attrib["weight_b"] * attrib["return_b"]
    )
    attrib["label"] = attrib["bucket"] + ", " + attrib["subbucket"]
    attrib = attrib.merge(colours, how="left")
    attrib.sort_values(["label"], inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=attrib["label"],
            y=attrib["allocation"],
            marker={"color": attrib["colour"]},
            name="Allocation",
            marker_pattern_shape=".",
        )
    )
    fig.add_trace(
        go.Bar(
            x=attrib["label"],
            y=attrib["selection"],
            marker={"color": attrib["colour"]},
            name="Selection",
            marker_pattern_shape="x",
        )
    )
    fig.add_trace(
        go.Bar(
            x=attrib["label"],
            y=attrib["interaction"],
            marker={"color": attrib["colour"]},
            name="Interaction",
            marker_pattern_shape="+",
        )
    )

    fig.update_layout(GRAPH_LAYOUT)
    fig.update_layout(barmode="relative")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
