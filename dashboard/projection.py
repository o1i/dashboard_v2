from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.graph_objects as go

from src.simulation import simulate, flat_tax, normal_portfolio_update

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Dashboard"),
        html.Div("Plausible values for net worth for different ages"),
        dcc.Graph(id="evolution"),
        html.Div(
            ["Age: ", input_age := dcc.Input(type="number", min=0, value=35), html.Br()]
        ),
        html.Div(
            [
                "# Sim: ",
                input_n_sim := dcc.Input(type="number", min=0, value=500),
                html.Br(),
            ]
        ),
        html.Div(
            [
                "Salary (dict):",
                salary_input := dcc.Textarea(
                    value="{35:140, 50:0}",
                ),
            ]
        ),
        html.Div(
            [
                "Consumption (dict):",
                consumption_input := dcc.Textarea(
                    value="{35:40, 40:50, 50:60}", style={"rows": 1}
                ),
            ]
        ),
        html.Div(
            [
                "Portfolio:",
                html.Div(
                    [
                        "Cash: ",
                        input_cash := dcc.Input(type="number", min=0, value=270),
                        html.Br(),
                        "US equities: ",
                        input_eq_us := dcc.Input(type="number", min=0, value=45),
                        html.Br(),
                        "Global equities: ",
                        input_eq_glob := dcc.Input(type="number", min=0, value=20),
                        html.Br(),
                        "Swiss equities: ",
                        input_eq_ch := dcc.Input(type="number", min=0, value=27),
                        html.Br(),
                        "Real estate: ",
                        input_re := dcc.Input(type="number", min=0, value=100),
                        html.Br(),
                        "Precious metals: ",
                        input_pm := dcc.Input(type="number", min=0, value=15),
                        html.Br(),
                    ]
                ),
            ]
        ),
        html.Button("Simulate", id="submit"),
    ]
)


@app.callback(
    Output("evolution", "figure"),
    Input("submit", "n_clicks"),
    State(input_age, component_property="value"),
    State(input_n_sim, component_property="value"),
    State(salary_input, component_property="value"),
    State(consumption_input, component_property="value"),
    State(input_cash, component_property="value"),
    State(input_eq_us, component_property="value"),
    State(input_eq_glob, component_property="value"),
    State(input_eq_ch, component_property="value"),
    State(input_re, component_property="value"),
    State(input_pm, component_property="value"),
)
def update_figure(_, age, n_sim, sal, con, cash, eq_us, eq_glob, eq_ch, re, pm):
    sim = simulate(
        age=age,
        age_max=90,
        portfolio=[cash, eq_us, eq_glob, eq_ch, re, pm],
        update_portfolio=normal_portfolio_update,
        income=pd.Series(eval(sal)),
        consumption=pd.Series(eval(con)),
        tax=flat_tax,
        n_sim=n_sim,
        quantiles=(0.01, 0.05, 0.5),
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.index, y=sim["q_50"], name="Median"))
    fig.add_trace(go.Scatter(x=sim.index, y=sim["q_5"], name="5% quantile"))
    fig.add_trace(go.Scatter(x=sim.index, y=sim["q_1"], name="1% quantile"))
    fig.update_layout(transition_duration=500)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
