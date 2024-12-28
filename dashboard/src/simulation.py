from typing import Callable

import numpy as np
import pandas as pd


def simulate(
    age: int,
    age_max: int,
    portfolio: np.array,
    update_portfolio: Callable[[np.array, np.array], np.array],
    income: pd.Series,
    consumption: pd.Series,
    tax: Callable[[float], float],
    n_sim: int,
    quantiles: tuple,
) -> pd.DataFrame:
    """
    Simulates possible paths for the development of the protfolio.
    - Consumption taken in the beginning of the year (from salary -> cash -> portfolio)
    - Portfolio values updated (assuming always perfect rebalancing)
    - Salary added at the end of the year
    - Taxes taken at the end of the year

    :param age: starting age
    :param age_max: maximal age of simulation
    :param portfolio: dict of portfolio (dimension d)
    :param update_portfolio: one-step update of a portfolio
    :param income: income, indexed on age
    :param consumption: indexed on age (may have gaps)
    :param tax: function income -> tax (in CHF)
    :param n_sim: number of simulation
    :param quantiles: quantiles of interest
    :return: q_<quantile_1>|q_<quantile_2>...
    """
    ages = np.array(range(age, age_max + 1))
    income = income.reindex(range(age, age_max)).ffill()
    consumption = consumption.reindex(range(age, age_max)).ffill()
    evolution = np.empty([n_sim, len(ages), len(portfolio)])  # beginning of age!
    evolution[:, 0, :] = np.tile(np.array(portfolio).reshape([1, -1]), [n_sim, 1])
    weights = portfolio / np.array(portfolio).sum()
    bankrupcy = np.zeros(n_sim)
    for i, a in enumerate(ages[:-1]):
        sal = income.loc[a]
        con = consumption.loc[a]
        consumption_from_sal = min(sal, con)
        consumption_from_portfolio = con - consumption_from_sal
        ind_bankrupcy = evolution[:, i, :].sum(axis=1) < consumption_from_portfolio
        bankrupcy[ind_bankrupcy] = 1
        available = evolution[:, i, :] - (
            weights.reshape([1, -1]) * consumption_from_portfolio
        )
        evolution[:, i + 1, :] = np.apply_along_axis(
            update_portfolio, 1, available, weights=weights
        )
        taxes = tax(sal)
        cf_to_portfolio = sal - taxes - consumption_from_sal
        evolution[:, i + 1, :] += weights.reshape([1, -1]) * cf_to_portfolio
        evolution[:, i + 1, :] *= (evolution[:, i + 1, :].sum(axis=1) > 0).reshape(
            [-1, 1]
        )

    return pd.DataFrame(
        np.quantile(evolution.sum(axis=2), quantiles, axis=0).transpose(),
        columns=[f"q_{round(q*100)}" for q in quantiles],
        index=ages,
    )


def normal_portfolio_update(start: np.array, weights: np.array) -> np.array:
    log_returns = np.random.normal(0.05, 0.1, len(start))
    changed = start * np.exp(log_returns)
    rebalanced = weights * changed.sum()
    return rebalanced


def flat_tax(income: float) -> float:
    return income * 0.3
