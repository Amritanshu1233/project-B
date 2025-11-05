
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------- Parameters -------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
START = "2020-01-01"
END = "2024-01-31"  # inclusive
WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])
CONF_LEVEL = 0.95
BOOTSTRAP_ROUNDS = 2000
ROLL_WINDOW = 60  # trading days
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- Utilities -------------------------

def download_data(tickers, start, end):
    """Download adjusted close prices for given tickers."""
    data = yf.download(tickers, start=start, end=end, progress=False)

    # Ensure 'Adj Close' exists
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    else:
        # fallback to 'Close' if Adj Close missing
        if 'Close' in data.columns:
            data = data['Close']
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' columns found in downloaded data.")
    
    # If only one ticker, convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="all")
    return data



def compute_log_returns(price_df):
    """Compute daily log returns from adjusted close prices."""
    return np.log(price_df / price_df.shift(1)).dropna()


# ------------------------- Risk Measures -------------------------

def var_historical(returns, alpha=0.95):
    """Historical VaR: alpha confidence -> (1-alpha) quantile of returns."""
    q = np.percentile(returns, 100 * (1 - alpha))
    return -q


def var_normal(mean, sigma, alpha=0.95):
    """Parametric Normal VaR (one-sided). Returns positive number representing loss."""
    z = stats.norm.ppf(1 - alpha)
    return -(mean + z * sigma)


def var_t(mean, sigma, df, alpha=0.95):
    """Parametric t-distribution VaR.
    We use the Student's t quantile. The t distribution in scipy is centered at 0,
    so we scale accordingly. Returns positive loss.
    """
    t_q = stats.t.ppf(1 - alpha, df)
    return -(mean + t_q * sigma * np.sqrt((df) / (df - 2)))


def expected_shortfall_historical(returns, alpha=0.95):
    """Historical Expected Shortfall (Average loss beyond VaR)."""
    threshold = np.percentile(returns, 100 * (1 - alpha))
    tail_losses = returns[returns <= threshold]
    if len(tail_losses) == 0:
        return -threshold
    return -tail_losses.mean()


def expected_shortfall_normal(mean, sigma, alpha=0.95):
    """Parametric ES under Normal assumptions."""
    z = stats.norm.ppf(1 - alpha)
    pdf = stats.norm.pdf(z)
    es = -(mean + sigma * pdf / (1 - alpha))
    return es


# ------------------------- Bootstrap -------------------------

def bootstrap_ci(data, statistic_func, rounds=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    stats_boot = np.empty(rounds)
    for i in range(rounds):
        sample = rng.choice(data, size=n, replace=True)
        stats_boot[i] = statistic_func(sample)
    lower = np.percentile(stats_boot, 100 * (1 - ci) / 2)
    upper = np.percentile(stats_boot, 100 * (1 + ci) / 2)
    return lower, upper, stats_boot


# ------------------------- Backtesting -------------------------

def backtest_var(returns, var_level):
    """Count exceptions where actual return < -VaR (i.e., loss exceeded VaR)."""
    exceptions = returns < -var_level
    num_exceptions = exceptions.sum()
    rate = num_exceptions / len(returns)
    return int(num_exceptions), rate, exceptions


# ------------------------- Analysis Pipeline -------------------------

def analyze_portfolio(tickers=TICKERS, start=START, end=END, weights=WEIGHTS):
    prices = download_data(tickers, start, end)
    print(f"Downloaded price data: {prices.shape[0]} rows, {prices.shape[1]} columns")

    log_returns = compute_log_returns(prices)
    log_returns = log_returns.dropna()

    # Equal-weighted portfolio returns (log returns approximate)
    portfolio_returns = log_returns.dot(weights)
    portfolio_returns.name = "Portfolio"

    # Summary statistics
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std(ddof=1)
    n = len(portfolio_returns)

    print(f"Sample days: {n}")
    print(f"Mean daily return (mu): {mu:.6f}")
    print(f"Std dev (sigma): {sigma:.6f}")

    # Fit Student's t to portfolio returns
    t_params = stats.t.fit(portfolio_returns.values)
    df_fit, loc_fit, scale_fit = t_params
    print(f"Fitted t params: df={df_fit:.3f}, loc={loc_fit:.6f}, scale={scale_fit:.6f}")

    # VaR calculations
    var_hist = var_historical(portfolio_returns.values, alpha=CONF_LEVEL)
    var_norm = var_normal(mu, sigma, alpha=CONF_LEVEL)
    var_tdist = var_t(mu, sigma, df_fit, alpha=CONF_LEVEL)

    es_hist = expected_shortfall_historical(portfolio_returns.values, alpha=CONF_LEVEL)
    es_norm = expected_shortfall_normal(mu, sigma, alpha=CONF_LEVEL)

    print("\nVaR Results (1-day, 95%):")
    print(f"  Historical VaR: {var_hist:.4%}")
    print(f"  Normal VaR:     {var_norm:.4%}")
    print(f"  t-dist VaR:     {var_tdist:.4%}")

    print("\nExpected Shortfall (ES, 95%):")
    print(f"  Historical ES: {es_hist:.4%}")
    print(f"  Normal ES:     {es_norm:.4%}")

    # Bootstrap CIs for mean and sigma
    mean_stat = lambda x: np.mean(x)
    sd_stat = lambda x: np.std(x, ddof=1)

    m_low, m_high, m_boot = bootstrap_ci(portfolio_returns.values, mean_stat, rounds=BOOTSTRAP_ROUNDS)
    s_low, s_high, s_boot = bootstrap_ci(portfolio_returns.values, sd_stat, rounds=BOOTSTRAP_ROUNDS)

    print("\nBootstrap 95% CI for mean:")
    print(f"  ({m_low:.6f}, {m_high:.6f})")
    print("Bootstrap 95% CI for sigma:")
    print(f"  ({s_low:.6f}, {s_high:.6f})")

    # Backtest Historical VaR
    num_exc, exc_rate, exc_series = backtest_var(portfolio_returns.values, var_hist)
    print("\nBacktest (Historical VaR):")
    print(f"  Exceptions: {num_exc} out of {n} days")
    print(f"  Exception rate: {exc_rate:.2%} (expected {100*(1-CONF_LEVEL):.2f}%)")

    # Rolling VaR
    rolling_var = portfolio_returns.rolling(window=ROLL_WINDOW).apply(lambda x: var_historical(x, alpha=CONF_LEVEL))

    # Save results
    results = {
        "mu": mu,
        "sigma": sigma,
        "var_historical": var_hist,
        "var_normal": var_norm,
        "var_t": var_tdist,
        "es_historical": es_hist,
        "es_normal": es_norm,
        "t_df": df_fit,
        "bootstrap_mean_ci": (m_low, m_high),
        "bootstrap_sigma_ci": (s_low, s_high),
        "backtest_exceptions": (num_exc, exc_rate)
    }

    # Save CSVs and plots
    portfolio_returns.to_csv(os.path.join(OUTPUT_DIR, "portfolio_returns.csv"))
    log_returns.to_csv(os.path.join(OUTPUT_DIR, "asset_log_returns.csv"))

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(log_returns.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Log-Returns Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Distribution of portfolio returns with fitted normal and t
    x = np.linspace(portfolio_returns.min() * 1.5, portfolio_returns.max() * 1.5, 1000)
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns, bins=80, kde=False, stat='density', label='Empirical')
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal pdf')
    # t pdf: adjust scale exactly using fitted params
    t_pdf = stats.t.pdf((x - loc_fit) / scale_fit, df_fit) / scale_fit
    plt.plot(x, t_pdf, label="Student's t pdf")
    plt.title('Portfolio Return Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "return_distribution.png"), dpi=150)
    plt.close()

    # Cumulative returns & drawdown
    cum_returns = (1 + portfolio_returns).cumprod() - 1
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(cum_returns.index, cum_returns.values)
    ax[0].set_title('Cumulative Returns')
    ax[0].grid(True)

    ax[1].plot(drawdown.index, drawdown.values)
    ax[1].set_title('Drawdown')
    ax[1].axhline(drawdown.min(), color='red', linestyle='--', linewidth=1)
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cum_returns_drawdown.png"), dpi=150)
    plt.close()

    # Rolling VaR plot
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_var.index, rolling_var.values, label=f"Rolling {ROLL_WINDOW}-day Historical VaR")
    plt.axhline(var_hist, color='r', linestyle='--', label='Static Historical VaR')
    plt.title(f"Rolling {ROLL_WINDOW}-day VaR (1-day, {int(CONF_LEVEL*100)}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rolling_var.png"), dpi=150)
    plt.close()

    # Save results summary
    summary_df = pd.DataFrame({
        'metric': list(results.keys()),
        'value': [
            results['mu'], results['sigma'], results['var_historical'], results['var_normal'],
            results['var_t'], results['es_historical'], results['es_normal'], results['t_df'],
            f"{results['bootstrap_mean_ci']}", f"{results['bootstrap_sigma_ci']}",
            f"{results['backtest_exceptions']}"
        ]
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_results.csv'), index=False)

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    return results, portfolio_returns, log_returns


# ------------------------- Entry Point -------------------------
if __name__ == '__main__':
    results, portfolio_returns, log_returns = analyze_portfolio()
    # Print concise results
    print('\nDone. Key results:')
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print('\nOpen the ./outputs/ folder to inspect charts and CSVs.')
