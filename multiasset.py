import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ETF definitions
etf_dict = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "IEMG": "iShares Core MSCI Emerging Markets ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "GLD": "SPDR Gold Shares",
    "USO": "United States Oil Fund",
    "VNQ": "Vanguard Real Estate ETF"
}

display_names = [f"{name} ({ticker})" for ticker, name in etf_dict.items()]
ticker_map = {f"{name} ({ticker})": ticker for ticker, name in etf_dict.items()}

# App config
st.set_page_config(layout="wide")
st.title("Multi-Asset Portfolio Dashboard")

# Sidebar
st.sidebar.title("ETF Selection")
default_selection = ["SPDR S&P 500 ETF Trust (SPY)", "iShares Core U.S. Aggregate Bond ETF (AGG)"]
selected = st.sidebar.multiselect("Choose ETFs:", display_names, default=default_selection)
tickers = [ticker_map[s] for s in selected]

st.sidebar.markdown("### ETF Descriptions")
for t in tickers:
    st.sidebar.markdown(f"- {t}: {etf_dict[t]}")

# Fetch data
@st.cache_data

def get_data(tickers):
    end = datetime.today()
    start = end - timedelta(days=365 * 10)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()

price_data = get_data(tickers)
returns = price_data.pct_change().dropna()

# Expected stats
avg_daily_return = returns.mean()
daily_std = returns.std()
expected_returns = ((1 + avg_daily_return) ** 252) - 1
expected_volatility = daily_std * np.sqrt(252)
cov_matrix = returns.cov() * 252

# Daily Return Distributions
st.subheader("Daily Return Distributions")
fig, axs = plt.subplots((len(tickers) + 1) // 2, 2, figsize=(14, 3.5 * ((len(tickers) + 1) // 2)))
axs = axs.flatten()

for i, ticker in enumerate(tickers):
    axs[i].hist(returns[ticker], bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    axs[i].axvline(avg_daily_return[ticker], color='red', linestyle='--',
                   label=f"μ: {expected_returns[ticker]:.2%}")
    axs[i].set_title(f"{ticker}")
    axs[i].legend()
    axs[i].set_xlabel("Daily Return")
    axs[i].set_ylabel("Frequency")

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
st.pyplot(fig)

# 🔗 Heatmap & Efficient Frontier
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔗 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(returns[tickers].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

with col2:
    st.subheader(" Efficient Frontier")

    def simulate_portfolios(n, exp_returns, cov):
        results = np.zeros((2, n))
        for i in range(n):
            weights = np.random.random(len(exp_returns))
            weights /= np.sum(weights)
            ret = np.dot(weights, exp_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            results[0, i] = vol
            results[1, i] = ret
        return results

    sim_results = simulate_portfolios(3000, expected_returns[tickers], cov_matrix.loc[tickers, tickers])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(sim_results[0], sim_results[1], alpha=0.3, c='gray')
    ax2.set_xlabel("Volatility (Std Dev)")
    ax2.set_ylabel("Expected Annual Return")
    ax2.set_title("Efficient Frontier")
    st.pyplot(fig2)

# Portfolio Weights
st.subheader(" Portfolio Weights (Always Sum to 100%)")

if "weights" not in st.session_state or len(st.session_state.weights) != len(tickers):
    equal_w = [100 / len(tickers)] * len(tickers)
    st.session_state.weights = equal_w

def adjust_weights(changed_index):
    new_value = st.session_state[f"slider_{changed_index}"]
    other_indices = [i for i in range(len(tickers)) if i != changed_index]
    remaining = 100 - new_value
    total_others = sum([st.session_state.weights[i] for i in other_indices])
    for i in other_indices:
        if total_others == 0:
            st.session_state.weights[i] = remaining / len(other_indices)
        else:
            st.session_state.weights[i] = st.session_state.weights[i] / total_others * remaining
        st.session_state[f"slider_{i}"] = int(st.session_state.weights[i])
    st.session_state.weights[changed_index] = new_value

for i, t in enumerate(tickers):
    st.slider(
        f"{t}",
        0,
        100,
        int(st.session_state.weights[i]),
        key=f"slider_{i}",
        on_change=adjust_weights,
        args=(i,)
    )

# Normalize
norm_w = [w / 100 for w in st.session_state.weights]

# Backtest
bt_start = datetime.today() - timedelta(days=730)
bt_data = price_data.loc[bt_start:].dropna()
bt_returns = bt_data.pct_change().dropna()
bt_weighted = bt_returns[tickers].dot(norm_w)
bt_cum = (1 + bt_weighted).cumprod() * 1000

# Safely calculate realized return
if not bt_cum.empty:
    realized_return = bt_cum.iloc[-1] / 1000 - 1
else:
    realized_return = np.nan
    st.warning("Backtest data is empty. Realized return could not be calculated.")

# SPY Benchmark
spy = yf.download("SPY", start=bt_start, end=datetime.today(), auto_adjust=True)["Close"]
spy_cum = (1 + spy.pct_change().dropna()).cumprod() * 1000

if not bt_cum.empty and not spy_cum.empty:
    bt_series = bt_cum.squeeze()
    spy_series = spy_cum.squeeze()
    common_index = bt_series.index.intersection(spy_series.index)
    bt_series = bt_series.loc[common_index]
    spy_series = spy_series.loc[common_index]

    st.subheader("Backtested Portfolio vs SPY Benchmark")
    fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
    ax_bt.plot(bt_series.index.to_numpy(), bt_series.to_numpy(), label="Your Portfolio", color='darkgreen', linewidth=2)
    ax_bt.plot(spy_series.index.to_numpy(), spy_series.to_numpy(), label="SPY Benchmark", color='royalblue', linestyle='--', linewidth=2)
    ax_bt.set_title("Cumulative Return – $1000 Initial Investment", fontsize=14, weight='bold')
    ax_bt.set_ylabel("Portfolio Value ($)")
    ax_bt.grid(True, linestyle='--', alpha=0.6)
    ax_bt.legend()
    st.pyplot(fig_bt)
else:
    st.warning("Not enough data to plot backtest vs benchmark.")

# Summary
port_return = np.dot(norm_w, expected_returns[tickers])
port_vol = np.sqrt(np.dot(norm_w, np.dot(cov_matrix.loc[tickers, tickers], norm_w)))

st.markdown(f"**Expected Annual Return:** `{port_return:.2%}`")
st.markdown(f"**Expected Volatility:** `{port_vol:.2%}`")
st.markdown(f"**2-Year Realized Return:** `{realized_return:.2%}`")
