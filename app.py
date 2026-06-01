import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Black Scholes
def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

# Monte Carlo
def monte_carlo_european(S0, K, T, r, sigma, M=100000, option_type='call'):
    Z = np.random.standard_normal(M)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    std_err = np.std(payoff) / np.sqrt(M) * np.exp(-r*T)
    return price, std_err

# Binomial tree
def binomial_tree_european(S0, K, T, r, sigma, N=100, option_type='call'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u - d)
    discount = np.exp(-r*dt)
    stock = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**(i-j)) * (d**j)
    option = np.zeros((N+1, N+1))
    if option_type == 'call':
        option[:, N] = np.maximum(stock[:, N] - K, 0)
    else:
        option[:, N] = np.maximum(K - stock[:, N], 0)
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option[j, i] = discount * (p * option[j, i+1] + (1-p) * option[j+1, i+1])
    return option[0, 0]

# LSM
def lsm_american_put(S0, K, T, r, sigma, M=50000, N_steps=50, degree=2):
    dt = T / N_steps
    discount = np.exp(-r * dt)
    S = np.zeros((M, N_steps+1))
    S[:, 0] = S0
    for t in range(1, N_steps+1):
        Z = np.random.standard_normal(M)
        S[:, t] = S[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    cashflow = np.maximum(K - S[:, -1], 0)
    for t in range(N_steps-1, 0, -1):
        in_the_money = S[:, t] < K
        X = S[in_the_money, t].reshape(-1, 1)
        Y = cashflow[in_the_money] * discount
        if len(X) > degree:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, Y)
            continuation = model.predict(X_poly)
            exercise = K - X.flatten()
            exercise_flag = exercise > continuation
            cashflow[in_the_money] = np.where(exercise_flag, exercise, cashflow[in_the_money])
        cashflow = cashflow * discount
    return np.mean(cashflow)

# Greeks
def greeks_bs(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    return {
        'delta_call': cdf_d1,
        'delta_put': cdf_d1 - 1,
        'gamma': pdf_d1 / (S0 * sigma * np.sqrt(T)),
        'vega': S0 * pdf_d1 * np.sqrt(T),
        'theta_call': - (S0 * pdf_d1 * sigma) / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2),
        'theta_put': - (S0 * pdf_d1 * sigma) / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2),
        'rho_call': K * T * np.exp(-r*T) * norm.cdf(d2),
        'rho_put': -K * T * np.exp(-r*T) * norm.cdf(-d2)
    }

# Implied volatility
def implied_volatility(market_price, S0, K, T, r, option_type='call', tol=1e-5):
    if market_price <= 0:
        return np.nan
    low, high = 1e-5, 5.0
    for _ in range(100):
        mid = (low+high)/2
        price = black_scholes_price(S0, K, T, r, mid, option_type)
        if price > market_price:
            high = mid
        else:
            low = mid
        if abs(price - market_price) < tol:
            break
    return (low+high)/2

# Fetch real SPX option data
def get_spx_iv_smile(ticker, expiry_str, option_type='put'):
    asset = yf.Ticker(ticker)

    try:
        opt_chain = asset.option_chain(expiry_str)
    except Exception as e:
        st.error(f"Failed to fetch option chain for {expiry_str}: {e}")
        return None, None, None, None, None, None, None, None

    chain = opt_chain.puts if option_type == "put" else opt_chain.calls
    current_price = asset.history(period="1d")['Close'].iloc[-1]
    chain = chain.copy()

    if "bid" in chain.columns and "ask" in chain.columns:
        chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0) & (chain["ask"] > chain["bid"])]
        chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    else:
        chain["mid"] = chain["lastPrice"]

    # Remove stale quotes
    chain = chain[chain["mid"] > 0.05]

    if "volume" in chain.columns:
        chain = chain[chain["volume"] > 0]

    strikes = chain["strike"].values
    market_prices = chain["mid"].values
    expiry_date = pd.to_datetime(expiry_str)

    T = (expiry_date - pd.Timestamp.now()).total_seconds() / (365.25 * 24 * 3600)
    T = max(T, 1/365)
    r = 0.05

    moneyness = strikes / current_price
    mask = ((moneyness >= 0.80) & (moneyness <= 1.05))
    strikes = strikes[mask]
    market_prices = market_prices[mask]
    moneyness = moneyness[mask]

    ivs = []
    valid_strikes = []
    valid_prices = []
    valid_moneyness = []

    for k, mp, m in zip(strikes, market_prices, moneyness):
        iv = implied_volatility(mp, current_price, k, T, r, option_type)
        if np.isfinite(iv):
            if 0.05 < iv < 1.00:
                ivs.append(iv)
                valid_strikes.append(k)
                valid_prices.append(mp)
                valid_moneyness.append(m)

    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (np.array(valid_strikes), np.array(ivs), np.array(valid_prices), np.array(valid_moneyness),
            current_price, T, expiry_str, download_time)

def main():
    st.set_page_config(layout="wide")
    st.title("Option Pricing Toolbox")
    st.markdown("Black-Scholes · Monte Carlo · Binomial Tree · American LSM · Implied Volatility Analytics")

    tab1, tab2, tab3 = st.tabs(["Basic Pricing", "American Option (LSM)", "Implied Volatility Analytics"])

    # Tab 1: European pricing
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input("S0", value=100.0, step=1.0, key="bs_S0")
            K = st.number_input("K", value=100.0, step=1.0, key="bs_K")
            T = st.number_input("T (years)", value=1.0, step=0.1, key="bs_T")
        with col2:
            r = st.number_input("r", value=0.05, step=0.01, key="bs_r")
            sigma = st.number_input("σ", value=0.2, step=0.01, key="bs_sigma")
            option_type = st.selectbox("Option type", ["call", "put"], key="bs_type")
        if st.button("Calculate European Prices"):
            bs = black_scholes_price(S0, K, T, r, sigma, option_type)
            mc, mc_err = monte_carlo_european(S0, K, T, r, sigma, M=100000, option_type=option_type)
            tree = binomial_tree_european(S0, K, T, r, sigma, N=100, option_type=option_type)
            st.subheader("Price Comparison")
            st.write(f"Black-Scholes: {bs:.4f}")
            st.write(f"Monte Carlo (100k): {mc:.4f} ± {mc_err:.4f}")
            st.write(f"Binomial Tree (100 steps): {tree:.4f}")
            greeks = greeks_bs(S0, K, T, r, sigma)
            st.subheader("Greeks")
            st.dataframe(pd.DataFrame(greeks.items(), columns=["Greek", "Value"]))

    # Tab 2: American put LSM
    with tab2:
        st.markdown("American Put Option Pricing using Longstaff-Schwartz")
        col1, col2 = st.columns(2)
        with col1:
            S0_am = st.number_input("S0", value=100.0, step=1.0, key="am_S0")
            K_am = st.number_input("K", value=100.0, step=1.0, key="am_K")
            T_am = st.number_input("T (years)", value=1.0, step=0.1, key="am_T")
        with col2:
            r_am = st.number_input("r", value=0.05, step=0.01, key="am_r")
            sigma_am = st.number_input("σ", value=0.2, step=0.01, key="am_sigma")
            M_am = st.slider("Number of paths (M)", 10000, 100000, 50000, step=10000)
        if st.button("Price American Put (LSM)"):
            with st.spinner("Running LSM simulation..."):
                price = lsm_american_put(S0_am, K_am, T_am, r_am, sigma_am, M=M_am, N_steps=50)
                euro_price = black_scholes_price(S0_am, K_am, T_am, r_am, sigma_am, option_type='put')
                st.success(f"American Put Price: {price:.4f}")
                st.info(f"European Put Price: {euro_price:.4f}")
                st.write(f"Early exercise premium: {price - euro_price:.4f}")

    # Tab 3: Volatility
    with tab3:
        st.markdown("Live Market Option Chain: Implied Volatility")
        popular_assets = ["^SPX", "SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
        selected_asset = st.selectbox("Popular Assets", popular_assets)
        ticker = st.text_input("Or Enter Any Yahoo Finance Ticker", value=selected_asset).strip().upper()
        st.caption("""
        ```
        Stocks: AAPL, NVDA, TSLA, META
        ETF: SPY, QQQ
        Index: ^SPX, ^NDX
        FX: EURUSD=X, GBPUSD=X, USDJPY=X
        Crypto: BTC-USD, ETH-USD
        """)

        view_type = st.radio("View", ["Real Market", "Theoretical Smile"])
        if view_type == "Theoretical Smile":
            m = np.linspace(0.8, 1.2, 50)
            ivs_theory = 0.15 + 0.30 * (m - 1.0)**2
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(m, ivs_theory, linewidth=2)
            ax.axvline(1.0, linestyle="--", color="red")
            ax.set_title("Theoretical Volatility Smile")
            ax.set_xlabel("Moneyness (K/S)")
            ax.set_ylabel("Implied Volatility")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            try:
                asset_ticker = yf.Ticker(ticker)
                available_exps = asset_ticker.options

                if not available_exps:
                    st.warning("No option chain available for this ticker.")
                else:
                    today = pd.Timestamp.today()
                    valid_exps = []
                    for exp in available_exps:
                        exp_date = pd.to_datetime(exp)
                        t_days = (exp_date - today).days
                        if t_days >= 7:
                            valid_exps.append(exp)
                    if not valid_exps:
                        valid_exps = available_exps

                    exp_options = [f"{exp} (T={max((pd.to_datetime(exp)-today).days,1)/365:.2f}y)" for exp in valid_exps]
                    selected_idx = st.selectbox("Expiration Date", range(len(exp_options)), format_func=lambda i: exp_options[i])
                    selected_expiry = valid_exps[selected_idx]
                    opt_type_spx = st.selectbox("Option Type", ["put", "call"])

                    if st.button("Fetch Market Data"):
                        with st.spinner("Downloading option chain..."):
                            result = get_spx_iv_smile(ticker, selected_expiry, opt_type_spx)
                            if result[0] is None:
                                st.error("Could not retrieve option chain.")
                            else:
                                strikes, ivs, market_prices, moneyness, spot, T, expiry_date, download_time = result
                                atm_idx = np.argmin(np.abs(strikes - spot))
                                atm_strike = strikes[atm_idx]
                                atm_iv = ivs[atm_idx]
                                st.info(f"Downloaded at: {download_time}")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Spot", f"{spot:.2f}")

                                with col2:
                                    st.metric("ATM Strike", f"{atm_strike:.0f}")

                                with col3:
                                    st.metric("ATM IV", f"{atm_iv:.2%}")

                                if opt_type_spx == "put":
                                    st.info(f"""
                                            ITM Put: Strike > {spot:.0f}
                                            ATM Put: Strike ≈ {spot:.0f}
                                            OTM Put: Strike < {spot:.0f}
                                            """)
                                else:
                                    st.info(f"""
                                            ITM Call: Strike < {spot:.0f}
                                            ATM Call: Strike ≈ {spot:.0f}
                                            OTM Call: Strike > {spot:.0f}
                                            """)

                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
                                ax1.plot(strikes, market_prices, 'o', markersize=4)
                                ax1.set_title(f"Market Prices ({ticker})")
                                ax1.set_xlabel("Strike")
                                ax1.set_ylabel("Option Price")
                                ax1.grid(True, alpha=0.3)
                                ax2.plot(moneyness, ivs, 's-', markersize=4, color='green')
                                ax2.axvline(1.0, linestyle='--', color='red')
                                ax2.text(1.0, max(ivs), "ATM", color='red')
                                ax2.set_title(f"IV Curve ({ticker})")
                                ax2.set_xlabel("Moneyness (K/S)")
                                ax2.set_ylabel("Implied Volatility")
                                ax2.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                st.caption(f"{len(strikes)} liquid strikes used.")

            except Exception as e:
                st.error(f"Error loading option chain: {e}")

if __name__ == "__main__":
    main()
