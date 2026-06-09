import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Forecasting Dashboard",
    page_icon="🦠",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🦠 COVID-19 Forecasting Dashboard")
st.markdown("**ARIMA vs SARIMAX** — Weekly case forecasting across 5 WHO regions")
st.markdown("---")

# ── Countries ─────────────────────────────────────────────────────────────────
COUNTRIES = {
    "India":        "SEARO",
    "Germany":      "EURO",
    "United States":"AMRO",
    "South Africa": "AFRO",
    "Japan":        "WPRO",
}

# ── Load data (cached so it doesn't re-download on every interaction) ─────────
@st.cache_data(show_spinner="Downloading latest COVID-19 data from OWID...")
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["location"].isin(COUNTRIES.keys())].copy()

    weekly_data = []
    for country in COUNTRIES:
        cdf = df[df["location"] == country].copy()
        cdf = cdf.set_index("date")
        weekly = cdf["new_cases"].resample("W").sum().reset_index()
        weekly["location"] = country
        weekly["who_region"] = COUNTRIES[country]
        weekly_data.append(weekly)

    result = pd.concat(weekly_data, ignore_index=True)
    result["new_cases"] = result["new_cases"].fillna(0).clip(lower=0).astype(int)
    return result

# ── Model (cached per country so it doesn't retrain on every click) ───────────
@st.cache_data(show_spinner="Training ARIMA & SARIMAX models...")
def run_models(country, forecast_horizon=8):
    df = load_data()
    ts = df[df["location"] == country].set_index("date")["new_cases"]
    train = ts[:-forecast_horizon]
    test  = ts[-forecast_horizon:]

    # ARIMA
    try:
        auto   = auto_arima(train, seasonal=False, trace=False,
                            error_action="ignore", suppress_warnings=True)
        order  = auto.order
    except:
        order  = (1, 1, 1)
    arima_fc = ARIMA(train, order=order).fit().forecast(steps=forecast_horizon)

    # SARIMAX
    try:
        sarimax_fc = SARIMAX(train, order=(1,1,1),
                             seasonal_order=(1,0,1,52)).fit(disp=False).forecast(steps=forecast_horizon)
    except:
        sarimax_fc = arima_fc

    def mape(actual, pred):
        a, p = np.array(actual), np.array(pred)
        safe = np.where(a == 0, 1, a)
        return round(float(np.mean(np.abs((a - p) / safe)) * 100), 2)

    return {
        "ts":         ts,
        "train":      train,
        "test":       test,
        "arima_fc":   arima_fc,
        "sarimax_fc": sarimax_fc,
        "arima_mape": mape(test, arima_fc),
        "sarimax_mape": mape(test, sarimax_fc),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
selected_country = st.sidebar.selectbox("Select Country", list(COUNTRIES.keys()))
forecast_horizon = st.sidebar.slider("Forecast Horizon (weeks)", 4, 16, 8)
show_raw         = st.sidebar.checkbox("Show raw data table", value=False)

region = COUNTRIES[selected_country]
st.sidebar.markdown(f"**WHO Region:** {region}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data source:** [Our World in Data](https://ourworldindata.org/covid-cases)")
st.sidebar.markdown("**Built by:** [Devang Yadav](https://github.com/Dushdeva)")

# ── Top metric cards ──────────────────────────────────────────────────────────
country_df = df[df["location"] == selected_country]
total_cases = int(country_df["new_cases"].sum())
peak_cases  = int(country_df["new_cases"].max())
peak_week   = country_df.loc[country_df["new_cases"].idxmax(), "date"].strftime("%b %Y")
data_weeks  = len(country_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌍 Country",       selected_country)
col2.metric("📊 Total Cases",   f"{total_cases:,}")
col3.metric("📈 Peak Week",     f"{peak_cases:,}", peak_week)
col4.metric("📅 Weeks of Data", data_weeks)

st.markdown("---")

# ── Raw time series chart ─────────────────────────────────────────────────────
st.subheader(f"📉 Weekly New Cases — {selected_country} ({region})")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(country_df["date"], country_df["new_cases"], color="#1A56A0", linewidth=1.5)
ax1.fill_between(country_df["date"], country_df["new_cases"], alpha=0.15, color="#1A56A0")
ax1.set_xlabel("Date")
ax1.set_ylabel("Weekly New Cases")
ax1.set_title(f"{selected_country} — Full COVID-19 Timeline")
ax1.grid(alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# ── Model forecast ────────────────────────────────────────────────────────────
st.subheader("🤖 ARIMA vs SARIMAX Forecast")

with st.spinner("Training models..."):
    m = run_models(selected_country, forecast_horizon)

# Forecast chart
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(m["test"].index, m["test"].values,
         "k-", linewidth=2, label="Actual")
ax2.plot(m["test"].index, m["arima_fc"].values,
         "o--", color="#1A56A0", label=f"ARIMA  (MAPE: {m['arima_mape']}%)")
ax2.plot(m["test"].index, m["sarimax_fc"].values,
         "s--", color="#E05C2A", label=f"SARIMAX (MAPE: {m['sarimax_mape']}%)")
ax2.axvline(x=m["test"].index[0], color="red", linestyle=":", alpha=0.7, label="Forecast Start")
ax2.set_xlabel("Date")
ax2.set_ylabel("Weekly New Cases")
ax2.set_title(f"{selected_country} — {forecast_horizon}-Week Forecast")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)

# Model result cards
st.markdown("#### Model Results")
r1, r2, r3 = st.columns(3)
better = "ARIMA" if m["arima_mape"] < m["sarimax_mape"] else "SARIMAX"
r1.metric("ARIMA MAPE",   f"{m['arima_mape']}%")
r2.metric("SARIMAX MAPE", f"{m['sarimax_mape']}%")
r3.metric("Better Model", better)

# MAPE note
st.info("ℹ️ High MAPE values occur when actual cases pass through near-zero during wave troughs — a known limitation of MAPE on COVID data. Relative model comparison between ARIMA and SARIMAX remains valid.")

st.markdown("---")

# ── All countries comparison ──────────────────────────────────────────────────
st.subheader("🌍 All Countries — Raw Time Series")

fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
colors = ["#1A56A0", "#E05C2A", "#2E8B57", "#8B2FC9", "#C9A02F"]

for idx, (country, region_code) in enumerate(COUNTRIES.items()):
    cdf = df[df["location"] == country]
    axes[idx].plot(cdf["date"], cdf["new_cases"],
                   color=colors[idx], linewidth=1.5)
    axes[idx].fill_between(cdf["date"], cdf["new_cases"],
                           alpha=0.15, color=colors[idx])
    axes[idx].set_title(f"{country} ({region_code})", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Date", fontsize=9)
    axes[idx].set_ylabel("Weekly Cases", fontsize=9)
    axes[idx].grid(alpha=0.3)
    axes[idx].tick_params(axis="x", rotation=30, labelsize=8)

axes[5].axis("off")
fig3.suptitle("Weekly COVID-19 Cases — All WHO Regions", fontsize=14, fontweight="bold")
fig3.tight_layout()
st.pyplot(fig3)

st.markdown("---")

# ── Raw data table ────────────────────────────────────────────────────────────
if show_raw:
    st.subheader(f"📋 Raw Weekly Data — {selected_country}")
    st.dataframe(
        country_df[["date", "new_cases", "who_region"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey; font-size:13px;'>"
    "Built by <a href='https://github.com/Dushdeva'>Devang Yadav</a> · "
    "Data: <a href='https://ourworldindata.org'>Our World in Data</a>"
    "</div>",
    unsafe_allow_html=True
)
