import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from analyticsHub.main import (
    get_all_performances,
    get_daily_recommendations,
    get_pre_market_outlook,
    generate_trading_simulation_df,
    visualize_performance_metric_distribution_for_each_forecast_threshold,
    visualize_impact_of_threshold_on_performance_metric
)

from analyticsHub.helper import (
    _get_chosen_performance_df
)

from utils.pipeline import get_split_dates

all_df = get_all_performances()
pre_market_outlook = get_pre_market_outlook()

st.sidebar.title("Analytics Hub")
app_mode = st.sidebar.radio(
    "Menu", 
    [
        "1. Pre-Market Outlook",
        "2. Model Performance", 
        "3. Trading Simulation",
        "4. Daily Recommendation",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("The Model Used for __Daily Recommendations__ and __Trading Simulation__ is the __Ensemble of Specific Ticker, Specific Industry, and IHSG Model__")

if app_mode == "1. Pre-Market Outlook":
    st.title("Pre-Market Macro Outlook")
    st.markdown(f"**Generated at:** {pre_market_outlook['timestamp']}")

    overall = pre_market_outlook["overall_outlook"]
    outlook_label = overall["outlook"]

    outlook_colors = {
        "Bullish": "🟢",
        "Neutral-to-Bullish": "🟡",
        "Neutral-to-Bearish": "🟠",
        "Bearish": "🔴",
    }
    outlook_icon = outlook_colors.get(outlook_label, "⚪")

    st.markdown(f"### {outlook_icon} Overall Outlook: **{outlook_label}** (Score: {overall['composite_score']:+.2f})")
    st.caption(overall["rationale"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        vix = pre_market_outlook["vix"]
        st.markdown("#### 📊 VIX")
        st.metric(
            label="VIX Level",
            value=f"{vix['value']:.2f}",
        )
        st.markdown(f"**Tier:** {vix['classification']['tier']}")
        st.markdown(f"**Sentiment:** {vix['classification']['sentiment']}")
        st.markdown(f"**Percentile:** {vix['percentile_rank']:.1f}th")
        st.caption(vix["classification"]["description"])

    with col2:
        usdidr = pre_market_outlook["usdidr"]
        st.markdown("#### 💱 USD/IDR")
        st.metric(
            label="Exchange Rate",
            value=f"{usdidr['value']:,.2f}",
            delta=f"{usdidr['classification']['change_pct']:+.2f}%",
            delta_color="inverse",
        )
        st.markdown(f"**Tier:** {usdidr['classification']['tier']}")
        st.markdown(f"**Sentiment:** {usdidr['classification']['sentiment']}")
        st.markdown(f"**vs 20-day SMA:** {usdidr['classification']['vs_sma_20_pct']:+.2f}%")
        st.caption(usdidr["classification"]["description"])

    col3, col4 = st.columns(2)

    with col3:
        sp500 = pre_market_outlook["sp500"]
        st.markdown("#### 🇺🇸 S&P 500")
        st.metric(
            label="Previous Close",
            value=f"{sp500['value']:,.2f}",
            delta=f"{sp500['classification']['change_pct']:+.2f}%",
        )
        st.markdown(f"**Tier:** {sp500['classification']['tier']}")
        st.markdown(f"**Sentiment:** {sp500['classification']['sentiment']}")
        st.markdown(f"**Percentile:** {sp500['percentile_rank']:.1f}th")
        st.caption(sp500["classification"]["description"])

    with col4:
        nikkei = pre_market_outlook["nikkei"]
        data_label = "Live Intraday" if nikkei["is_live"] else "Previous Close"
        st.markdown(f"#### 🇯🇵 Nikkei 225 ({data_label})")
        st.metric(
            label=data_label,
            value=f"{nikkei['value']:,.2f}",
            delta=f"{nikkei['classification']['change_pct']:+.2f}%",
        )
        st.markdown(f"**Tier:** {nikkei['classification']['tier']}")
        st.markdown(f"**Sentiment:** {nikkei['classification']['sentiment']}")
        st.markdown(f"**Percentile:** {nikkei['percentile_rank']:.1f}th")
        st.caption(nikkei["classification"]["description"])

elif app_mode == "2. Model Performance":
    st.title("Model Performance")
    st.markdown("Inspect The Performance for Each Variations of the Model")
    
    chosen_model_versions = st.multiselect("Pick the Model's Version", all_df['model_version'].unique())
    chosen_model_label_types = st.multiselect("Pick the Model's Label Type", all_df['label_type'].unique())
    chosed_model_windows = st.multiselect("Pick the Model's Window", all_df['window'].unique())

    selected_model_identifier, selected_performance_df = _get_chosen_performance_df(all_df, chosen_model_versions, chosen_model_label_types, chosed_model_windows)
    for model_identifier, performance_df in zip(selected_model_identifier, selected_performance_df):
        st.write(f"### {model_identifier}")
        st.dataframe(performance_df)

elif app_mode == "3. Trading Simulation":
    st.title("Trading Simulation")

    trading_simulation_rolling_window = st.selectbox("Pick the Forecast Rolling Window", [val.stem for val in Path('data/stock/forecast/model_v4/medianGain').iterdir()])

    trading_simulation_df = generate_trading_simulation_df(trading_simulation_rolling_window)

    splits = get_split_dates(f'Median Gain {trading_simulation_rolling_window}')
    start_testing_market_date = splits['test']['start_date']
    end_testing_market_date = splits['test']['end_date']

    fig_1_profit = visualize_performance_metric_distribution_for_each_forecast_threshold(trading_simulation_df, trading_simulation_rolling_window, 'Profit')
    fig_1_loss = visualize_performance_metric_distribution_for_each_forecast_threshold(trading_simulation_df, trading_simulation_rolling_window, 'Loss')

    fig_2_profit = visualize_impact_of_threshold_on_performance_metric(trading_simulation_df, trading_simulation_rolling_window, 'Profit') 
    fig_2_loss = visualize_impact_of_threshold_on_performance_metric(trading_simulation_df, trading_simulation_rolling_window, 'Loss') 

    st.markdown(f"Simulating a Trading Activity Following the Output of The Model on the __Testing Data__ (from __{start_testing_market_date}__ to __{end_testing_market_date}__)")

    st.plotly_chart(fig_1_profit)
    st.plotly_chart(fig_1_loss)

    st.plotly_chart(fig_2_profit)
    st.plotly_chart(fig_2_loss)

elif app_mode == "4. Daily Recommendation":
    st.title("Daily Recommendation")
        
    daily_recommend_rolling_window = st.selectbox("Pick the Forecast Rolling Window", [val.stem for val in Path('data/stock/forecast/model_v4/medianGain').iterdir()])
    
    forecast_df, forecast_date = get_daily_recommendations(daily_recommend_rolling_window)

    st.markdown(f"Daily Recommendation on __{forecast_date}__")

    st.dataframe(forecast_df)