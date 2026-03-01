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
    generate_trading_simulation_df,
    visualize_profit_distribution_for_each_forecast_threshold,
    visualize_impact_of_threshold_on_profit
)

from analyticsHub.helper import _get_chosen_performance_df

all_df = get_all_performances()
forecast_df, forecast_date = get_daily_recommendations()
trading_simulation_df = generate_trading_simulation_df()
fig_1 = visualize_profit_distribution_for_each_forecast_threshold(trading_simulation_df)
fig_2 = visualize_impact_of_threshold_on_profit(trading_simulation_df)

st.sidebar.title("Analytics Hub")
app_mode = st.sidebar.radio(
    "Menu", 
    [
        "1. Model Performance", 
        "2. Daily Recommendation", 
        "3. Trading Simulation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("The Model Used for __Daily Recommendations__ and __Trading Simulation__ is the __Ensemble of Specific Ticker, Specific Industry, and IHSG Model__")

if app_mode == "1. Model Performance":
    st.title("Model Performance")
    st.markdown("Inspect The Performance for Each Variations of the Model")
    
    chosen_model_versions = st.multiselect("Pick the Model's Version", all_df['model_version'].unique())
    chosen_model_label_types = st.multiselect("Pick the Model's Label Type", all_df['label_type'].unique())
    chosed_model_windows = st.multiselect("Pick the Model's Window", all_df['window'].unique())

    selected_model_identifier, selected_performance_df = _get_chosen_performance_df(all_df, chosen_model_versions, chosen_model_label_types, chosed_model_windows)
    for model_identifier, performance_df in zip(selected_model_identifier, selected_performance_df):
        st.write(f"### {model_identifier}")
        st.dataframe(performance_df)

elif app_mode == "2. Daily Recommendation":
    st.title("Daily Recommendation")
    st.markdown(f"Daily Recommendation on __{forecast_date}__")

    st.dataframe(forecast_df)

elif app_mode == "3. Trading Simulation":
    st.title("Trading Simulation")
    # st.markdown(f"Daily Recommendation for {forecast_date} from the Final Model")

    st.plotly_chart(fig_1)

    st.plotly_chart(fig_2)