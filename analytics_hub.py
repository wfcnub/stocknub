import pandas as pd
import streamlit as st

from analyticsHub.main import (
    get_all_performances,
    get_daily_recommendations,
    visualize_performance_metric_distribution_for_each_forecast_threshold,
    visualize_impact_of_threshold_on_performance_metric
)

from analyticsHub.helper import (
    _get_available_score_windows,
    _get_chosen_performance_df
)

from utils.pipeline import get_split_dates
from utils import paths

all_df = get_all_performances()

st.sidebar.title("Analytics Hub")
app_mode = st.sidebar.radio(
    "Menu", 
    [
        "1. Model Performance", 
        "2. Trading Simulation",
        "3. Daily Recommendation",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("The Model Used for __Daily Recommendations__ and __Trading Simulation__ is the __Ensemble of Specific Ticker, Specific Industry, and IHSG Model__")

if app_mode == "1. Model Performance":
    st.title("Model Performance")
    st.markdown("Inspect The Performance for Each Variations of the Model")

    if all_df.empty:
        st.warning(
            "No model performance metrics are available. Train the models and "
            "refresh this page."
        )
        st.stop()
    
    chosen_model_versions = st.multiselect("Pick the Model's Version", all_df['model_version'].unique())
    chosen_model_label_types = st.multiselect("Pick the Model's Label Type", all_df['label_type'].unique())
    chosed_model_windows = st.multiselect("Pick the Model's Window", all_df['window'].unique())

    selected_model_identifier, selected_performance_df = _get_chosen_performance_df(all_df, chosen_model_versions, chosen_model_label_types, chosed_model_windows)
    for model_identifier, performance_df in zip(selected_model_identifier, selected_performance_df):
        st.write(f"### {model_identifier}")
        st.dataframe(performance_df)

elif app_mode == "2. Trading Simulation":
    st.title("Trading Simulation")

    trading_windows = _get_available_score_windows(
        require_simulation=True,
        require_split=True,
    )
    if not trading_windows:
        st.warning(
            "No complete score, simulation, and split artifacts are available. "
            "Run the score-generation pipeline stage and refresh this page."
        )
        st.stop()
    trading_simulation_rolling_window = st.selectbox(
        "Pick the Forecast Rolling Window",
        trading_windows,
    )

    trading_simulation_path = paths.get_trading_simulation_path(trading_simulation_rolling_window)
    trading_simulation_df = pd.read_csv(trading_simulation_path)

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

elif app_mode == "3. Daily Recommendation":
    st.title("Daily Recommendation")

    recommendation_windows = _get_available_score_windows(
        require_simulation=True,
    )
    if not recommendation_windows:
        st.warning(
            "No complete score and simulation artifacts are available. Run "
            "the score-generation pipeline stage and refresh this page."
        )
        st.stop()
    daily_recommend_rolling_window = st.selectbox(
        "Pick the Forecast Rolling Window",
        recommendation_windows,
    )
    
    forecast_df, forecast_date = get_daily_recommendations(daily_recommend_rolling_window)

    st.markdown(f"Daily Recommendation on __{forecast_date}__")

    st.dataframe(forecast_df)
