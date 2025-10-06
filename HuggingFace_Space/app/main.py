"""This Module is the main entry point for the Streamlit application."""

# main.py
import sys
import json
import pandas as pd
import torch
import joblib
import streamlit as st
from scripts import Agent, convert_inputs
from scripts import (
    MODEL_WEIGHTS_FULL_PATH,
    CONFIG_PATH,
    FEATURE_SCALER_PATH,
    FEATURE_NAMES as feature_names
)

# Main Loop
# Call this function, during script execution; Main script entry point
if __name__ == '__main__':
    st.title("Agent")
    st.subheader("Predicts the MAX Temperature (F) of a given day", divider=True)

    # Load model weights
    try:
        model_weights = torch.load(MODEL_WEIGHTS_FULL_PATH, weights_only=True)
        print(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")
    except FileNotFoundError:
        print(
            f"❌ Model Weights file not found at '{MODEL_WEIGHTS_FULL_PATH}'. " "" \
            "Please ensure the file exists or fix path to file."
        )
        sys.exit(1)

    # Load configuration file
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{CONFIG_PATH}'. "
              "Please ensure the file exists or fix path to file.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        sys.exit(1)

    # Load feature scaler
    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        print(f"✅ Feature Scaler loaded successfully from {FEATURE_SCALER_PATH}")

    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{FEATURE_SCALER_PATH}'. "
              "Please ensure the file exists or fix path to file.")
        sys.exit(1)


    MODEL_CONFIG = config.get("model", {})

    try:
        agent = Agent(cfg=MODEL_CONFIG)    # Create agent instance
        agent.load_state_dict(state_dict=model_weights)
    except RuntimeError as e:
        print(f"❌ A runtime error occurred while creating model or loading model weights: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Model weights file not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing key in model configuration: {e}")
        sys.exit(1)


    agent.eval().to('cpu')

    with st.form("my_form"):
        st.write("Please provide the following information:")

        # User inputs
        d_o_y = st.number_input("What day of the year is it (1-355)?", min_value=1, max_value=365,
                                value=220, step=1, key="d_o_y")

        precip = st.slider("What are today's expected precipitation levels (in inches)?",
                           min_value=0.0, max_value=10.0, value=.1, key="precip")

        l_precip = st.slider("What are yesterday's precipitation levels (in inches)?",
                             min_value=0.0, max_value=20.0, value=.1, key="l_precip")

        wind = st.number_input("What is the average wind speed for today?", min_value=0,
                               max_value=75, value=2, step=1, key="wind")

        min_temp = st.number_input("What is the minimum temperature for today?", min_value=0,
                                   max_value=75, value=2, step=1, key="min_temp")


        # Process the inputs and sample from the model
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            # Create a list of features from the user's inputs
            INPUTS=[d_o_y, precip, l_precip, wind, min_temp]

            # Convert inputs to the correct format using the expansion operator
            converted_inputs = convert_inputs(*INPUTS)

            # Create a DataFrame for the scaler using the feature names to prevent warnings
            input_df = pd.DataFrame([converted_inputs], columns=feature_names)

            # Transform input tensor by scaling the inputs using the pre-fitted scaler
            inputs = feature_scaler.transform(input_df)

            inputs = torch.tensor(inputs, dtype=torch.float32) # Convert to tensor

            # Get the agent's prediction
            pred = agent.get_prediction(inputs)

            # Display the agent's prediction
            st.success(f"Agent Predicts: **{pred:.2f}** (°F)")
