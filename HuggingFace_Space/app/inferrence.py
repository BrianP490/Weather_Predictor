# main.py
import json, logging, argparse, time, torch, joblib, os, sys
import streamlit as st
from scripts import ModuleLayer, Agent
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# Same feature order names and order as during the data pipeline during model training
feature_names = ['DAY_OF_YEAR', 'PRECIPITATION', 'LAGGED_PRECIPITATION', 'AVG_WIND_SPEED', 'MIN_TEMP']

def convert_inputs(*args) -> list:
    """Convert user inputs into a list of features for the model.
    Args:
        *args: Variable length argument list containing user inputs in the following order:
            age (int): Age of the individual (18-64)
            sex (str): Sex of the individual
            bmi (float): BMI of the individual
            children (int): Number of children of the individual
            smoker (bool): Age of the individual
            region (str): Region of the individual
    Returns:
        features: A list of converted features ready for model input.
            """
    features = []

    regions_dict = {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}

    try:
        
        # age
        age = args[0]
        if not (18 <= age <= 64):
            raise ValueError("Age out of range.")
        features.append(float(age))

        # sex
        sex = args[1]
        if not isinstance(sex, str):
            raise ValueError("Sex must be a string.")
        features.append(1.0 if sex.lower() == 'male' else 0.0)

        # bmi
        bmi = args[2]
        if not (15.96 <= bmi <= 53.13):
            raise ValueError("BMI out of range.")
        features.append(float(bmi))

        # children
        children = args[3]
        if not (0 <= children <= 5):
            raise ValueError("Children out of range.")
        features.append(float(children))

        # smoker
        smoker = args[4]
        if not isinstance(smoker, str):
            raise ValueError("Smoker must be a string.")
        features.append(1.0 if smoker.lower() == 'yes' else 0.0)

        # region
        region = args[5].lower()
        if region not in regions_dict:
            raise ValueError("Region not recognized.")
        features.append(float(regions_dict[region]))

    except Exception as e:
        st.error(f"Error in input conversion: {e}")

    return features

# ## Main Loop
# Call this function, during script execution; Main script entry point
if __name__ == '__main__':
    st.title("Agent")
    st.subheader("Predicts the MAX Temperature (F) of a given day", divider=True)

    MODEL_DIRECTORY = './model_weights'
    MODEL_WEIGHTS_FILE_NAME = 'trained-model.pt'
    MODEL_WEIGHTS_FULL_PATH = Path(__file__).parent / MODEL_DIRECTORY / MODEL_WEIGHTS_FILE_NAME

    CONFIG_DIRECTORY = './configs'
    CONFIG_FILE_NAME = 'config.json'
    CONFIG_PATH = Path(__file__).parent / CONFIG_DIRECTORY / CONFIG_FILE_NAME

    SCALER_DIRECTORY = './scalers'
    FEATURE_SCALER_FILE_NAME = 'feature-scaler.joblib'
    FEATURE_SCALER_PATH = Path(__file__).parent / SCALER_DIRECTORY / FEATURE_SCALER_FILE_NAME


    try:
        model_weights = torch.load(MODEL_WEIGHTS_FULL_PATH, weights_only=True)
        print(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")
    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{MODEL_WEIGHTS_FULL_PATH}'. Please ensure the file exists or fix path to file.")
        sys.exit(1)

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{CONFIG_PATH}'. Please ensure the file exists or fix path to file.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        sys.exit(1)


    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        print(f"✅ Model weights loaded successfully from {MODEL_WEIGHTS_FULL_PATH}")

    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{FEATURE_SCALER_PATH}'. Please ensure the file exists or fix path to file.")
    except Exception as e:
        print(f"❌ An unexpected error occurred when loading scalers: {e}")
        sys.exit(1)


    MODEL_CONFIG = config.get("model", {})
    
    agent = Agent(cfg=MODEL_CONFIG)    # Create agent instance
    try:
        agent.load_state_dict(state_dict=model_weights)
    except Exception as e:
        print(f"❌ An error occurred while loading model weights: {e}")
        sys.exit(1)
    agent.eval().to('cpu')

    with st.form("my_form"):
        st.write("Please provide the following information:")

        # User inputs
        d_o_y = st.number_input("What day of the year is it (1-355)?", min_value=1, max_value=365, value=220, step=1, key="d_o_y")
        precip = st.slider("What are today's expected precipitation levels (in inches)?", min_value=0, max_value=10, value=.1, key="precip")
        l_precip = st.slider("What are yesterday's precipitation levels (in inches)?", min_value=0, max_value=20, value=.1, key="l_precip")
        wind = st.number_input("What is the average wind speed for today?", min_value=0, max_value=75, value=2, step=1, key="wind")
        min_temp = st.number_input("What is the minimum temperature for today?", min_value=0, max_value=75, value=2, step=1, key="min_temp")


        # Process the inputs and sample from the model
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            INPUTS=[d_o_y, precip, l_precip, wind, min_temp]
            # Create a list of features from the user's inputs
            converted_inputs = convert_inputs(**INPUTS)

            # Create a DataFrame for the scaler using the feature names to prevent warnings
            input_df = pd.DataFrame([converted_inputs], columns=feature_names)

            # Transform input tensor
            inputs = feature_scaler.transform(input_df)  # Scale the inputs using the pre-fitted scaler

            inputs = torch.tensor(inputs, dtype=torch.float32) # Convert to tensor

            pred = agent.get_prediction(inputs)
            st.success(f"Agent Predicts: **{pred:.2f}** (°F)")
