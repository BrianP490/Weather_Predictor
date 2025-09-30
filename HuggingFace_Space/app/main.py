# main.py
import torch
import streamlit as st
from scripts import Agent, ModuleLayer
import os
import joblib
st.title("Oil Predictor")

cfg = {
    "in_dim": 22,    # Number of Features as input
    "intermediate_dim": 128,    
    "out_dim": 1,   
    "num_blocks": 12,   # Number of reapeating Layer Blocks
    "dropout_rate": 0.1     # Rate for dropout layer
}

agent = Agent(cfg)    # Create agent instance

# Dynamically create the path to the model's weights 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory of current running file

weights_file = os.path.join(BASE_DIR, "model_weights", "Agent_02.pt") # create the full path to the model weights
# Create the full path to the scalers
features_scaler_file = os.path.join(BASE_DIR, "scalers", "feature-scaler.joblib") 
label_scaler_file = os.path.join(BASE_DIR, "scalers", "label-scaler.joblib") 

features_scaler = joblib.load(features_scaler_file) # Load feature scaler
label_scaler = joblib.load(label_scaler_file)     # Load label scaler

agent.load_state_dict(torch.load(weights_file, weights_only=True)) # Load the agent's model weights


# User inputs
open = st.number_input("What is the opening price of oil?", min_value=0.0, value=200.0, key="open")
high = st.number_input("What was the previous Highest price of oil?", min_value=0.0, value=200.0, key="high")
low = st.number_input("What was the previous Lowest price of oil?", min_value=0.0, value=200.0, key="low")
cali = st.number_input("What was the 'First purchase' price for crude oil in California of the previous month? ($/bbl)", min_value=0.0, value=200.0, key="cali")
texas = st.number_input("What was the 'First purchase' price for crude oil in Texas of the previous month? ($/bbl)", min_value=0.0, value=200.0, key="texas")
us_crude_first = st.number_input("What was the 'First purchase' average price in the U.S. of the previous month? ($/bbl)", min_value=0.0, value=200.0, key="us_crude_first")

US_Imports_from_Canada = st.number_input("What was the U.S. crude oil imports from Canada of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_Canada")
US_Imports_from_Colombia = st.number_input("What was the U.S. crude oil imports from Colombia of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_Colombia")
US_Imports_from_United_Kingdom = st.number_input("What was the U.S. crude oil imports from the United Kingdom of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_United_Kingdom")
US_Imports_from_Mexico = st.number_input("What was the U.S. crude oil imports from Mexico of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_Mexico")
US_Imports_from_OPEC_Countries = st.number_input("What was the U.S. crude oil imports from OPEC Countries of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_OPEC_Countries")
US_Imports_from_Non_OPEC_Countries = st.number_input("What was the U.S. crude oil imports from Non-OPEC Countries of the previous month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports_from_Non_OPEC_Countries")
US_Imports = st.number_input("What was the U.S. crude oil imports? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Imports")

US_Exports_to_Canada = st.number_input("How much oil did the U.S. export to Canada last month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Exports_to_Canada")
US_Exports = st.number_input("What was the U.S. crude oil exports last month? (thousand bbl/day)", min_value=0.0, value=200.0, key="US_Exports")

US_Net_Imports_from_Canada = st.number_input("What was the U.S. crude oil NET imports from Canada of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_Canada")
US_Net_Imports_from_Colombia = st.number_input("What was the U.S. crude oil NET imports from Colombia of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_Colombia")
US_Net_Imports_from_Mexico = st.number_input("What was the U.S. crude oil NET imports from Mexico of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_Mexico")
US_Net_Imports_from_United_Kingdom = st.number_input("What was the U.S. crude oil NET imports from the United Kingdom of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_United_Kingdom")
US_Net_Imports_from_OPEC_Countries = st.number_input("What was the U.S. crude oil NET imports from OPEC Countries of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_OPEC_Countries")
US_Net_Imports_from_Non_OPEC_Countries = st.number_input("What was the U.S. crude oil NET imports from OPEC Countries of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports_from_Non_OPEC_Countries")
US_Net_Imports = st.number_input("What was the U.S. crude oil NET imports of the previous month? (thousand bbl/day)", value=200.0, key="US_Net_Imports")



# Prepare input tensor
raw_inputs = (open, high, low, cali, texas, us_crude_first, US_Imports_from_Canada, US_Imports_from_Colombia, US_Imports_from_United_Kingdom, US_Imports_from_Mexico, US_Imports_from_OPEC_Countries, US_Imports_from_Non_OPEC_Countries, US_Imports, US_Exports_to_Canada, US_Exports, US_Net_Imports_from_Canada, US_Net_Imports_from_Colombia, US_Net_Imports_from_Mexico, US_Net_Imports_from_United_Kingdom, US_Net_Imports_from_OPEC_Countries, US_Net_Imports_from_Non_OPEC_Countries, US_Net_Imports)

inputs = features_scaler.transform([raw_inputs])  # Scale the inputs using the pre-fitted scaler

inputs = torch.tensor(inputs, dtype=torch.float32) # Convert to tensor

# Predict Price
if st.button("Get Agent's Prediction"):
    unnormalized_pred = agent.get_prediction(inputs)
    pred = label_scaler.inverse_transform([[unnormalized_pred]])[0,0]  # Un-normalize the prediction
    st.success(f"Agent Predicts The Closing Price is: **{pred:.7f}**")