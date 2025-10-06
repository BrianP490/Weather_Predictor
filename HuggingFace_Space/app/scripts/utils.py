"""This module contains utility functions for input conversion and validation."""

import streamlit as st

def convert_inputs(*args) -> list:
    """
    Convert user inputs into a list of features for the model.
    Args:
        *args: Variable length argument list containing user inputs
            - Day of the year (int): Must be between 1 and 365.
            - Today's expected precipitation (float): Must be between 0.0 and 10.0 inches.
            - Yesterday's precipitation (float): Must be between 0.0 and 20.0 inches.
            - Wind speed (float): Must be between 0.0 and 75.0 mph.
            - Today's minimum temperature (float): Must be between 0.0 and 75.0 °F.

    Returns:
        features: A list of converted features ready for model input.
    """
    features = []   # Create empty list to store all the features

    try:

        # Day (#) of the year
        feature_1 = args[0]
        if not 1 <= feature_1 <= 365:
            raise ValueError("Age out of range.")
        features.append(float(feature_1))

        # Today's expected precipitation levels (in inches)
        feature_2 = args[1]
        if not 0.0 <= feature_2 <= 10.0:
            raise ValueError("Today's expected precipitation levels out of range.")
        features.append(float(feature_2))

        # Yesterday's precipitation levels (in inches)
        feature_3 = args[2]
        if not 0.0 <= feature_3 <= 20.0:
            raise ValueError(" Yesterday's precipitation levels  out of range.")
        features.append(float(feature_3))

        # Wind speed
        feature_4 = args[3]
        if not 0.0 <= feature_4 <= 75.0:
            raise ValueError("Wind speed out of range.")
        features.append(float(feature_4))

        # Today's minimum temperature
        feature_5 = args[4]
        if not 0.0 <= feature_5 <= 75.0:
            raise ValueError("Today's minimum temperature out of range.")
        features.append(float(feature_5))

    except IndexError as e:
        st.error(f"Error in indexing inputs: {e}")
    except TypeError as e:
        st.error(f"Type error in input conversion: {e}")
    except ValueError as e:
        st.error(f"Value error in input conversion: {e}")

    return features
