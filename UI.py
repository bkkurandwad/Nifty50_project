import pandas as pd
import numpy as np
import streamlit as st
from stable_baselines3 import DQN

# Load the trained DQN model
model = DQN.load("dqn_trading_model")

# Load the dataset
data = pd.read_csv("processed_dataset.csv")
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date is in datetime format
data.set_index('Date', inplace=True)  # Set Date as the index for easier querying

# Function to get the state for prediction
def get_state_for_prediction(selected_date):
    try:
        # Retrieve the row corresponding to the selected date
        row = data.loc[selected_date]
        # Extract the required features
        state = row[['Close', 'SMA_20', 'EMA_10', 'RSI', 'Volatility']].values
        
        # Convert to numeric (handle any potential non-numeric values)
        state = pd.to_numeric(state, errors='coerce')  # Convert to numeric, replace invalid with NaN
        if np.isnan(state).any():
            raise ValueError("The state contains NaN values. Check the dataset for missing or invalid entries.")
        
        state = state.astype(float)  # Ensure the data type is float
        return state
    except KeyError:
        st.error(f"No data available for the selected date: {selected_date}.")
        return None
    except ValueError as e:
        st.error(str(e))
        return None

# Function to predict the action
def predict_action(state):
    state = state.reshape(1, -1)  # Reshape input to match model's expected input
    action, _ = model.predict(state, deterministic=True)
    return action

# Streamlit app
def main():
    st.title("DQN Trading Model Prediction")
    st.write("This application predicts trading actions (HOLD, BUY, or SELL) using a DQN model.")

    # Display the dataset for reference
    with st.expander("View Dataset"):
        st.dataframe(data)

    # Prediction mode selection
    st.header("Prediction Mode")
    choice = st.radio(
        "Choose your prediction mode:",
        options=["Predict for today (last available date)", "Predict for a specific past date"]
    )

    if choice == "Predict for today (last available date)":
        # Use the last available date in the CSV
        last_date = data.index[-1]
        st.write(f"Using last available date: **{last_date.date()}**")
        state = get_state_for_prediction(last_date)
    elif choice == "Predict for a specific past date":
        # Ask for a specific past date
        user_date = st.text_input("Enter the date (YYYY-MM-DD):")
        if user_date:
            try:
                selected_date = pd.to_datetime(user_date)
                state = get_state_for_prediction(selected_date)
            except ValueError:
                st.error("Invalid date format. Please enter a valid date in YYYY-MM-DD format.")
                state = None
    else:
        st.error("Invalid choice.")
        state = None

    # Predict the action if state is valid
    if state is not None:
        action = predict_action(state)
        if action == 0:
            st.success("Prediction: **HOLD**")
        elif action == 1:
            st.success("Prediction: **BUY**")
        elif action == 2:
            st.success("Prediction: **SELL**")
        else:
            st.error("Unknown action predicted.")

if __name__ == "__main__":
    main()