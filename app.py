from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

app = Flask(__name__)

# Load dataset
data = pd.read_csv('energy_consumption.csv')

# Preprocess the data
data['date'] = pd.to_datetime(data['date'], dayfirst=True)
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Features and Target
X = data[['year', 'month', 'temp', 'feels_like']]
y = data[['active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ✅ Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set! Check your .env file.")

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

ai_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="""
    You are an AI expert in energy management. Based on the predicted data, generate energy-saving tips in a clear, structured way and provide in point by point format in precise and short dont make it too long.
    """
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    
    # Predict values
    user_input = np.array([[year, month, 30, 35]])  # Example temp values
    predictions = model.predict(user_input).flatten()

    active_power, current, voltage = predictions[0], predictions[1], predictions[2]

    # Generate AI-based suggestions
    chat_session = ai_model.start_chat()
    ai_prompt = f"The predicted energy usage is:\nActive Power: {active_power:.2f} kW\nCurrent: {current:.2f} A\nVoltage: {voltage:.2f} V.\nProvide optimization suggestions."
    ai_response = chat_session.send_message(ai_prompt)
    ai_suggestions = ai_response.text.strip()

    response = {
        'predictions': predictions.tolist(),
        'ai_suggestions': ai_suggestions
    }
    
    return jsonify(response)

@app.route('/graph')
def graph():
    # Get the last actual data from dataset
    last_actual = y.iloc[-1].values[:3]  # Active Power, Current, Voltage

    # Example predicted values
    predicted = model.predict([X.iloc[-1].values]).flatten()[:3]

    categories = ["Active Power (kW)", "Current (A)", "Voltage (V)"]
    actual_values = last_actual
    predicted_values = predicted
    differences = np.abs(actual_values - predicted_values)

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, actual_values, width, label="Actual", color='blue')
    ax.bar(x, predicted_values, width, label="Predicted", color='orange')
    ax.bar(x + width, differences, width, label="Difference", color='red')

    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Actual vs Predicted Energy Data with Differences")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

   

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # ✅ Now runs on port 5001
