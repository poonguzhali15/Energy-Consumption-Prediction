# **Energy Consumption Prediction with AI-based Suggestions**

## **Project Description**
This project predicts household energy consumption based on historical data and provides AI-generated suggestions to optimize energy usage. The dataset includes energy consumption records from a residence in northeast Mexico, enriched with weather data from OpenWeather.

Users input a **year and month**, and the system generates a **graphical prediction** of energy consumption for that period. Additionally, the system uses **Generative AI (GenAI)** to suggest energy-saving strategies based on the predicted trends.

---

## **Problem Statement**
Managing household energy consumption efficiently is crucial for **reducing electricity costs** and **optimizing energy use**. However, predicting future consumption patterns is challenging due to fluctuations in **weather conditions, appliance usage, and seasonal variations**.

This project aims to **predict future energy consumption** using **machine learning (RandomForest Regressor)** and **provide AI-driven recommendations** to help users optimize their energy usage.

---

## **Solution Approach**
The project follows these steps:

1. **Data Collection**  
   - The dataset is sourced from **Mendeley Data**:  
     **Title:** *Household energy consumption enriched with weather data in northeast of Mexico*  
   - It includes **14 months of minute-level energy consumption** and **weather metrics**.

2. **Data Preprocessing**  
   - Cleaning missing values.  
   - Aggregating minute-level data into daily/monthly consumption.  
   - Extracting features like temperature, humidity, and energy usage patterns.

3. **Model Training**  
   - **Algorithm Used:** `RandomForestRegressor`  
   - The model learns patterns from past energy usage and weather data to predict future consumption.

4. **User Interaction and Prediction**  
   - The user enters **year and month** as input.  
   - The system **predicts energy consumption** for the selected period and visualizes it as a **graph**.

5. **AI-Based Suggestions**  
   - Based on the predicted energy consumption, the system **calls the GenAI API**.  
   - The AI **analyzes the trends** and provides **personalized suggestions** to optimize energy usage.

---

## **Libraries and Technologies Used**
- **Backend:** Flask  
- **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Generative AI API:** OpenAI/GenAI for recommendations  
- **Deployment:** Flask for the web application  

---

## **How the Project Works**
1. **User Inputs**  
   - The user selects a **year and month** for energy consumption prediction.

2. **Prediction Generation**  
   - The **RandomForest Regressor** model predicts energy consumption for the given period.  
   - A **graph is generated** to visualize the predicted usage.

3. **AI-Based Energy Optimization Suggestions**  
   - The system **calls the GenAI API** with the predicted consumption data.  
   - The AI **analyzes the data** and provides **energy-saving recommendations**.  
   - The user receives actionable insights to **reduce energy consumption**.

---

## **Example Usage**
1. User selects **"January 2025"**.
2. The system **predicts** energy consumption for January 2025.
3. A **graph** is displayed showing the predicted usage.
4. The AI **suggests** actions like:
   - Reduce AC usage on high-temperature days.
   - Optimize heating during peak hours.
   - Adjust lighting schedules to save energy.

---

## **Dataset used**
📌 **Dataset:** [Your Dataset Link Here](https://data.mendeley.com/datasets/tvhygj8rgg/1)

---




## **Conclusion**
This project combines **machine learning (RandomForest Regressor)** with **Generative AI-based recommendations** to help households **predict and optimize their energy consumption**. It provides users with **graphical insights** and **AI-driven suggestions** for smarter energy management.

---
