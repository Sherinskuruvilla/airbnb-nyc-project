# 🌍 Airbnb Listings Price Prediction Using Machine Learning Models

## Objective

This project is for creating a model to predict Airbnb rental pricing in New York City based on location, and properties. Using a dataset containing information about location such as neighborhood group, latitude, longitude and properties of the rental apartment such as amenities, type of the room, accommodates, number of reviews etc. The goal is to find how accurately price can be predicted using supervised learning techniques.

---

## 🔍 Business Problem and Goal

New Airbnb investors in NYC struggle to find profitable rentals due to unclear pricing and demand insights. My Goal is to develop a model to predict listing prices using location and property features to guide smart investment decisions.

---

## 📊 Features Available:

- host_is_superhost(categorical_column)
- neighbourhood_group_cleansed(categorical_column)
- latitude(numerical_column)
- longitude(numerical_column)
- room_type(categorical_column)
- accommodates(numerical_column)
- bedrooms(numerical_column)
- price(numerical_column)-Target
- maximum_nights(numerical_column)
- availability_365(numerical_column)
- number_of_reviews(numerical_column)
- calculated_host_listings_count(numerical_column)
- total_amenities(numerical_column)

---

## 📘 Models Used:

- K-Nearest Neighbors (KNN)
- Decision Tree Regressor
- Random Forest Regressor
- Adaboost Regressor
- Gradient Boosting Regressor
---

## ✅ Expected Outcomes
- Identify which features have the strongest correlation with price.
- Evaluate the performance of multiple machine learning models in predicting price
- Compare models (e.g., KNN, Decision Tree, Random forest) to determine which offers the most accurate and reliable predictions.
- Understand how data preprocessing (such as feature selection and scaling) impacts model performance.

## ✅ Model Performance

🎯 Two models performed equally in terms of R2 score.
🌲 Random Forest regressor and Gradient Boosting regressor have R2 score of 0.76 
Other regressors have the following scores
🔹 Adaboost Regressor-0.75
🔹 KNN Regressor-0.71
🔹 Decision Tree Regressor-0.67
🧮 Error Metrics
  Adaboost Regressor performed well in terms of error metrics and has the lowest MAPE of 26.99

🌲Overall best performer-Random Forest Regressor with the following error metrics and score
      MAE,  44.54
      MSE,  4895.08
      RMSE,  69.96
      R2 score,  0.76
      MAPE,  27.49

⚙️ Model parameters were optimized using Optua,randomized search and grid search to maximize R2 and minimize error metrics

## 🧾 Dataset Description

## 🧱 Raw Datasets:
- **airbnbnyc2025.csv**

### Obstacles:
 
- Unstructured data- price and amenities needed cleaning
- Lot of metadata
-Wide price variability made the prediction tricky
  
---

## 💻 Technologies Used

| Area                 | Tools/Technologies                                      |
|----------------------|---------------------------------------------------------|
| Data Manipulation    | Python (Pandas, NumPy)                                  |
| Data Visualization   | Matplotlib, Seaborn, Pyplot                             |
| Documentation        | Jupyter Notebook, Markdown, GitHub,                     |
| Version Control      | Git, GitHub, Anaconda Powershell                        |
| Statistical Analysis | Scipy, statsmodels                                      |


---

## 📦 Deliverables

- ✅ [Repository "airbnb-nyc-project" on GitHub](https://github.com/Sherinskuruvilla/airbnb-nyc-project) 
- ✅ [Raw dataset](https://insideairbnb.com/get-the-data/)
- ✅ Jupyter Notebooks:
 - airbnbnyc_clean.ipynb
 - airbnbnyc_featureengineering.ipynb
 - airbnbnyc_regressionmodels.ipynb
- ✅ [ Trello](https://trello.com/b/brrBX69u/airbnb-newyork-price-prediction-ml-model)
- ✅ README documentation: README.md
-✅Presentation](https://docs.google.com/presentation/d/1B9ia4K43jkjU_EubWwtzHZ1kNZSL-ojrfKgdNC43UHA/edit?slide=id.g3644964c809_1_103#slide=id.g3644964c809_1_103) 
- A Streamlit App has created to predict the Airbnb rental price
---

## 👨‍💼 Target Audience

### Airbnb Investors
---

##  Contributor
- Sherin  Shaji Kuruvilla
---




