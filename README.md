# Regression Examples in Python

This repository contains beginner-friendly implementations of **Simple Linear Regression** and **Multiple Linear Regression** using simulated data. It’s designed to help reinforce fundamental machine learning concepts and demonstrate how to apply them using Python and `scikit-learn`.

## 📘 Theory: Multiple Linear Regression Explained

The formula for **Multiple Linear Regression** is:

    y=β0 + β1x1 + β2x2 + ... βnxn + ϵ

where:
- y: The Dependent Variable (our target, what we want to predict, e.g., SystolicBP).
- beta_0: The y-intercept. This is the predicted value of y when all independent variables (x_1,x_2,...,x_n) are zero.
- x_1,x_2,...,x_n: These are the Independent Variables (or Features; e.g., x_1 could be Age, x_2 could be Cholesterol, x_3 could be Weight).
- beta_1, beta_2,..., beta_n: These are the Coefficients for each independent variable. They are the "slopes" associated with each x.
    - A beta_i (e.g., beta_1) represents the expected change in y for a one-unit increase in that specific x_i, while holding all other independent variables constant. 
- epsilon: This is the Error Term (or Residual). It represents the part of y that the model cannot explain. It includes random variability and the effects of any factors not included in the model. 

A residual is the difference between the actual observed value (y_actual) and the value predicted by your model (y_predicted):

    ϵ = y(actual) - y(predicted)

- A positive residual means the model under-predicted (actual was higher than predicted).
- A negative residual means the model over-predicted (actual was lower than predicted).
- A residual of zero means the model predicted perfectly for that point.

- Positive residual: model underpredicted.
- Negative residual: model overpredicted.
- Residual = 0: perfect prediction.

---

## 📊 What This Project Covers

- ✅ Data simulation using NumPy (Age, Cholesterol, and Blood Pressure)
- ✅ Exploratory Data Analysis (EDA) using Seaborn and correlation matrices
- ✅ Simple Linear Regression (BP ~ Age)
- ✅ Multiple Linear Regression (BP ~ Age + Cholesterol)
- ✅ Model evaluation (R² score, residuals, Mean Squared Error)
- ✅ Visualisations of relationships and regression lines
- ✅ Making predictions for new (unseen) data points

---

## 📁 Files

| File | Description |
|------|-------------|
| `linear_regression.py` | The main Python script with inline commentary and implementation |
| `requirements.txt` | Dependencies for running the code |
| `README.md` | This file |

---

## 📦 Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:
```txt
matplotlib==3.10.3
numpy==2.3.1
pandas==2.3.0
scikit-learn==1.7.0
seaborn==0.13.2
```

---

## 🛠 How to Run

1.	Clone the repo:
```bash
git clone https://github.com/adabyt/linear_regression_intro.git
cd linear_regression_intro
```

2.	Install requirements:
```bash
pip install -r requirements.txt
```

3.	Run the script:
```bash
python linear_regression.py
```

---

## 🔍 Example Output
	•	Intercepts and coefficients printed to terminal
	•	R² values for both simple and multiple regression
	•	Prediction for a new sample: Age = 55, Cholesterol = 220
	•	Residual values
	•	Visualization of:
	•	Pairplot of features
	•	Simple regression line fit

---

## 🎓 Why This Project?

This project was created as part of my self-study in machine learning. It helped me:
	•	Understand linear models mathematically and programmatically
	•	Learn to evaluate models using metrics like R² and MSE
	•	Practice data visualization and simulation

I’ve intentionally kept learning comments and explanations to help others following a similar path.

---


