import pandas as pd         
import numpy as np          
import matplotlib.pyplot as plt 
import seaborn as sns       

from sklearn.linear_model import LinearRegression           
from sklearn.model_selection import train_test_split        
from sklearn.metrics import mean_squared_error, r2_score    


# ----- 1. Loading Simulated Data -----
np.random.seed(42) 
n_samples = 100 

# Generate Age between 25 and 70
age = np.random.randint(25, 70, n_samples)
# Generate Cholesterol between 150 and 250
cholesterol = np.random.randint(150, 250, n_samples)

# Generate Blood Pressure (BP) based on a linear relationship with Age and Cholesterol, plus some random noise
bp = 0.8 * age + 0.2 * cholesterol + np.random.normal(0, 10, n_samples)

# Create a Pandas DataFrame from simulated data
df = pd.DataFrame({
    'Age': age,
    'Cholesterol': cholesterol,
    'BP': bp
})

print("--- Data Head (First 5 rows) ---")
print(df.head())        # Shows the first few rows of the DataFrame

print("\n--- Data Information (Data Types, Non-Null Counts) ---")
df.info()               # Gives a summary of the DataFrame, including data types and missing values

print("\n--- Data Description (Summary Statistics) ---")
print(df.describe())    # Provides descriptive statistics like mean, min, max, quartiles


print("-"*100)


# ----- 2. Exploratory Data Analysis (EDA) -----

print("\n--- Pairplot (visualizing relationships between variables) ---")
# A pairplot creates a grid of scatter plots for each pair of variables,
# and histograms/KDE plots for single variables.
sns.pairplot(df)
plt.suptitle('Pairplot of BP, Age, and Cholesterol', y=1.02) 
plt.show()

# Correlation matrix: Quantifies linear relationships
print("\n--- Correlation Matrix ---")
# df.corr() calculates the Pearson correlation coefficient between all pairs of columns.
# Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
print(df.corr())


print("-"*100)


# ----- 3. Simple Linear Regression (BP ~ Age) with Scikit-learn -----

# Define the independent variable (X) and dependent variable (y)
# Scikit-learn expects X to be a 2D array (DataFrame or NumPy array),
# even if it's just a single feature, so need double square brackets [['Age']].
X_simple = df[['Age']]
y = df['BP']

# Create an instance of the LinearRegression model
model_simple = LinearRegression()

# Train the model 
# The .fit() method finds the best-fit line (i.e., calculates the intercept and coefficient)
model_simple.fit(X_simple, y)

print("\n--- Simple Linear Regression Results (BP ~ Age) ---")
# The intercept (beta_0) is the predicted BP when Age is 0
print(f"Intercept (BP at Age 0): {model_simple.intercept_:.4f}")

# The coefficient (beta_1) tells us how much BP changes for every one-unit increase in Age
print(f"Coefficient (Age): {model_simple.coef_[0]:.4f}") # .coef_ returns an array, so we access the first element

# R-squared: This metric tells us the proportion of variance in the dependent variable (BP)
# that can be explained by the independent variable (Age). 
# Higher is better (0 to 1).
print(f"R-squared: {model_simple.score(X_simple, y):.4f}")

# --- Plotting the Simple Regression Line ---
plt.figure(figsize=(8, 6)) 
sns.scatterplot(x='Age', y='BP', data=df) 
# Plot the regression line: predict BP values for all ages in our data
plt.plot(df['Age'], model_simple.predict(X_simple), color='red', linewidth=2)
plt.title('Simple Linear Regression: Blood Pressure vs. Age')
plt.xlabel('Age (Years)')
plt.ylabel('Blood Pressure')
plt.grid(True) 
plt.show()


print("-"*100)


# ----- 4. Multiple Linear Regression (BP ~ Age + Cholesterol) -----

# Define the independent variables (X) with multiple features
# Again, X needs to be a 2D array (DataFrame)
X_multiple = df[['Age', 'Cholesterol']]
# y remains the same

# Create another instance of the Linear Regression model
model_multiple = LinearRegression()

# Fit the model with multiple independent variables
model_multiple.fit(X_multiple, y)

print("\n--- Multiple Linear Regression Results (BP ~ Age + Cholesterol) ---")
print(f"Intercept: {model_multiple.intercept_:.4f}")

# The .coef_ attribute will now be an array containing a coefficient for each feature
# The order matches the order of columns in X_multiple (Age, then Cholesterol)
print(f"Coefficients (Age, Cholesterol): {model_multiple.coef_}")

# R-squared for the multiple regression model
print(f"R-squared: {model_multiple.score(X_multiple, y):.4f}")


print("-"*100)


# ----- 5. Making Predictions with the Multiple Regression Model -----
# Predict BP for a new hypothetical person
# This person is 55 years old with a cholesterol level of 220
new_data_point = pd.DataFrame({'Age': [55], 'Cholesterol': [220]})

# Use the .predict() method to get the prediction
predicted_bp_new_person = model_multiple.predict(new_data_point)
print(f"\nPredicted BP for a person (Age=55, Cholesterol=220): {predicted_bp_new_person[0]:.4f}")


print("-"*100)


# ----- Optional: Calculating Residuals for the multiple model -----

# Residuals are the differences between actual and predicted values
df['Predicted_BP_multiple'] = model_multiple.predict(X_multiple)
df['Residuals_multiple'] = df['BP'] - df['Predicted_BP_multiple']
print("\nOriginal BP vs. Predicted BP (Multiple Regression):")
print(df[['BP', 'Predicted_BP_multiple', 'Residuals_multiple']].head())

# ----- Optional: Mean Squared Error (another common evaluation metric) -----

# A lower MSE means the model's predictions are closer to the actual values, i.e. indicates better model performance
mse = mean_squared_error(y, df['Predicted_BP_multiple'])
print(f"\nMean Squared Error (MSE): {mse:.4f}")
