import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def roast_category(roast):
    categories = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }

    return categories.get(roast, float("nan"))

df = pd.read_csv("coffee_analysis.csv")

# Convert roast categories into numerical values
df["roast_cat"] = df["roast"].map(roast_category)

# Check the results
print(df[["roast", "roast_cat"]].head(10))

print(df.head())

# Select the feature used to make predictions
X = df[["100g_USD"]]

# Select the value we want to predict
y = df["rating"]

# Remove rows with missing values
valid_rows = X["100g_USD"].notna() & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Create the linear regression model
model = LinearRegression()

# Train the model using coffee prices and ratings
model.fit(X, y)

print("Model trained successfully!")

# Save the trained model
with open("model_1.pickle", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

# Select the two features for the decision tree
X_tree = df[["100g_USD", "roast_cat"]]

# Select the rating we want to predict
y_tree = df["rating"]

# Remove rows with missing ratings or prices
valid_rows_tree = X_tree["100g_USD"].notna() & y_tree.notna()

X_tree = X_tree.loc[valid_rows_tree]
y_tree = y_tree.loc[valid_rows_tree]

# Create and train the decision tree model
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_tree, y_tree)

print("Decision tree trained successfully!")

# Save the trained decision tree model
with open("model_2.pickle", "wb") as file:
    pickle.dump(dtr, file)

print("Decision tree saved successfully!")