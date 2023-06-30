import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
data = pd.read_csv('forest_fire_data.csv')

# Step 2: Data Preprocessing
# Perform data cleaning and feature engineering here

# Step 3: Exploratory Data Analysis
# Visualize the data and explore relationships between variables

# Step 4: Feature Selection
# Select the relevant features for predicting forest fires

# Step 5: Model Building
X = data.drop('fire_occurrence', axis=1)
y = data['fire_occurrence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 7: Deployment
# Deploy the model in a production environment
