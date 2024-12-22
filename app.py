import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from flask import Flask, request, render_template, jsonify
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the dataset
data_path = "C:/Users/USER/Desktop/Assignment_Data.csv"
data = pd.read_csv(data_path)

# Data Preprocessing
# Handle missing values in numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'PaymentMethod', 'Churn']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature Engineering
data['AverageSpendPerMonth'] = data['TotalCharges'] / data['Tenure']
data['ServiceUsageTotal'] = data['ServiceUsage1'] + data['ServiceUsage2'] + data['ServiceUsage3']

# Define features and target
X = data.drop(columns=['CustomerID', 'Churn'])
y = data['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# Select the best model
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")
print("Classification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# Save the scaler and model
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("churn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Recommendation system setup
recommendation_features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceUsageTotal']
recommendation_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
recommendation_model.fit(data[recommendation_features])

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('input_form.html', churn_probability=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from form
        customer_data = {
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'Tenure': float(request.form['Tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
            'PaymentMethod': request.form['PaymentMethod'],
            'ServiceUsage1': float(request.form['ServiceUsage1']),
            'ServiceUsage2': float(request.form['ServiceUsage2']),
            'ServiceUsage3': float(request.form['ServiceUsage3'])
        }

        # Transform categorical variables using LabelEncoders
        for key in ['Gender', 'PaymentMethod']:
            customer_data[key] = label_encoders[key].transform([customer_data[key]])[0]

        # Calculate additional features
        customer_data['AverageSpendPerMonth'] = customer_data['TotalCharges'] / customer_data['Tenure']
        customer_data['ServiceUsageTotal'] = customer_data['ServiceUsage1'] + customer_data['ServiceUsage2'] + customer_data['ServiceUsage3']

        # Create DataFrame for the input
        input_data = pd.DataFrame([customer_data])

        # Load scaler and model
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model = pickle.load(open("churn_model.pkl", "rb"))

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Predict churn probability
        churn_probability = model.predict_proba(scaled_data)[0, 1]

        return render_template('input_form.html', churn_probability=round(churn_probability, 2))
    except Exception as e:
        return render_template('input_form.html', churn_probability=None, error=str(e))

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        customer_id = request.json['CustomerID']
        if customer_id not in data['CustomerID'].values:
            return jsonify({"error": "CustomerID not found"})
        
        customer_index = data[data['CustomerID'] == customer_id].index[0]
        distances, indices = recommendation_model.kneighbors([data.loc[customer_index, recommendation_features]])
        recommendations = data.iloc[indices[0]].drop(index=customer_index)
        return jsonify({"recommendations": recommendations['CustomerID'].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
