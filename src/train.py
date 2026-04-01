import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("data/data.csv")

# Convert categorical columns to numeric
df = pd.get_dummies(df)

# Choose features (will be modified this for different runs)
features = df.columns.drop("species")
n_estimators = 300
max_depth = None

X = df[features]
y = df["species"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# experiment
mlflow.set_experiment("2022bcd0053_experiment")

# run
with mlflow.start_run():
    
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # log
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("features", str(list(features)))

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)

import joblib
joblib.dump(model, "model.pkl")