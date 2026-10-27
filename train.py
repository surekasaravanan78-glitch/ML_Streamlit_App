from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training successful!")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# Train on the complete dataset and save the final model
model.fit(X, y)
joblib.dump(model, "iris_model.pkl")

print("Model saved as iris_model.pkl")