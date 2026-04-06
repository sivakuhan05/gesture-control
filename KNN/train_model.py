import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gesture_data.csv"
MODEL_FILE = BASE_DIR / "gesture_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
LABELS_FILE = BASE_DIR / "labels.pkl"
ALLOWED_LABELS = ["open", "close", "pointer", "ok"]

# Load dataset
data = pd.read_csv(DATA_FILE, header=None)

# Split features and labels
X = data.iloc[:, :-1]  # all columns except last
y = data.iloc[:, -1]   # last column = label

# Keep only the gestures we want to classify.
mask = y.isin(ALLOWED_LABELS)
X = X[mask]
y = y[mask]

if len(X) == 0:
    raise ValueError(
        "No matching rows found for labels: "
        + ", ".join(ALLOWED_LABELS)
        + ". Please recollect data."
    )

label_counts = y.value_counts()
missing_labels = [label for label in ALLOWED_LABELS if label not in label_counts.index]
if missing_labels:
    raise ValueError(
        "Missing label data for: "
        + ", ".join(missing_labels)
        + ". Collect samples for all labels before training."
    )

# Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model (small search over k).
best_model = None
best_score = -1.0
for k in [3, 5, 7, 9, 11]:
    candidate = KNeighborsClassifier(n_neighbors=k, weights="distance")
    candidate.fit(X_train, y_train)
    score = candidate.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = candidate

model = best_model

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nLabel counts:")
print(label_counts)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, model.predict(X_test), labels=ALLOWED_LABELS))
print("\nClassification report:")
print(classification_report(y_test, model.predict(X_test), labels=ALLOWED_LABELS, zero_division=0))

# Save model and scaler
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(ALLOWED_LABELS, f)

print("Model saved!")
