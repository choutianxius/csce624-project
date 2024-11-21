import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def calculate_rubine_features(drawing):
    x = np.concatenate([stroke[0] for stroke in drawing])
    y = np.concatenate([stroke[1] for stroke in drawing])
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    if len(x) > 1:
        initial_angle = np.arctan2(y[1] - y0, x[1] - x0)
        f1 = np.cos(initial_angle)
        f2 = np.sin(initial_angle)
    else:
        f1 = f2 = 0

    f3 = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    if len(x) > 1:
        endpoint_angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    else:
        endpoint_angle = 0
    f4 = endpoint_angle

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    f5 = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    f6 = np.arctan2(max_y - min_y, max_x - min_x)
    f7 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if f7 > 0:
        end_angle = np.arctan2(y1 - y0, x1 - x0)
        f8 = np.cos(end_angle)
        f9 = np.sin(end_angle)
    else:
        f8 = f9 = 0

    angles = np.arctan2(np.diff(y), np.diff(x))
    f10 = np.sum(np.diff(angles))
    f11 = np.sum(np.abs(np.diff(angles)))

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]


def load_data_from_folder(folder_path):
    features = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.ndjson'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    # label = data['word']
                    label = file_path.split('/')[-1].split('.')[0]
                    drawing = data['drawing']
                    rubine_features = calculate_rubine_features(drawing)
                    features.append(rubine_features)
                    labels.append(label)
    return features, labels


# Load and prepare data
folder = '../sampled_masked/'
train_features, train_labels = load_data_from_folder(folder + 'training')
val_features, val_labels = load_data_from_folder(folder + 'validation')
test_features, test_labels = load_data_from_folder(folder + 'test')

X_train = np.array(train_features + val_features)
y_train = np.array(train_labels + val_labels)
X_test = np.array(test_features)
y_test = np.array(test_labels)

# Encode labels for compatibility with classifiers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# List of classifiers to try
classifiers = {
    "RandomForest": RandomForestClassifier(),
    # "K-Nearest Neighbors": KNeighborsClassifier(),
    # "Support Vector Machine": SVC(probability=True),
    # "Logistic Regression": LogisticRegression(max_iter=1000),
    # "Decision Tree": DecisionTreeClassifier()
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train_encoded)

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        y_proba = clf.decision_function(X_test)
        y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1)  # Convert decision function output to probabilities
    else:
        print(f"{name} does not support probability prediction. Skipping.")
        continue

    # Get top 5 predictions for each sample
    top_5_predictions = np.argsort(y_proba, axis=1)[:, -5:]

    # Check if any of the top 5 predictions match the true label
    correct_top_5 = sum(
        y_test_encoded[i] in top_5_predictions[i]
        for i in range(len(y_test_encoded))
    )

    accuracy_top_5 = correct_top_5 / len(y_test_encoded)
    print(f"Top-5 Accuracy for {name}: {accuracy_top_5}")
