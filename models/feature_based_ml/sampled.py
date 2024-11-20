import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def calculate_rubine_features(drawing):
    # Flatten the drawing data to combine all strokes
    x = np.concatenate([stroke[0] for stroke in drawing])
    y = np.concatenate([stroke[1] for stroke in drawing])

    # Initial and final points
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # Feature 1 & 2: Cosine and Sine of the initial angle
    if len(x) > 1:
        initial_angle = np.arctan2(y[1] - y0, x[1] - x0)
        f1 = np.cos(initial_angle)
        f2 = np.sin(initial_angle)
    else:
        f1 = f2 = 0

    # Feature 3: Length of the gesture path
    f3 = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Feature 4: Angle at the first stroke endpoint
    if len(x) > 1:
        endpoint_angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    else:
        endpoint_angle = 0
    f4 = endpoint_angle

    # Feature 5: Bounding box diagonal length
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    f5 = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    # Feature 6: Angle of the bounding box diagonal
    f6 = np.arctan2(max_y - min_y, max_x - min_x)

    # Feature 7: Distance between the start and end points
    f7 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # Feature 8 & 9: Cosine and Sine of the angle between start and end points
    if f7 > 0:
        end_angle = np.arctan2(y1 - y0, x1 - x0)
        f8 = np.cos(end_angle)
        f9 = np.sin(end_angle)
    else:
        f8 = f9 = 0

    # Feature 10: Total angle traversed
    angles = np.arctan2(np.diff(y), np.diff(x))
    f10 = np.sum(np.diff(angles))

    # Feature 11: Total absolute angle traversed
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
                    label = data['word']
                    drawing = data['drawing']
                    rubine_features = calculate_rubine_features(drawing)
                    features.append(rubine_features)
                    labels.append(label)

    return features, labels


# Load and prepare data
folder = '../sampled/'
train_features, train_labels = load_data_from_folder(folder + 'training')
val_features, val_labels = load_data_from_folder(folder + 'validation')
test_features, test_labels = load_data_from_folder(folder + 'test')

X_train = np.array(train_features + val_features)
y_train = np.array(train_labels + val_labels)
X_test = np.array(test_features)
y_test = np.array(test_labels)

# List of classifiers to try
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier()
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy for {name}: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
