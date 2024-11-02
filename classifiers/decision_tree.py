import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Collect data and labels
data = []
labels = []
test_data = []
test_labels = []

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

subjects = []
for directory in os.listdir(parent_dir):
    if directory.startswith("subject") and directory[7:].isdigit():
        subjects.append(int(directory[7:]))

subjects.sort()

for subject_num in subjects:
    extracted_features_dir = os.path.join(parent_dir, f"subject{subject_num}", "extracted-features")
    feature_files = os.listdir(extracted_features_dir)
    feature_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    subject_data = []
    for filename in feature_files:
        file_path = os.path.join(extracted_features_dir, filename)
        with open(file_path, "r") as feature_file:
            signature_data = []
            for line in feature_file:
                signature_data.extend([float(x) for x in line.strip().split(", ")])
            subject_data.append(signature_data)


    # Split into training and testing: 10 signatures for training, the rest for testing
    np.random.shuffle(subject_data)
    training_data = subject_data[:10]
    testing_data = subject_data[10:]

    # Assuming the first 10 signatures are genuine and the rest are forgeries
    data.extend(training_data)
    labels.extend([subject_num] * 10)
    test_data.extend(testing_data)
    test_labels.extend([subject_num] * len(testing_data))

X_train = np.array(data)
y_train = np.array(labels)

X_test = np.array(test_data)
y_test = np.array(test_labels)

# Ensure consistent lengths
assert len(X_train) == len(y_train), "Inconsistent number of samples in training data and labels"
assert len(X_test) == len(y_test), "Inconsistent number of samples in test data and labels"

# Train Decision Tree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predict on test data
y_pred = tree.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred))