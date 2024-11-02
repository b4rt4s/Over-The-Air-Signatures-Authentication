import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

    np.random.shuffle(subject_data)
    training_data = subject_data[:10]
    testing_data = subject_data[10:]

    data.extend(training_data)
    labels.extend([subject_num] * 10)
    test_data.extend(testing_data)
    test_labels.extend([subject_num] * len(testing_data))

X_train = np.array(data)
y_train = np.array(labels)

X_test = np.array(test_data)
y_test = np.array(test_labels)

assert len(X_train) == len(y_train), "Inconsistent number of samples in training data and labels"
assert len(X_test) == len(y_test), "Inconsistent number of samples in test data and labels"

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)

# Funkcje do obliczania FAR i FRR
def calculate_FAR(conf_matrix):
    total_false_accepts = 0
    total_impostor_attempts = 0
    num_classes = len(conf_matrix)
    for i in range(num_classes):
        false_accepts_for_i = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        total_false_accepts += false_accepts_for_i
        total_impostor_attempts_for_i = sum(sum(conf_matrix[j]) for j in range(num_classes) if j != i)
        total_impostor_attempts += total_impostor_attempts_for_i
    return total_false_accepts / total_impostor_attempts

def calculate_FRR(conf_matrix):
    total_false_rejects = 0
    total_genuine_attempts = 0
    num_classes = len(conf_matrix)
    for i in range(num_classes):
        false_rejects_for_i = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)
        total_false_rejects += false_rejects_for_i
        total_genuine_attempts_for_i = sum(conf_matrix[i])
        total_genuine_attempts += total_genuine_attempts_for_i
    return total_false_rejects / total_genuine_attempts


# Oblicz FAR i FRR
far = calculate_FAR(conf_matrix)
frr = calculate_FRR(conf_matrix)

print(f"FAR: {far:.2f}")
print(f"FRR: {frr:.2f}")
