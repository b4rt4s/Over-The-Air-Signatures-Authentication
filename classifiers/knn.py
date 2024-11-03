import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Funkcje do obliczania FAR i FRR
def calculate_FRR_and_FAR(conf_matrix):
    total_false_rejections = 0
    total_false_acceptances = 0
    total_genuine_attempts = 0
    total_impostor_attempts = 0
    num_classes = len(conf_matrix)
    total_samples = np.sum(conf_matrix)
    
    for i in range(num_classes):
        # Genuine attempts for class i
        genuine_attempts_i = np.sum(conf_matrix[i, :])
        total_genuine_attempts += genuine_attempts_i
        # False rejections for class i (off-diagonal in row i)
        false_rejections_i = genuine_attempts_i - conf_matrix[i, i]
        total_false_rejections += false_rejections_i
        
        # Impostor attempts for class i (samples not belonging to class i)
        impostor_attempts_i = total_samples - genuine_attempts_i
        total_impostor_attempts += impostor_attempts_i
        # False acceptances for class i (off-diagonal in column i)
        false_acceptances_i = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        total_false_acceptances += false_acceptances_i
        
    FRR = total_false_rejections / total_genuine_attempts if total_genuine_attempts > 0 else 0.0
    FAR = total_false_acceptances / total_impostor_attempts if total_impostor_attempts > 0 else 0.0
    return FRR, FAR

# Przygotowanie danych
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
    
    # Losowe przemieszanie danych i podział na treningowe i testowe
    np.random.shuffle(subject_data)
    training_data = subject_data[:10]  # Pierwsze 10 do treningu
    testing_data = subject_data[10:]   # Reszta do testu

    data.extend(training_data)
    labels.extend([subject_num] * len(training_data))
    test_data.extend(testing_data)
    test_labels.extend([subject_num] * len(testing_data))

X_train = np.array(data)
y_train = np.array(labels)

X_test = np.array(test_data)
y_test = np.array(test_labels)

assert len(X_train) == len(y_train), "Inconsistent number of samples in training data and labels"
assert len(X_test) == len(y_test), "Inconsistent number of samples in test data and labels"

# Trening klasyfikatora KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = knn.predict(X_test)

# Obliczanie macierzy konfuzji
conf_matrix = confusion_matrix(y_test, y_pred, labels=subjects)

# Obliczanie FAR i FRR
FRR, FAR = calculate_FRR_and_FAR(conf_matrix)

print(f"Total FRR: {FRR * 100:.2f}%")
print(f"Total FAR: {FAR * 100:.2f}%")
