import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.model_selection import GridSearchCV

# Przygotowanie danych
data = []
labels = []
test_data = []
test_labels = []

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

# Wczytanie numerów podpisów, które zostaną użyte do treningu modeli
selected_numbers_file = os.path.join(parent_dir, "results", "selected_numbers_file_and_features.txt")
with open(selected_numbers_file, 'r') as file:
    selected_numbers_line = file.readline().strip()
    selected_numbers = [int(x.strip()) for x in selected_numbers_line.split(",")]

# Wczytanie wszystkich numerów profil, tj. 56
subjects = []
for directory in os.listdir(parent_dir):
    if directory.startswith("subject") and directory[7:].isdigit():
        subjects.append(int(directory[7:]))

subjects.sort()

# Podział podpisów z każdego subjecta na dane treningowe i testowe
for subject_num in subjects:
    extracted_features_dir = os.path.join(parent_dir, f"subject{subject_num}", "extracted-features")
    feature_files = os.listdir(extracted_features_dir)
    feature_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    
    subject_training_data = []
    subject_testing_data = []
    
    for filename in feature_files:
        number = int(filename.split('-')[-1].split('.')[0])
        file_path = os.path.join(extracted_features_dir, filename)
        
        with open(file_path, "r") as feature_file:
            signature_data = []
            for line in feature_file:
                signature_data.extend([float(x) for x in line.strip().split(", ")])
        
        if number in selected_numbers:
            subject_training_data.append(signature_data)
        else:
            subject_testing_data.append(signature_data)
    
    data.extend(subject_training_data)
    labels.extend([subject_num - 1] * len(subject_training_data))
    test_data.extend(subject_testing_data)
    test_labels.extend([subject_num - 1] * len(subject_testing_data))

X_train = np.array(data)
y_train = np.array(labels)
X_test = np.array(test_data)
y_test = np.array(test_labels)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parametry do Grid Search
parameters = {'kernel':('linear', 'rbf', 'sigmoid', 'poly'), 'C':[0.1, 0.2, 0.3, 1, 5, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
svc = SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train_scaled, y_train)

# Predykcja na zbiorze testowym z najlepszym modelem
y_pred = clf.predict(X_test_scaled)

# Obliczanie macierzy konfuzji
conf_matrix = confusion_matrix(y_test, y_pred, labels=subjects)

# Obliczanie FAR i FRR
TP = np.diag(conf_matrix)
FN = np.sum(conf_matrix, axis=1) - TP
print(FN)
FP = np.sum(conf_matrix, axis=0) - TP
TN = np.sum(conf_matrix) - (TP + FP + FN)

FAR = sum(FP) / (sum(FP) + sum(TN))
FRR = sum(FN) / (sum(TP) + sum(FN))

print(f"TP: {sum(TP)}, TN:{sum(TN)}, FP: {sum(FP)}, FN: {sum(FN)}")
print(f"FAR = {round(FAR, 3)*100}%", f"FRR = {round(FRR, 3)*100}%")

# Wyświetlanie macierzy konfuzji
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=subjects)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
plt.title('Confusion Matrix')
plt.show()
