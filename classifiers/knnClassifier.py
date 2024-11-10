import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer

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
            # Przekształcenie danych na macierz 20x10
            # signature_data = np.array(signature_data).reshape(-1, 10).flatten() # nie ma wpływu na obliczenia
            # signature_data = np.array(signature_data).reshape(-1, 10)
            # # Obliczenie średniej i odchylenia standardowego dla każdej cechy
            # mean_features = np.mean(signature_data, axis=0)
            # std_features = np.std(signature_data, axis=0)
            # # Połączenie średnich i odchyleń w jeden wektor cech
            # signature_data = np.concatenate((mean_features, std_features))
        
        if number in selected_numbers:
            subject_training_data.append(signature_data)
        else:
            subject_testing_data.append(signature_data)
    
    # Dodajemy dane treningowe i testowe do ogólnych list
    data.extend(subject_training_data)
    labels.extend([subject_num] * len(subject_training_data))
    test_data.extend(subject_testing_data)
    test_labels.extend([subject_num] * len(subject_testing_data))

X_train = np.array(data)
y_train = np.array(labels)

X_test = np.array(test_data)
y_test = np.array(test_labels)

# Normalizacja danych
scaler = PowerTransformer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definiowanie zakresu K
param_grid = {'n_neighbors': list(range(1, 21))}

# Inicjalizacja klasyfikatora KNN
knn = KNeighborsClassifier(weights='distance')

# Inicjalizacja GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,  # liczba podziałów w cross-validation
    scoring='accuracy',  # możesz dostosować metrykę
    n_jobs=-1  # użyj wszystkich dostępnych rdzeni procesora
)

# Przeprowadzenie Grid Search
grid_search.fit(X_train_scaled, y_train)

# Najlepsza wartość K
best_k = grid_search.best_params_['n_neighbors']

# Trenowanie KNN z najlepszym K
knn_best = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_best.fit(X_train_scaled, y_train)

# Ustal próg prawdopodobieństwa
probability_threshold = 0.1  # Możesz dostosować tę wartość

# Uzyskanie prawdopodobieństw przynależności do każdej z klas
# Przykładowo sprawdza podpis testowy nr 1 i zwraca prawdopodobieństwo przynależności do każdej z klas
y_proba = knn_best.predict_proba(X_test_scaled)

# Przechowywanie decyzji dla każdego podpisu testowego
test_decisions = []

for i in range(len(X_test_scaled)):
    sample_decisions = {}
    for idx, cls in enumerate(knn_best.classes_):
        prob = y_proba[i][idx]
        if prob >= probability_threshold:
            sample_decisions[cls] = True  # Należy do klasy
        else:
            sample_decisions[cls] = False  # Nie należy do klasy
    test_decisions.append(sample_decisions)

# Obliczanie FAR i FRR
total_FAR = 0
total_FRR = 0
total_impostor_attempts = 0
total_genuine_attempts = 0

# Sprawdzam decyzje dla każdego podpisu testowego porównywanego z prawdziwą klasą i 55 złymi klasami
for i, decisions in enumerate(test_decisions):
    actual_class = y_test[i]
    for cls, decision in decisions.items():
        if cls == actual_class:
            total_genuine_attempts += 1
            if not decision:
                total_FRR += 1  # Fałszywe odrzucenie
        else:
            total_impostor_attempts += 1
            if decision:
                total_FAR += 1  # Fałszywa akceptacja

FAR = total_FAR / total_impostor_attempts if total_impostor_attempts > 0 else 0
FRR = total_FRR / total_genuine_attempts if total_genuine_attempts > 0 else 0

TP = total_genuine_attempts - total_FRR
TN = total_impostor_attempts - total_FAR
FP = total_FAR
FN = total_FRR

# Precision = TP / (TP + FP)
# Jak wiele z przewidzianych pozytywnie przypadków jest rzeczywiście pozytywnych
# How many of the positively predicted cases are actually positive
if (TP + FP) > 0:
    precision = TP / (TP + FP)

# Recall = TP / (TP + FN)
# Jak wiele z rzeczywistych pozytywnych przypadków zostało prawidłowo wykrytych przez model
# How many of the actual positive cases were correctly detected by the model
if (TP + FN) > 0:
    recall = TP / (TP + FN)

# F1-score = 2 x (Precision x Recall) / (Precision + Recall)
# Harmoniczna średnia Precision i Recall
# Harmonic Mean Precision and Recall
if (precision + recall) > 0:
    f1_score = (2 * (precision * recall))/(precision + recall)

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Jak wiele przypadków zostało sklasyfikowanych poprawnie
# How many cases were classified correctly
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"The best K value: {best_k}")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Total FAR: {(FAR*100):.2f}%, Total FRR: {(FRR*100):.2f}%")
print(f"Precision = {(precision*100):.2f}%, Recall = {(recall*100):.2f}%, F1-score = {(f1_score*100):.2f}%")
print(f"Accuracy: {(accuracy*100):.2f}%")
print("")
