import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
import pandas as pd

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
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Najlepsze max_depth
best_depth = 2

# Trenowanie drzewa decyzyjnego z najlepszym max_depth
tree_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree_best.fit(X_train_scaled, y_train)

# Uzyskanie prawdopodobieństw przynależności do każdej z klas
y_proba = tree_best.predict_proba(X_test_scaled)

# Lista progów od 0.01 do 0.05 z krokiem 0.0005
thresholds = np.arange(0.01, 0.05, 0.005)
far_list = []
frr_list = []

for threshold in thresholds:
    total_FAR = 0
    total_FRR = 0
    total_impostor_attempts = 0
    total_genuine_attempts = 0
    
    for i in range(len(X_test_scaled)):
        actual_class = y_test[i]
        probas = y_proba[i]
        
        for idx, cls in enumerate(tree_best.classes_):
            prob = probas[idx]
            if cls == actual_class:
                total_genuine_attempts += 1
                if prob < threshold:
                    total_FRR += 1  # Fałszywe odrzucenie
            else:
                total_impostor_attempts += 1
                if prob >= threshold:
                    total_FAR += 1  # Fałszywa akceptacja
    
    FAR = (total_FAR / total_impostor_attempts) * 100 if total_impostor_attempts > 0 else 0
    FRR = (total_FRR / total_genuine_attempts) * 100 if total_genuine_attempts > 0 else 0

    TP = total_genuine_attempts - total_FRR
    TN = total_impostor_attempts - total_FAR
    FP = total_FAR
    FN = total_FRR

    # Precision = TP / (TP + FP)
    # Jak wiele z przewidzianych pozytywnie przypadków jest rzeczywiście pozytywnych
    # How many of the positively predicted cases are actually positive
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # Recall = TP / (TP + FN)
    # Jak wiele z rzeczywistych pozytywnych przypadków zostało prawidłowo wykrytych przez model
    # How many of the actual positive cases were correctly detected by the model
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # F1-score = 2 x (Precision x Recall) / (Precision + Recall)
    # Harmoniczna średnia Precision i Recall
    # Harmonic Mean Precision and Recall
    if (precision + recall) > 0:
        f1_score = (2 * (precision * recall))/(precision + recall)
    else:
        f1_score = 0

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Jak wiele przypadków zostało sklasyfikowanych poprawnie
    # How many cases were classified correctly
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    print(f"The best max_depth value: {best_depth}, Threshold: {threshold:.3f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Total FAR: {FAR:.2f}%, Total FRR: {FRR:.2f}%")
    print(f"Precision = {precision*100:.2f}%, Recall = {recall*100:.2f}%, F1-score = {f1_score*100:.2f}%")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("")

    far_list.append(FAR)
    frr_list.append(FRR)

# Znalezienie EER (gdzie FAR jest najbliżone do FRR)
differences = np.abs(np.array(far_list) - np.array(frr_list))
min_index = np.argmin(differences)
eer_threshold = thresholds[min_index]
eer = (far_list[min_index] + frr_list[min_index]) / 2

print(f"\nEqual Error Rate (EER): {eer:.3f}% with threshold {eer_threshold:.3f}")

results = pd.DataFrame({
    'threshold': thresholds,
    'far': far_list,
    'frr': frr_list
})

# Rysowanie wykresu FAR i FRR w zależności od progu z zaznaczeniem punktu EER
plt.figure(figsize=(10, 5))
plt.plot(results['threshold'], results['far'], label='FAR (%)', marker='o', color='orange')
plt.plot(results['threshold'], results['frr'], label='FRR (%)', marker='x', color='green')
plt.xlabel('Probability Threshold')
plt.ylabel('Error Rate (%)')
plt.title('FAR and FRR vs Probability Threshold')
plt.legend()
plt.grid(True)

# Narysowanie punktu EER na wykresie
plt.plot(eer_threshold, eer, 'ro', label='EER Point')
plt.annotate(f'EER = {eer:.2f}%\nThreshold = {eer_threshold:.3f}',
             (eer_threshold, eer),
             textcoords="offset points",
             xytext=(0,10),
             ha='center',
             color='red')

plt.legend()
plt.tight_layout()
plt.show()