import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

# Wczytanie numerów podpisów użytych do stworzenia profili
# Loading the numbers of signatures used to create profiles
selected_numbers_file = os.path.join(parent_dir, "results", "selected_numbers_file_and_features.txt")
with open(selected_numbers_file, 'r') as file:
    selected_numbers_line = file.readline().strip()
    selected_numbers = [int(x.strip()) for x in selected_numbers_line.split(",")]

# Wczytanie wszystkich numerów profili, tj. 56
# Loading all profile numbers, i.e., 56
subjects = []
for directory in os.listdir(parent_dir):
    if directory.startswith("subject") and directory[7:].isdigit():
        subjects.append(int(directory[7:]))

subjects.sort()

# Parametry dla Drzewa Decyzyjnego
# Parameters for Decision Tree
max_depth_values = range(1, 21)  # Zakres głębokości drzewa do przetestowania

# Listy do przechowywania wyników
# Lists to store the results
far_list = []
frr_list = []

for max_depth in max_depth_values:
    total_FRR = 0.0
    total_FAR = 0.0
    total_N_genuine_attempts = 0
    total_N_impostor_attempts = 0

    # Słownik do przechowywania modeli wytrenowanych dla każdego użytkownika
    # Dictionary to store models trained for each user
    user_models = {}

    # Trenujemy modele dla wszystkich użytkowników
    # Training models for all users
    for subject_num in subjects:
        # Przygotowanie danych treningowych i testowych dla danego użytkownika
        # Preparing training and testing data for the user
        subject_training_data = []
        subject_training_labels = []
        subject_testing_data = []
        subject_testing_labels = []

        extracted_features_dir = os.path.join(parent_dir, f"subject{subject_num}", "extracted-features")
        feature_files = os.listdir(extracted_features_dir)
        feature_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))

        for filename in feature_files:
            number = int(filename.split('-')[-1].split('.')[0])
            file_path = os.path.join(extracted_features_dir, filename)

            with open(file_path, "r") as feature_file:
                signature_data = []
                for line in feature_file:
                    signature_data.extend([float(x) for x in line.strip().split(", ")])

            if number in selected_numbers:
                subject_training_data.append(signature_data)
                # Klasa pozytywna w treningowych
                # Positive class in training
                subject_training_labels.append(1)
            else:
                subject_testing_data.append(signature_data)
                # Klasa pozytywna w testowych
                # Positive class in testing
                subject_testing_labels.append(1)  

        # Dane treningowe negatywne (podpisy innych użytkowników)
        # Negative training data (signatures of other users)
        for impostor_num in subjects:
            if impostor_num == subject_num:
                continue
            impostor_extracted_features_dir = os.path.join(parent_dir, f"subject{impostor_num}", "extracted-features")
            impostor_feature_files = os.listdir(impostor_extracted_features_dir)
            impostor_feature_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))

            for filename in impostor_feature_files:
                number = int(filename.split('-')[-1].split('.')[0])
                file_path = os.path.join(impostor_extracted_features_dir, filename)

                with open(file_path, "r") as feature_file:
                    signature_data = []
                    for line in feature_file:
                        signature_data.extend([float(x) for x in line.strip().split(", ")])

                if number in selected_numbers:
                    subject_training_data.append(signature_data)
                    # Klasa negatywna w treningowych
                    # Negative class in training
                    subject_training_labels.append(0)

        # Konwersja do tablic numpy
        # Converting to numpy arrays
        X_train = np.array(subject_training_data)
        y_train = np.array(subject_training_labels)
        X_test = np.array(subject_testing_data)
        y_test = np.array(subject_testing_labels)

        # Trening Drzewa Decyzyjnego dla danego użytkownika
        # Training Decision Tree classifier for the user
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)

        # Zapisujemy model użytkownika
        # Saving the user model
        user_models[subject_num] = clf

        # Obliczanie FRR dla danego użytkownika
        # Calculating FRR for the user
        N_genuine_attempts = len(y_test)
        total_N_genuine_attempts += N_genuine_attempts

        y_pred = clf.predict(X_test)
        false_rejections = np.sum((y_test == 1) & (y_pred == 0))
        total_FRR += false_rejections

    # Obliczanie FAR
    # Calculating FAR
    for subject_num in subjects:
        # Pobieramy podpisy testowe użytkownika
        # Getting the test signatures of the user
        extracted_features_dir = os.path.join(parent_dir, f"subject{subject_num}", "extracted-features")
        feature_files = os.listdir(extracted_features_dir)
        feature_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))

        subject_testing_data = []
        for filename in feature_files:
            number = int(filename.split('-')[-1].split('.')[0])
            if number not in selected_numbers:
                file_path = os.path.join(extracted_features_dir, filename)
                with open(file_path, "r") as feature_file:
                    signature_data = []
                    for line in feature_file:
                        signature_data.extend([float(x) for x in line.strip().split(", ")])
                    subject_testing_data.append(signature_data)

        subject_testing_data = np.array(subject_testing_data)
        # np. 5 podpisów * 55 innych użytkowników
        # e.g., 5 signatures * 55 other users
        N_impostor_attempts = len(subject_testing_data) * (len(subjects) - 1)
        total_N_impostor_attempts += N_impostor_attempts

        # Dla każdego innego użytkownika (impostora)
        # For each other user (impostor)
        for impostor_num in subjects:
            if impostor_num == subject_num:
                continue
            impostor_clf = user_models[impostor_num]
            if len(subject_testing_data) > 0:
                y_pred_impostor = impostor_clf.predict(subject_testing_data)
                false_acceptances = np.sum(y_pred_impostor == 1)
                total_FAR += false_acceptances

    # Obliczanie całkowitego FAR i FRR
    # Calculating the total FAR and FRR
    FRR_rate = (total_FRR / total_N_genuine_attempts) * 100 if total_N_genuine_attempts > 0 else 0.0
    FAR_rate = (total_FAR / total_N_impostor_attempts) * 100 if total_N_impostor_attempts > 0 else 0.0

    far_list.append(FAR_rate)
    frr_list.append(FRR_rate)

    print(f"Max Depth = {max_depth}: FAR = {FAR_rate:.2f}%, FRR = {FRR_rate:.2f}%")

# Obliczenie EER
# Calculating the EER
far_array = np.array(far_list)
frr_array = np.array(frr_list)
differences = np.abs(far_array - frr_array)
min_index = np.argmin(differences)
EER = (far_array[min_index] + frr_array[min_index]) / 2
EER_max_depth = max_depth_values[min_index]

print(f"\nThe best EER = {EER:.2f}% at Max Depth = {EER_max_depth}")

# Wczytanie informacji z pliku 'selected_numbers_file_and_features.txt'
# Loading information from the 'selected_numbers_file_and_features.txt' file
data_dir = os.path.join("signatures-database", "results")
selected_numbers_file = os.path.join(data_dir, "selected_numbers_file_and_features.txt")

if os.path.exists(selected_numbers_file):
    with open(selected_numbers_file, 'r') as file:
        lines = file.readlines()

        if len(lines) >= 3:
            training_signature_numbers = lines[0].strip()
            feature_numbers = lines[1].strip()
            num_parts = lines[2].strip()

# Dodanie dodatkowych informacji w tytule wykresu lub podtytule
# Adding additional information in the title of the graph or subtitle
title = 'FAR and FRR depending on the max depth of the decision tree classifier'
subtitle = (f"Used signatures to train model by their numbers: {training_signature_numbers}\n"
            f"Threshold #1: Selected features by their numbers: {feature_numbers}\n"
            f"Threshold #2: Division of signatures into time section: {num_parts}\n"
            f"Threshold #3: Parameter 'max_depth' of the decision tree: {max_depth_values}\n")

# Rysowanie wykresu FAR i FRR w zależności od progu z zaznaczeniem punktu EER
# Drawing a graph of FAR and FRR depending on the threshold with the EER point marked
plt.figure(figsize=(10, 5))
plt.plot(max_depth_values, far_array, label='FAR (%)', marker='o', color='orange')
plt.plot(max_depth_values, frr_array, label='FRR (%)', marker='x', color='green')
plt.xlabel('Max depth of the decision tree (max_depth)')
plt.ylabel('Error Rate (%)')
plt.title(title)
plt.suptitle(subtitle, fontsize=10)
plt.legend()
plt.grid(True)

# Zaznaczenie punktu EER
# Marking the EER point
plt.plot(EER_max_depth, EER, 'ro')
plt.annotate(f'EER = {EER:.2f}%', (EER_max_depth, EER), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()
