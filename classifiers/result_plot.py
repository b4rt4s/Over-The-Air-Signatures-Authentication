import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytywanie katalogu z plikami wynikowymi
# Loading the directory with result files
data_dir = os.path.join("signatures-database", "results")

# Pobranie listy plików wynikowych i posortowanie jej wg nazw plików
# Getting the list of result files and sorting it by file names
data_files = []

for f in os.listdir(data_dir):
    if f.startswith("results_") and f.endswith(".txt"):
        data_files.append(f)

data_files.sort(key=lambda filename: float(filename.replace('results_', '').replace('.txt', '')))

if not data_files:
    print("No result files found in the 'results' directory.")
    exit()

# Wylistowanie wartości EER dla każdego z plików w folderze results
# Listing the EER values for each of the files in the results folder
eer_list = []

for file in data_files:
    file_path = os.path.join(data_dir, file)
    data_list = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            far = float(parts[0].strip())
            frr = float(parts[1].strip())
            genuine_features_threshold = int(parts[2].strip())
            data_list.append([far, frr, genuine_features_threshold])
    if not data_list:
        continue

    data = pd.DataFrame(data_list, columns=["far", "frr", "threshold"])
    differences = np.abs(data['far'] - data['frr'])
    min_index = np.argmin(differences)
    EER = (data['far'].iloc[min_index] + data['frr'].iloc[min_index]) / 2
    
    eer_list.append(EER)

    print(f"{file} EER = {EER:.2f}%")

# Znalezienie najlepszego EER i wylistowanie dla niego pliku
# Finding the best EER and listing the file for it
best_eer = min(eer_list)
best_file = data_files[eer_list.index(best_eer)]

print(f"\nThe best EER {best_eer:.2f}% in {best_file}")

# Wyświetlenie plików do wyboru
# Displaying files to choose from
print("\nAvailable files to plot:")

for index, file in enumerate(data_files):
    print(f"{index+1}: {file}")

# Wybranie pliku do wyświetlenia
# Choosing a file to display
selected_index = int(input("\nChoose number of file to plot: ")) - 1

if selected_index < 0 or selected_index >= len(data_files):
    print("Incorrect selection.")
    exit()

selected_file = data_files[selected_index]

# Wczytanie danych z wybranego pliku
# Loading data from the selected file
file_path = os.path.join(data_dir, selected_file)
data_list = []
with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split(',')
        far = float(parts[0].strip())
        frr = float(parts[1].strip())
        threshold = int(parts[2].strip())
        data_list.append([far, frr, threshold])

# Konwersja danych do DataFrame
# Data conversion to DataFrame
data = pd.DataFrame(data_list, columns=["far", "frr", "threshold"])

# Wyznaczanie współczynnika EER
# Determining the EER coefficient
differences = np.abs(data['far'] - data['frr'])
min_index = np.argmin(differences)
EER = (data['far'].iloc[min_index] + data['frr'].iloc[min_index]) / 2
EER_threshold = data['threshold'].iloc[min_index]

# Pobranie wartości mean_t_of_feature_matches_threshold z nazwy pliku
# Getting the value of mean_t_of_feature_matches_threshold from the file name
mean_t_of_feature_matches_threshold = selected_file.replace('results_', '').replace('.txt', '')

# Wczytanie informacji z pliku 'selected_numbers_file_and_features.txt'
# Loading information from the 'selected_numbers_file_and_features.txt' file
selected_numbers_file = os.path.join(data_dir, "selected_numbers_file_and_features.txt")

if os.path.exists(selected_numbers_file):
    with open(selected_numbers_file, 'r') as file:
        lines = file.readlines()
        
        if len(lines) >= 6:
            training_signature_numbers = lines[0].strip()
            feature_numbers = lines[1].strip()
            num_parts = lines[2].strip()
            sigma_multiplier = lines[3].strip()
            mean_t_of_feature_matches_thresholds = lines[4].strip()
            genuine_features_thresholds = lines[5].strip()
        else:
            print("Plik 'selected_numbers_file_and_features.txt' nie zawiera wystarczającej liczby linii.")
else:
    print("Brak pliku 'selected_numbers_file_and_features.txt' w katalogu 'results'.")
    training_signature_numbers = ""
    feature_numbers = ""
    num_parts = ""
    sigma_multiplier = ""
    mean_t_of_feature_matches_thresholds = ""
    genuine_features_thresholds = ""

# Rysowanie wykresu FAR i FRR w zależności od progu z zaznaczeniem punktu EER
plt.figure(figsize=(10, 5))
plt.plot(data['threshold'], data['far'], label='FAR (%)', marker='o')
plt.plot(data['threshold'], data['frr'], label='FRR (%)', marker='x')
plt.xlabel('Genuine Features Threshold (k)')
plt.ylabel('Error Rate (%)')

# Dodanie dodatkowych informacji w tytule wykresu lub podtytule
title = 'FAR i FRR w zależności od progu cech zgodnych (k) z zaznaczeniem EER'
subtitle = (f"Użyte podpisy: {training_signature_numbers}\n"
            f"Użyte cechy: {feature_numbers}\n"
            f"Ilość podziałów czasowych: {num_parts}, "
            f"Mnożnik sigma: {sigma_multiplier}, "
            f"Mean t of feature matches threshold: {mean_t_of_feature_matches_threshold}")

plt.title(title)
plt.suptitle(subtitle, fontsize=10)

plt.legend()
plt.grid(True)

# Zaznaczenie punktu EER na wykresie
plt.plot(EER_threshold, EER, 'ro')  # Rysowanie czerwonego punktu
plt.annotate(f'EER = {EER:.2f}%', (EER_threshold, EER), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()
