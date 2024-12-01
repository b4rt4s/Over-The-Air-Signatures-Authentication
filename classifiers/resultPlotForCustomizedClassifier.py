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

print(f"\nNajniższe EER {best_eer:.2f}% w {best_file}")

# Wyświetlenie plików do wyboru
# Displaying files to choose from
print("\nDostępne pliki, dla których można wyświetlić wykres:")

for index, file in enumerate(data_files):
    print(f"{index+1}: {file}")

# Wybranie pliku do wyświetlenia
# Choosing a file to display
selected_index = int(input("\nWybierz plik, którego wykres chcesz wyświetlić: ")) - 1

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

# Rysowanie wykresu FAR i FRR w zależności od progu z zaznaczeniem punktu EER
# Drawing a graph of FAR and FRR depending on the threshold with the EER point marked
plt.figure(figsize=(10, 5))
plt.plot(data['threshold'], data['far'], label='FAR w (%)', marker='o', color='orange')
plt.plot(data['threshold'], data['frr'], label='FRR w (%)', marker='x', color='green')
plt.xlabel('Próg minimalnej liczby zgodnych cech', fontsize=12)
plt.ylabel('Wskaźnik błędu (%)', fontsize=12)

# Dodanie dodatkowych informacji w tytule wykresu lub podtytule
title = 'Współczynniki błędów FAR i FRR w zależności od progu minimalnej liczby zgodnych cech'
subtitle = (f"Numery podpisów użytych użytych do zbioru uczącego i zbudowania profili: {training_signature_numbers}\n"
            f"Próg decyzyjny #1: Numery wybranych cech: {feature_numbers}\n"
            f"Próg decyzyjny #2: Liczba podziałów czasowych dla danej sygnatury:  {num_parts}\n"
            f"Próg decyzyjny #3: Wielokrotność odchylenia standardowego: {sigma_multiplier}\n"
            f"Próg decyzyjny #4: Średnia zgodność metryk na cechę: {mean_t_of_feature_matches_threshold}\n"
            f"Próg decyzyjny #5: Minimalne liczby zgodnych cech: {genuine_features_thresholds}")

plt.title(title, fontsize=12)
plt.suptitle(subtitle, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Narysowanie punktu EER na wykresie
# Drawing the EER point on the chart
plt.plot(EER_threshold, EER, 'ro')
plt.annotate(f'EER = {EER:.2f}%', (EER_threshold, EER), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=11)  

plt.tight_layout()
plt.show()
