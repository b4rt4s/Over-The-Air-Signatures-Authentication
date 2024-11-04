import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytywanie danych z plików
# Data loading from files
data_dir = os.path.join("signatures-database", "results")
data_files = [f for f in os.listdir(data_dir) if f.startswith("far")]

data_list = []
for file in data_files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, "r") as f:
        for line in f:
            far, frr, threshold = line.strip().split(',')
            data_list.append([float(far), float(frr), threshold]) 

# Konwersja danych do DataFrame
# Data conversion to DataFrame
data = pd.DataFrame(data_list, columns=["far", "frr", "threshold"])

# Wyznaczanie współczynnika EER
# Determining the EER coefficient
differences = np.abs(data['far'] - data['frr'])
min_index = np.argmin(differences)
EER = (data['far'].iloc[min_index] + data['frr'].iloc[min_index]) / 2
EER_threshold = data['threshold'].iloc[min_index]

# Rysowanie wykresu FAR i FRR w zależności od progu z zaznaczeniem punktu EER
# Drawing a graph of FAR and FRR depending on the threshold with the EER point marked
plt.figure(figsize=(10, 5))
plt.plot(data['threshold'], data['far'], label='FAR (%)', marker='o')
plt.plot(data['threshold'], data['frr'], label='FRR (%)', marker='x')
plt.xlabel('Threshold')
plt.ylabel('Error Rate (%)')
plt.title('FAR and FRR vs. Threshold with EER Point')
plt.legend()
plt.grid(True)

# Zaznaczenie punktu EER na wykresie
plt.plot(EER_threshold, EER, 'ro')  # Rysowanie czerwonego punktu
plt.annotate(f'EER = {EER:.2f}%', (EER_threshold, EER), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
