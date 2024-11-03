import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Wczytywanie danych z pliku
data = pd.read_csv('far_frr_results.txt', header=None, names=['FAR', 'FRR', 'Threshold'])

# Sortowanie danych według kolumny 'Threshold'
data.sort_values('Threshold', inplace=True)

# Wyznaczanie punktu EER
differences = np.abs(data['FAR'] - data['FRR'])
min_index = np.argmin(differences)
EER = (data['FAR'][min_index] + data['FRR'][min_index]) / 2
EER_threshold = data['Threshold'][min_index]

# Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(data['Threshold'], data['FAR'], label='FAR (%)', marker='o')
plt.plot(data['Threshold'], data['FRR'], label='FRR (%)', marker='x')
plt.xlabel('Threshold')
plt.ylabel('Error Rate (%)')
plt.title('FAR and FRR vs. Threshold with EER Point')
plt.legend()
plt.grid(True)

# Zaznaczenie punktu EER na wykresie
plt.plot(EER_threshold, EER, 'ro')  # Rysowanie czerwonego punktu
plt.annotate(f'EER = {EER:.2f}%', (EER_threshold, EER), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
