import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Załaduj dane z pliku, przekształcając przecinki na kropki w przypadku użycia przecinka jako separatora dziesiętnego
data = pd.read_csv('far_frr_results.txt', converters={
    'far': lambda x: float(x.replace(',', '.')),
    'frr': lambda x: float(x.replace(',', '.'))
})

# Sortowanie danych według kolumny 'Threshold' (jeśli dane nie są posortowane)
data.sort_values('threshold', inplace=True)

# Wyznaczanie punktu EER
differences = np.abs(data['far'] - data['frr'])
min_index = np.argmin(differences)
EER = (data['far'].iloc[min_index] + data['frr'].iloc[min_index]) / 2
EER_threshold = data['threshold'].iloc[min_index]

# Rysowanie wykresu
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
