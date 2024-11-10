import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
import numpy as np

# Przykładowe dane
X = np.random.rand(100, 2) * 100  # Dwie cechy o różnych skalach

# Wykres oryginalnych danych
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Oryginalne dane')

# Standaryzacja danych
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Wykres danych po skalowaniu
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.title('Dane po standaryzacji')

plt.show()
