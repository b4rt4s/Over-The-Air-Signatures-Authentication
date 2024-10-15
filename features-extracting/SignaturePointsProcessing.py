import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from itertools import chain

matplotlib.use('TkAgg') 

directory = int(input("Enter directory number to read: "))
file = int(input("Enter file number to read: "))

filename = f'../signatures-database/subject{directory}/sign_{file}.txt'

print(filename)
xy_list = []

with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        match = re.search(r"id: (-?\d+), x: (-?\d+), y: (-?\d+), time: (-?\d+)", line)
        if match:
            id_val = int(match.group(1))
            x_val = int(match.group(2))
            y_val = -int(match.group(3))
            xy_list.append((x_val, y_val))
        else:
            print("Nie znaleziono danych w linii:", line)

# Usuwanie punktów, które są zbyt blisko siebie
def remove_close_points(points, min_distance=5):
    cleaned_points = []
    points = np.array(points)
    for point in points:
        if cleaned_points:  # Sprawdza, czy lista jest pusta
            if all(np.linalg.norm(point - np.array(cleaned_points), axis=1) >= min_distance):
                cleaned_points.append(point.tolist())
        else:
            cleaned_points.append(point.tolist())  # Dodaje pierwszy punkt bez sprawdzania
    return cleaned_points

def interpolate_points(points, min_distance=10, max_distance=100):
    """
    Interpoluje punkty, gdy odległość między kolejnymi punktami przekracza max_distance.
    """
    interpolated_points = []
    
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        distance = np.linalg.norm(np.array(start_point) - np.array(end_point))
        interpolated_points.append(start_point)
        if distance > min_distance and distance < max_distance:
            # Obliczanie liczby punktów do interpolacji
            num_points = int(distance // min_distance)
            for j in range(1, num_points + 1):
                # Interpolacja liniowa
                interp_point = [start_point[0] + j * (end_point[0] - start_point[0]) / (num_points + 1),
                                start_point[1] + j * (end_point[1] - start_point[1]) / (num_points + 1)]
                interpolated_points.append(interp_point)
    #interpolated_points.append(points[-1])  # Dodaj ostatni punkt
    return interpolated_points


# def interpolate_points(points, max_distance=20):
#     """
#     Interpoluje punkty liniowo, gdy odległość między kolejnymi punktami przekracza max_distance.
#     """
#     if not points:
#         return []

#     # Rozpakowanie punktów do osobnych list x i y
#     x, y = zip(*points)
#     x = np.array(x)
#     y = np.array(y)

#     # Tworzenie funkcji interpolującej
#     f = interp1d(x, y, kind='linear', fill_value="extrapolate")

#     # Przetwarzanie punktów i interpolacja gdzie potrzeba
#     interpolated_points = [points[0]]
#     for i in range(1, len(points)):
#         start_point = points[i - 1]
#         end_point = points[i]
#         distance = np.linalg.norm(np.array(start_point) - np.array(end_point))

#         if distance > max_distance:
#             # Obliczanie liczby nowych punktów
#             num_points = int(distance // max_distance)
#             # Nowe x do interpolacji
#             new_x = np.linspace(start_point[0], end_point[0], num=num_points + 2)[1:-1]
#             # Obliczanie nowych y przez funkcję interpolującą
#             new_y = f(new_x)
#             # Dodawanie nowych punktów
#             new_points = list(zip(new_x, new_y))
#             interpolated_points.extend(new_points)

#         interpolated_points.append(end_point)

#     return interpolated_points

# Dodajemy interpolację do listy punktów

list_new = []
list_newNew = []
temp = 0
for i in xy_list:
    if not i[0] == -1:
        list_newNew.append(i)
        temp = 0 
    elif i[0] == -1 and temp == 0:
        list_new.append(list_newNew)
        temp = 1
        list_newNew = []

if list_new == []:
    list_new.append(list_newNew)


xy_list2 = []
xy_list3 = []

for i in list_new:
    xy_list2.append(remove_close_points(i))

flattened_list = list(chain.from_iterable(xy_list2))

for i in xy_list2:
    xy_list3.append(interpolate_points(i))

flattened_list3 = list(chain.from_iterable(xy_list3))

# Tworzenie figury i osi
fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 wykresy pionowo, 1 kolumna, rozmiar figury 10x15 cali

filtered_list2 = [(x, y) for x, y in xy_list if x != -1 or y != 1]

# Wykres 1: Oryginalne punkty
x_vals, y_vals = zip(*filtered_list2)  # Rozpakowywanie listy na x i y
axs[0].scatter(x_vals, y_vals, marker='o', color='blue', s=1)
axs[0].set_title('Original Points')
axs[0].set_xlabel('x - axis')
axs[0].set_ylabel('y - axis')

# Wykres 2: Punkty po usunięciu bliskich sobie
# folder subject6 sign_1 cos nie dziala
x_vals2, y_vals2 = zip(*flattened_list)  # Rozpakowywanie listy na x i y
axs[1].scatter(x_vals2, y_vals2, marker='o', color='red', s=1)
axs[1].set_title('Points After Removing Close Ones')
axs[1].set_xlabel('x - axis')
axs[1].set_ylabel('y - axis')

# Wykres 3: Punkty po interpolacji
# folder subject6 sign_1 cos nie dziala
x_vals3, y_vals3 = zip(*flattened_list3)  # Rozpakowywanie listy na x i y
axs[2].scatter(x_vals3, y_vals3, marker='o', color='green', s=1)
axs[2].set_title('Points After Interpolation')
axs[2].set_xlabel('x - axis')
axs[2].set_ylabel('y - axis')

# Ustawienie odpowiedniego odstępu między wykresami
plt.tight_layout()

# Wyświetlenie wykresów
plt.show()