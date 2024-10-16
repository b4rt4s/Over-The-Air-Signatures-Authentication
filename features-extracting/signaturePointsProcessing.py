import re
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import os

directory = int(input("Enter directory number to read: "))
file_num = int(input("Enter file number to read: "))

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)
directory_path = os.path.join(parent_dir, f"subject{directory}")
filename = os.path.join(directory_path, f"sign_{file_num}.txt")

if not os.path.isfile(filename):
    print(f"File {filename} does not exist.")
    exit(1)

xy_list = []
times_list = []

with open(filename, "r") as file:
    for line in file:
        line = line.strip()
        match = re.search(
            r"id: (-?\d+), x: (-?\d+), y: (-?\d+), time: (\d+:\d+:\d+)", line
        )
        if match:
            x_coord = int(match.group(2))
            y_coord = -int(match.group(3))
            xy_list.append((x_coord, y_coord))

            time_str = match.group(4)
            minutes, seconds, milliseconds = map(int, time_str.split(":"))
            time_tuple = (minutes, seconds, milliseconds)
            times_list.append(time_tuple)
        else:
            xy_list.append((-1, -1))
            times_list.append((-1, -1, -1))

# Usuwanie punktów, które są zbyt blisko siebie
def remove_close_points(points, times, min_distance=5):
    cleaned_points = []
    cleaned_times = []
    points = np.array(points)
    times = np.array(times)
    for point, time in zip(points, times):
        if cleaned_points:  # Sprawdza, czy lista jest pusta
            if all(
                np.linalg.norm(point - np.array(cleaned_points), axis=1) >= min_distance
            ):
                cleaned_points.append(point.tolist())
                cleaned_times.append(time.tolist())
        else:
            cleaned_points.append(
                point.tolist()
            )  # Dodaje pierwszy punkt bez sprawdzania
            cleaned_times.append(time.tolist())
    return cleaned_points, cleaned_times


def interpolate_points(points, times, min_distance=10, max_distance=100):
    """
    Interpoluje punkty i czasy, gdy odległość między kolejnymi punktami przekracza max_distance.
    Uwzględnia przeskoki między minutami, sekundami i milisekundami, w tym cofnięcia czasu.
    """
    interpolated_points = []
    interpolated_times = []

    def time_in_microseconds(time):
        """Konwertuje czas [minuty, sekundy, milisekundy] na mikrosekundy."""
        return time[0] * 60 * 1_000_000 + time[1] * 1_000_000 + time[2]

    def time_from_microseconds(us):
        """Konwertuje czas z mikrosekund na [minuty, sekundy, milisekundy]."""
        minutes = us // (60 * 1_000_000)
        us %= (60 * 1_000_000)
        seconds = us // 1_000_000
        milliseconds = us % 1_000_000
        
        # Naprawa przeskoku do 60
        if minutes >= 60:
            minutes = 0
        
        return [int(minutes), int(seconds), int(milliseconds)]

    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]

        # Oryginalny czas w formacie [minuty, sekundy, milisekundy]
        start_time = times[i]
        end_time = times[i + 1]

        # Konwersja czasu na mikrosekundy dla łatwiejszego interpolowania
        start_time_us = time_in_microseconds(start_time)
        end_time_us = time_in_microseconds(end_time)

        # W przypadku cofnięcia czasu, zakładamy że czas przeszedł przez pełną godzinę
        if end_time_us < start_time_us:
            end_time_us += 60 * 60 * 1_000_000  # Dodaj 1 godzinę w mikrosekundach

        # Obliczenie odległości między punktami
        distance = np.linalg.norm(np.array(start_point) - np.array(end_point))

        # Dodanie początkowego punktu i czasu
        interpolated_points.append(start_point)
        interpolated_times.append(start_time)

        if distance > min_distance and distance < max_distance:
            # Obliczanie liczby punktów do interpolacji
            num_points = int(distance // min_distance)

            for j in range(1, num_points + 1):
                # Interpolacja liniowa dla punktów
                interp_point = [
                    start_point[0] + j * (end_point[0] - start_point[0]) / (num_points + 1),
                    start_point[1] + j * (end_point[1] - start_point[1]) / (num_points + 1),
                ]

                # Interpolacja liniowa dla czasu w mikrosekundach
                interp_time_us = start_time_us + j * (end_time_us - start_time_us) / (num_points + 1)
                interp_time = time_from_microseconds(interp_time_us)

                interpolated_points.append(interp_point)
                interpolated_times.append(interp_time)

    # Dodaj ostatni punkt i czas
    interpolated_points.append(points[-1])
    interpolated_times.append(times[-1])

    return interpolated_points, interpolated_times





# Podział punktów na grupy punktów oznaczające przerwy w pisaniu
partial_xy_list = []
full_xy_list = []

for xy in xy_list:
    if xy != (-1, -1):
        partial_xy_list.append(xy)
    elif partial_xy_list != []:
        full_xy_list.append(partial_xy_list)
        partial_xy_list = []

if full_xy_list == []:
    full_xy_list.append(partial_xy_list)

# Podział czasów na grupy czasów oznaczające przerwy w pisaniu
partial_times_list = []
full_times_list = []

for time in times_list:
    if time != (-1, -1, -1):
        partial_times_list.append(time)
    elif partial_times_list != []:
        full_times_list.append(partial_times_list)
        partial_times_list = []

if full_times_list == []:
    full_times_list.append(partial_times_list)

xy_list2 = []
times_list2 = []

# Przetwarzanie punktów i czasów
for xy_points, time_points in zip(full_xy_list, full_times_list):
    cleaned_points, cleaned_times = remove_close_points(xy_points, time_points)
    xy_list2.append(cleaned_points)
    times_list2.append(cleaned_times)

for i in times_list2:
    for j in i:
        print(j)
    print("DUPA")
# Wyodrębnienie nazwy pliku z obiektu pliku
output_filename = os.path.join(directory_path, f"processed_sign_{file_num}.txt")

# Zapis do pliku txt
with open(output_filename, "w") as f:
    for points, times in zip(xy_list2, times_list2):
        for (x, y), (minutes, seconds, milliseconds) in zip(points, times):
            f.write(f"x: {x}, y: {-y}, time: {minutes}:{seconds}:{milliseconds}\n")
        f.write(f"BREAK\n")

cleaned_xy_list = list(chain.from_iterable(xy_list2))

xy_list3 = []
times_list3 = []

for xy_points, time_points in zip(xy_list2, times_list2):
    interpolated_points, interpolated_times = interpolate_points(xy_points, time_points)
    xy_list3.append(interpolated_points)
    times_list3.append(interpolated_times)

for i in times_list3:
    for j in i:
        print(j)
    print("DUPA")

flattened_list3 = list(chain.from_iterable(xy_list3))

# Tworzenie figury i osi
fig, axs = plt.subplots(
    3, 1, figsize=(10, 15)
)  # 3 wykresy pionowo, 1 kolumna, rozmiar figury 10x15 cali

filtered_xy_list = [(x, y) for x, y in xy_list if x != -1 and y != -1]

# Wykres 1: Oryginalne punkty
x_vals, y_vals = zip(*filtered_xy_list)  # Rozpakowywanie listy na x i y
axs[0].scatter(x_vals, y_vals, marker="o", color="blue", s=1)
axs[0].set_title("Original Points")
axs[0].set_xlabel("x - axis")
axs[0].set_ylabel("y - axis")

# Wykres 2: Punkty po usunięciu bliskich sobie
x_vals2, y_vals2 = zip(*cleaned_xy_list)  # Rozpakowywanie listy na x i y
axs[1].scatter(x_vals2, y_vals2, marker="o", color="red", s=1)
axs[1].set_title("Points After Removing Close Ones")
axs[1].set_xlabel("x - axis")
axs[1].set_ylabel("y - axis")

# Wykres 3: Punkty po interpolacji
x_vals3, y_vals3 = zip(*flattened_list3)  # Rozpakowywanie listy na x i y
axs[2].scatter(x_vals3, y_vals3, marker="o", color="green", s=1)
axs[2].set_title("Points After Interpolation")
axs[2].set_xlabel("x - axis")
axs[2].set_ylabel("y - axis")

# Ustawienie odpowiedniego odstępu między wykresami
plt.tight_layout()

# Wyświetlenie wykresów
plt.show()
