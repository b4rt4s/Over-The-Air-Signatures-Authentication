import re
import os
import numpy as np
import random
from scipy.stats import skew

# Średnia szybkość
def average_speed(points, times): 
    total_distance = 0
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Obliczanie dystansu między punktami
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    # Obliczanie całkowitego czasu
    total_time = (times[-1] - times[0]) / 1000000
    
    # Unikamy dzielenia przez zero
    if total_time == 0:
        return 0
    
    # Obliczanie średniej szybkości w jednostkach na sekundę
    average_speed = total_distance / total_time
    
    return average_speed

# Średnia dodatnia szybkość w osi X
def average_positive_speed_x(points, times):
    total_distance_x = 0
    positive_time = 0

    for i in range(1, len(points)):
        x1 = points[i - 1][0]
        x2 = points[i][0]
        if x2 > x1:  # Sprawdzamy tylko ruch w prawo (dodatni)
            distance_x = x2 - x1
            delta_t = (times[i] - times[i - 1]) / 1000000
            if delta_t > 0:
                total_distance_x += distance_x
                positive_time += delta_t

    return total_distance_x / positive_time if positive_time > 0 else 0

# Średnia dodatnia szybkość w osi Y
def average_positive_speed_y(points, times):
    total_distance_y = 0
    positive_time = 0

    for i in range(1, len(points)):
        y1 = points[i - 1][1]
        y2 = points[i][1]
        if y2 > y1:  # Sprawdzamy tylko ruch w górę (dodatni)
            distance_y = y2 - y1
            delta_t = (times[i] - times[i - 1]) / 1000000
            if delta_t > 0:
                total_distance_y += distance_y
                positive_time += delta_t

    return total_distance_y / positive_time if positive_time > 0 else 0

# Całkowita długość ścieżki (dystans)
def total_path_length(points):
    total_distance = 0
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    return total_distance

# Średnie przyspieszenie
def average_acceleration(points, times):
    if len(points) < 3 or len(times) < 3:
        return 0  # Potrzebujemy przynajmniej trzy punkty do obliczenia przyspieszenia
    
    velocities = []
    velocities_times = []
    # Obliczanie prędkości między kolejnymi punktami
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        t1 = times[i - 1] / 1000000  # Konwersja na sekundy
        t2 = times[i] / 1000000
        delta_t = t2 - t1
        if delta_t == 0:
            continue  # Unikamy dzielenia przez zero
        
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / delta_t
        velocities.append(velocity)
        velocities_times.append((t1 + t2) / 2)  # Czas środkowy dla prędkości
    
    if len(velocities) < 2:
        return 0  # Nie można obliczyć przyspieszenia z mniej niż dwóch prędkości
    
    total_acceleration = 0
    count = 0
    # Obliczanie przyspieszenia między kolejnymi prędkościami
    for i in range(1, len(velocities)):
        v1 = velocities[i - 1]
        v2 = velocities[i]
        t1 = velocities_times[i - 1]
        t2 = velocities_times[i]
        delta_t = t2 - t1
        if delta_t == 0:
            continue  # Unikamy dzielenia przez zero
        acceleration = (v2 - v1) / delta_t
        total_acceleration += abs(acceleration)
        count += 1
    
    if count == 0:
        return 0  # Unikamy dzielenia przez zero
    
    average_acceleration = total_acceleration / count
    return average_acceleration

# Nachylenie linii prostej
def fragment_slope(points): 
    if len(points) < 2:
        return 0
    x_coords = [x for x, y in points]
    y_coords = [y for x, y in points]
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    return m  # Nachylenie linii prostej

# Przesunięcie w osi X
def displacement_x(points):
    if len(points) < 2:
        return 0
    x_start = points[0][0]
    x_end = points[-1][0]
    return x_end - x_start

# Przesunięcie w osi Y
def displacement_y(points):
    if len(points) < 2:
        return 0
    y_start = points[0][1]
    y_end = points[-1][1]
    return y_end - y_start

# Całkowity czas podpisu w danym fragmencie
def total_signing_time(times):
    if not times:
        return 0
    total_time = (times[-1] - times[0]) / 1000000
    return total_time

# Kąt między startowym a końcowym punktem
def angle_between_start_end(points):
    if len(points) < 2:
        return 0
    x_start, y_start = points[0]
    x_end, y_end = points[-1]
    angle = np.arctan2(y_end - y_start, x_end - x_start)
    return angle

# # Skośność współrzędnych X
# def skewness_x(points):
#     x_coords = [x for x, y in points]
#     return skew(x_coords) if len(x_coords) > 2 else 0

# # Skośność współrzędnych Y
# def skewness_y(points):
#     y_coords = [y for x, y in points]
#     return skew(y_coords) if len(y_coords) > 2 else 0

# # Przemieszczenie
# def fragment_displacement(points):
#     if len(points) < 2:
#         return 0
#     x_start, y_start = points[0]
#     x_end, y_end = points[-1]
#     displacement = np.hypot(x_end - x_start, y_end - y_start)
#     return displacement

# def max_min_coordinates(points):
#     x_coords = [x for x, y in points]
#     y_coords = [y for x, y in points]
#     max_x, min_x = max(x_coords), min(x_coords)
#     max_y, min_y = max(y_coords), min(y_coords)
#     return max_x, min_x, max_y, min_y

# def average_speed_x(points, times):
#     total_distance_x = 0
#     for i in range(1, len(points)):
#         x1 = points[i - 1][0]
#         x2 = points[i][0]
#         distance_x = abs(x2 - x1)
#         total_distance_x += distance_x
    
#     total_time = (times[-1] - times[0]) / 1000000  # Konwersja mikrosekund na sekundy
#     if total_time == 0:
#         return 0
    
#     average_speed_x = total_distance_x / total_time
#     return average_speed_x

# def average_speed_y(points, times):
#     total_distance_y = 0
#     for i in range(1, len(points)):
#         y1 = points[i - 1][1]
#         y2 = points[i][1]
#         distance_y = abs(y2 - y1)
#         total_distance_y += distance_y
    
#     total_time = (times[-1] - times[0]) / 1000000  # Konwersja mikrosekund na sekundy
#     if total_time == 0:
#         return 0
    
#     average_speed_y = total_distance_y / total_time
#     return average_speed_y

# def average_distance_from_origin(points):
#     if not points:
#         return 0
#     total_distance = 0
#     for x, y in points:
#         distance = np.hypot(x, y)
#         total_distance += distance
#     return total_distance / len(points)

# def path_efficiency(points):
#     if len(points) < 2:
#         return 1
#     start = points[0]
#     end = points[-1]
#     direct_distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
#     total_path = total_path_length(points)
#     return direct_distance / total_path if total_path != 0 else 0

# def maximum_curvature(points):
#     max_curvature = 0
#     for i in range(1, len(points) - 1):
#         x0, y0 = points[i - 1]
#         x1, y1 = points[i]
#         x2, y2 = points[i + 1]
#         v0 = np.array([x1 - x0, y1 - y0])
#         v1 = np.array([x2 - x1, y2 - y1])
#         norm_v0 = np.linalg.norm(v0)
#         norm_v1 = np.linalg.norm(v1)
#         if norm_v0 == 0 or norm_v1 == 0:
#             continue
#         cos_theta = np.dot(v0, v1) / (norm_v0 * norm_v1)
#         cos_theta = np.clip(cos_theta, -1, 1)
#         angle = np.arccos(cos_theta)
#         if angle > max_curvature:
#             max_curvature = angle
#     return max_curvature

# def average_velocity_x(points, times):
#     total_velocity_x = 0
#     count = 0
#     for i in range(1, len(points)):
#         x1 = points[i - 1][0]
#         x2 = points[i][0]
#         t1 = times[i - 1] / 1000000
#         t2 = times[i] / 1000000
#         delta_t = t2 - t1
#         if delta_t == 0:
#             continue
#         velocity_x = (x2 - x1) / delta_t
#         total_velocity_x += velocity_x
#         count += 1
#     return total_velocity_x / count if count > 0 else 0

# def average_velocity_y(points, times):
#     total_velocity_y = 0
#     count = 0
#     for i in range(1, len(points)):
#         y1 = points[i - 1][1]
#         y2 = points[i][1]
#         t1 = times[i - 1] / 1000000  # Konwersja mikrosekund na sekundy
#         t2 = times[i] / 1000000
#         delta_t = t2 - t1
#         if delta_t == 0:
#             continue
#         velocity_y = (y2 - y1) / delta_t
#         total_velocity_y += velocity_y
#         count += 1
#     return total_velocity_y / count if count > 0 else 0

# def average_cos_alpha(points):
#     total_cos_alpha = 0
#     count = 0
#     for i in range(1, len(points)):
#         dx = points[i][0] - points[i - 1][0]
#         dy = points[i][1] - points[i - 1][1]
#         distance = np.hypot(dx, dy)
#         if distance == 0:
#             continue
#         cos_alpha = dx / distance
#         total_cos_alpha += cos_alpha
#         count += 1
#     return total_cos_alpha / count if count > 0 else 0

# def average_sin_alpha(points):
#     total_sin_alpha = 0
#     count = 0
#     for i in range(1, len(points)):
#         dx = points[i][0] - points[i - 1][0]
#         dy = points[i][1] - points[i - 1][1]
#         distance = np.hypot(dx, dy) # Obliczanie odległości euklidesowej między punktami
#         if distance == 0:
#             continue
#         sin_alpha = dy / distance
#         total_sin_alpha += sin_alpha
#         count += 1
#     return total_sin_alpha / count if count > 0 else 0

def split_points_and_times(xy_list, times_list, num_parts=20): # Manipulacja liczbą podziałów na czasy w danym podpisie
    if len(xy_list) != len(times_list):
        raise ValueError("xy_list and times_list must have the same length")

    N = len(xy_list) # np. 505
    sublists = []
    base_size = N // num_parts # np. 10
    remainder = N % num_parts # fragmentów do rozdsysponowania np. 5
    
    start = 0
    for i in range(num_parts):
        # Calculate the end index for the current sublist
        end = start + base_size
        if i < remainder:
            end += 1  # Add one more element to the first 'remainder' sublists
        
        # Append the current sublist of points and times
        sublists.append((xy_list[start:end], times_list[start:end]))
        
        # Update the start index for the next sublist
        start = end
    
    return sublists

def process_directory(directory):
    parent_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "signatures-database",
    )
    subject_dir = os.path.join(parent_dir, f"subject{directory}", "normalized-signs")
    extracted_features_dir = os.path.join(parent_dir, f"subject{directory}", "extracted-features")
    os.makedirs(extracted_features_dir, exist_ok=True)

    # Pobierz wszystkie nazwy plików w katalogu
    all_filenames = []
    for f in os.listdir(subject_dir):
        if os.path.isfile(os.path.join(subject_dir, f)):
            all_filenames.append(f)

    # Sortowanie nazw plików według numerów w ich nazwach
    sorted_filenames = sorted(all_filenames, key=lambda x: int(re.search(r'\d+', x).group()))

    # Losowo wybieramy 10 plików do policzenia profilu użytkownika
    selected_filenames = random.sample(all_filenames, min(10, len(all_filenames)))
    # selected_filenames = ["normalized-sign_15.txt", "normalized-sign_3.txt", "normalized-sign_11.txt"]
    selected_filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    '''
    Tu definiujemy główne listy dla naszych danych cech.
    Każda z list zawiera podlisty danej cechy dla każdego podziału czasowego t.
    Struktura: [Podpis1:[cecha1_t1, cecha1_t2, cecha1_t3, ...], Podpis2:[cecha1_t1, cecha1_t2, cecha1_t3, ...], ...]
    '''
    average_speed_list_for_selected_signs = []
    average_positive_speed_x_list_for_selected_signs = []
    average_positive_speed_y_list_for_selected_signs = []
    total_path_length_list_for_selected_signs = []
    average_acceleration_list_for_selected_signs = []
    fragment_slope_list_for_selected_signs = []
    displacement_x_list_for_selected_signs = []
    displacement_y_list_for_selected_signs = []
    total_signing_time_list_for_selected_signs = []
    angle_between_start_end_list_for_selected_signs = []

    for filename in sorted_filenames:
        xy_list = []
        times_list = []

        file_path = os.path.join(subject_dir, filename)

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line != "BREAK":
                    match = re.search(r"x: (-?\d+), y: (-?\d+), t: (\d+)", line)
                    if match:
                        x_coord = int(match.group(1))
                        y_coord = int(match.group(2))
                        time = int(match.group(3))
                        xy_list.append((x_coord, y_coord))
                        times_list.append(time)
                else:
                    xy_list.append("BREAK")
                    times_list.append("BREAK")

        # Usuń przerwy przed podziałem
        xy_list = [point for point in xy_list if point != "BREAK"]
        times_list = [time for time in times_list if time != "BREAK"]

        # Podział punktów i czasów
        sublists = split_points_and_times(xy_list, times_list)

        '''
        Tu definiujemy listy dla naszych cech.
        Każda z list zawiera metryczki wyliczone na bazie punktów z danego przedziału t.
        struktura: [cecha1_t1, cecha1_t2, cecha1_t3, ...]
        '''
        average_speed_list_for_sign = []
        average_positive_speed_x_list_for_sign = []
        average_positive_speed_y_list_for_sign = []
        total_path_length_list_for_sign = []
        average_acceleration_list_for_sign = []
        fragment_slope_list_for_sign = []
        displacement_x_list_for_sign = []
        displacement_y_list_for_sign = []
        total_signing_time_list_for_sign = []
        angle_between_start_end_list_for_sign = []

        for i, (points, times) in enumerate(sublists):
            average_speed_val = average_speed(points, times)
            average_speed_list_for_sign.append(average_speed_val)

            average_positive_speed_x_val = average_positive_speed_x(points, times)
            average_positive_speed_x_list_for_sign.append(average_positive_speed_x_val)

            average_positive_speed_y_val = average_positive_speed_y(points, times)
            average_positive_speed_y_list_for_sign.append(average_positive_speed_y_val)

            total_path_length_val = total_path_length(points)
            total_path_length_list_for_sign.append(total_path_length_val)

            average_acceleration_val = average_acceleration(points, times)
            average_acceleration_list_for_sign.append(average_acceleration_val)

            fragment_slope_val = fragment_slope(points)
            fragment_slope_list_for_sign.append(fragment_slope_val)

            displacement_x_val = displacement_x(points)
            displacement_x_list_for_sign.append(displacement_x_val)

            displacement_y_val = displacement_y(points)
            displacement_y_list_for_sign.append(displacement_y_val)

            total_signing_time_val = total_signing_time(times)
            total_signing_time_list_for_sign.append(total_signing_time_val)

            angle_between_start_end_val = angle_between_start_end(points)
            angle_between_start_end_list_for_sign.append(angle_between_start_end_val)

            # print(f"Sublist {i} for {filename}:")
            # print(f"Average speed: {average_speed_val}")
            # print(f"Points: {points}")
            # print(f"Times: {times}")

        # Zapisz cechy do pliku
        '''
        Struktura pliku z przykładem V (szybkości średniej):
        T/C (Czas na cechy)
            C1 C2 C3 C4 ... CN
        T1 V1 
        T2 V2
        T3 V3
        .  .
        .  .
        .  .
        TN

        Tu dopisujemy cechy do plików.
        '''
        file_number = re.search(r'\d+', filename).group()
        extracted_features_filename = os.path.join(extracted_features_dir, f"extracted-features-{file_number}.txt")
        with open(extracted_features_filename, "w") as feature_file:
            for i in range(len(average_speed_list_for_sign)):
                feature_file.write(f"{average_speed_list_for_sign[i]}, {average_positive_speed_x_list_for_sign[i]}, {average_positive_speed_y_list_for_sign[i]}, {total_path_length_list_for_sign[i]}, {average_acceleration_list_for_sign[i]}, {fragment_slope_list_for_sign[i]}, {displacement_x_list_for_sign[i]}, {displacement_y_list_for_sign[i]}, {total_signing_time_list_for_sign[i]}, {angle_between_start_end_list_for_sign[i]}\n") # Tu dopisujemy cechy

        if filename in selected_filenames:
            average_speed_list_for_selected_signs.append(average_speed_list_for_sign)
            average_positive_speed_x_list_for_selected_signs.append(average_positive_speed_x_list_for_sign)
            average_positive_speed_y_list_for_selected_signs.append(average_positive_speed_y_list_for_sign)
            total_path_length_list_for_selected_signs.append(total_path_length_list_for_sign)
            average_acceleration_list_for_selected_signs.append(average_acceleration_list_for_sign)
            fragment_slope_list_for_selected_signs.append(fragment_slope_list_for_sign)
            displacement_x_list_for_selected_signs.append(displacement_x_list_for_sign)
            displacement_y_list_for_selected_signs.append(displacement_y_list_for_sign)
            total_signing_time_list_for_selected_signs.append(total_signing_time_list_for_sign)
            angle_between_start_end_list_for_selected_signs.append(angle_between_start_end_list_for_sign)

    # Transpose the list of average speeds and total path lengths to get lists for each time point
    transposed_average_speed_list = list(map(list, zip(*average_speed_list_for_selected_signs)))
    transposed_average_positive_speed_x_list = list(map(list, zip(*average_positive_speed_x_list_for_selected_signs)))
    transposed_average_positive_speed_y_list = list(map(list, zip(*average_positive_speed_y_list_for_selected_signs)))
    transposed_total_path_length_list = list(map(list, zip(*total_path_length_list_for_selected_signs)))
    transposed_average_acceleration_list = list(map(list, zip(*average_acceleration_list_for_selected_signs)))
    transposed_fragment_slope_list = list(map(list, zip(*fragment_slope_list_for_selected_signs)))
    transposed_displacement_x_list = list(map(list, zip(*displacement_x_list_for_selected_signs)))
    transposed_displacement_y_list = list(map(list, zip(*displacement_y_list_for_selected_signs)))
    transposed_total_signing_time = list(map(list, zip(*total_signing_time_list_for_selected_signs)))
    transposed_angle_between_start_end = list(map(list, zip(*angle_between_start_end_list_for_selected_signs)))

    # for i in transposed_average_speed_list:
    #    print(i)

    # Prepare the data to be saved
    '''
    Tabelka z danymi profilu użytkownika
    t0: cecha1(mean, std), cecha2(mean, std), cecha3(mean, std), ..., cecha10(mean, std)
    t1: cecha1(mean, std), cecha2(mean, std), cecha3(mean, std), ..., cecha10(mean, std)
    ...
    t100: cecha1(mean, std), cecha2(mean, std), cecha3(mean, std), ..., cecha10(mean, std)
    '''
    profile_data = []
    for i, (speeds, pos_speeds_x, pos_speeds_y, lengths, accelerations, slopes, displacements_x, displacements_y, signing_times, angles) in enumerate(zip(
            transposed_average_speed_list, transposed_average_positive_speed_x_list, transposed_average_positive_speed_y_list, 
            transposed_total_path_length_list, transposed_average_acceleration_list, transposed_fragment_slope_list,
            transposed_displacement_x_list, transposed_displacement_y_list, transposed_total_signing_time, transposed_angle_between_start_end)):
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        mean_pos_speed_x = np.mean(pos_speeds_x)
        std_pos_speed_x = np.std(pos_speeds_x)
        mean_pos_speed_y = np.mean(pos_speeds_y)
        std_pos_speed_y = np.std(pos_speeds_y)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        mean_acceleration = np.mean(accelerations)
        std_acceleration = np.std(accelerations)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        mean_displacement_x = np.mean(displacements_x)
        std_displacement_x = np.std(displacements_x)
        mean_displacement_y = np.mean(displacements_y)
        std_displacement_y = np.std(displacements_y)
        mean_signing_time = np.mean(signing_times)
        std_signing_time = np.std(signing_times)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        profile_data.append(f"{mean_speed}, {std_speed}, {mean_pos_speed_x}, {std_pos_speed_x}, {mean_pos_speed_y}, {std_pos_speed_y}, "
                            f"{mean_length}, {std_length}, {mean_acceleration}, {std_acceleration}, {mean_slope}, {std_slope}, "
                            f"{mean_displacement_x}, {std_displacement_x}, {mean_displacement_y}, {std_displacement_y}, "
                            f"{mean_signing_time}, {std_signing_time}, {mean_angle}, {std_angle}")

    # Save the data to a file within the profiles directory
    profiles_dir = os.path.join(parent_dir, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    profile_filename = os.path.join(profiles_dir, f"profile-{directory}.txt")
    with open(profile_filename, "w") as profile_file:
        for line in profile_data:
            profile_file.write(line + "\n")

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

choice = input("Enter 'range' to specify a range of subjects or 'all' to process all subjects: ").strip().lower()

if choice == 'range':
    start = int(input("Enter start subject number: "))
    end = int(input("Enter end subject number: "))
    for directory in range(start, end + 1):
        process_directory(directory)
elif choice == 'all':
    for directory in os.listdir(parent_dir):
        if directory.startswith("subject") and directory[7:].isdigit():
            process_directory(int(directory[7:]))
else:
    print("Invalid choice. Please enter 'range' or 'all'.")
