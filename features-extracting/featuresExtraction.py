import re
import os
import numpy as np
import random
import pandas as pd

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

def total_path_length(points):
    total_distance = 0
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    return total_distance


def split_points_and_times(xy_list, times_list, num_parts=100): # Manipulacja liczbą podziałów na czasy w danym podpisie
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
    # selected_filenames = random.sample(all_filenames, min(10, len(all_filenames)))
    selected_filenames = ["normalized-sign_15.txt", "normalized-sign_3.txt", "normalized-sign_11.txt"]
    selected_filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    '''
    Tu definiujemy główne listy dla naszych danych cech.
    Każda z list zawiera podlisty danej cechy dla każdego podziału czasowego t.
    Struktura: [Podpis1:[cecha1_t1, cecha1_t2, cecha1_t3, ...], Podpis2:[cecha1_t1, cecha1_t2, cecha1_t3, ...], ...]
    '''
    average_speed_list_for_selected_signs = []

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

        for i, (points, times) in enumerate(sublists):
            average_speed_val = average_speed(points, times)
            average_speed_list_for_sign.append(average_speed_val)
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
                feature_file.write(f"{average_speed_list_for_sign[i]}\n") # Tu dopisujemy cechy

        if filename in selected_filenames:
            average_speed_list_for_selected_signs.append(average_speed_list_for_sign)

    #ranspose the list of average speeds to get lists of speeds for each time point
    transposed_average_speed_list = list(map(list, zip(*average_speed_list_for_selected_signs)))
    # for i in transposed_average_speed_list:
    #     print(i)

    # Prepare the data to be saved
    profile_data = []
    for i, speeds in enumerate(transposed_average_speed_list):
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        profile_data.append(f"t_{i}: Mean speed: {mean_speed}, Standard deviation: {std_speed}")

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
