import re
import os
import numpy as np
import random
from scipy.stats import skew

'''
Implementacja cechy 1: średnia szybkość
Implemenation of feature 1: average speed
'''
def average_speed(points, times): 
    total_distance = 0

    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Obliczanie dystansu między punktami
        # Compute the distance between the points
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    # Obliczanie całkowitego czasu
    # Compute the total time
    total_time = (times[-1] - times[0]) / 1000000
    
    if total_time == 0:
        return 0
    
    # Obliczanie średniej szybkości w jednostkach na sekundę
    # Compute the average speed in units per second
    average_speed = total_distance / total_time
    
    return average_speed

'''
Implementacja cechy 2: średnia dodatnia szybkość w osi X
Implemenation of feature 2: average positive speed in X axis
'''
def average_positive_speed_x(points, times):
    total_distance_x = 0
    positive_time = 0

    for i in range(1, len(points)):
        x1 = points[i - 1][0]
        x2 = points[i][0]
        
        # Sprawdzenie tylko ruchu w prawią stronę (dodatniego)
        # Check only movement in the right direction (positive)
        if x2 > x1:
            distance_x = x2 - x1
            delta_t = (times[i] - times[i - 1]) / 1000000
            if delta_t > 0:
                total_distance_x += distance_x
                positive_time += delta_t

    return total_distance_x / positive_time if positive_time > 0 else 0

'''
Implementacja cechy 3: średnia dodatnia szybkość w osi Y
Implemenation of feature 3: average positive speed in Y axis
'''
def average_positive_speed_y(points, times):
    total_distance_y = 0
    positive_time = 0

    for i in range(1, len(points)):
        y1 = points[i - 1][1]
        y2 = points[i][1]

        # Sprawdzenie tylko ruchu w górę (dodatniego)
        # Check only movement upwards (positive)
        if y2 > y1:
            distance_y = y2 - y1
            delta_t = (times[i] - times[i - 1]) / 1000000
            if delta_t > 0:
                total_distance_y += distance_y
                positive_time += delta_t

    return total_distance_y / positive_time if positive_time > 0 else 0

'''
Implementacja cechy 4: całkowita długość ścieżki (łączny dystans między punktami)
Implemenation of feature 4: total path length (total distance between points)
'''
def total_path_length(points):
    total_distance = 0

    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
        
    return total_distance

'''
Implementacja cechy 5: średnie przyspieszenie
Implemenation of feature 5: average acceleration
'''
def average_acceleration(points, times):
    # Zapewnienie warunku minimum 3 punktów do obliczenia szybkości między nimi
    # Ensure a minimum of 3 points to calculate the speed between them
    if len(points) < 3 or len(times) < 3:
        return 0
    
    # Obliczanie prędkości między kolejnymi punktami
    # Compute the speed between consecutive points
    speeds = []
    speeds_times = []

    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        t1 = times[i - 1] / 1000000
        t2 = times[i] / 1000000
        delta_t = t2 - t1

        if delta_t == 0:
            continue
        
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = distance / delta_t
        speeds.append(speed)

        # Obliczanie środkowego czasu dla prędkości
        # Compute the middle time for the speed
        speeds_times.append((t1 + t2) / 2)
    
    # Zapewnienie warunku minimum 2 punktów do obliczenia przyspieszenia między nimi
    # Ensure a minimum of 2 points to calculate the acceleration between them
    if len(speeds) < 2:
        return 0
    
    # Obliczanie przyspieszenia między kolejnymi punktami
    # Compute the acceleration between consecutive points
    total_acceleration = 0
    count = 0
   
    for i in range(1, len(speeds)):
        v1 = speeds[i - 1]
        v2 = speeds[i]

        t1 = speeds_times[i - 1]
        t2 = speeds_times[i]
        delta_t = t2 - t1

        if delta_t == 0:
            continue

        acceleration = (v2 - v1) / delta_t
        total_acceleration += abs(acceleration)
        count += 1
    
    if count == 0:
        return 0
    
    # Obliczenie średniego przyspieszenia
    # Compute the average acceleration
    average_acceleration = total_acceleration / count

    return average_acceleration

'''
Implementacja cechy 6: nachylenie linii prostej
Implemenation of feature 6: fragment slope
'''
def fragment_slope(points): 
    if len(points) < 2:
        return 0
    
    # Oddzielenie współrzędnych X i Y
    # Separate the X and Y coordinates
    x_coords = [x for x, y in points]
    y_coords = [y for x, y in points]
    
    # Utworzenie transponowanej macierzy A składającej się z współrzędnych X i jedynek
    # Przykład dla 3 punktów: A = [[x1, 1], [x2, 1], [x3, 1]]
    # Create the transposed matrix A consisting of X coordinates and ones
    # Example for 3 points: A = [[x1, 1], [x2, 1], [x3, 1]]
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T

    # Dopasowanie linii prostej do punktów metodą najmniejszych kwadratów, aby uzyskać nachylenie (m) i wyraz wolny (c)
    # Fit a line to the points using the least squares method to get the slope (m) and intercept (c)
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    
    return m

'''
Implementacja cechy 7: przesunięcie w osi X
Implemenation of feature 7: displacement in X axis
'''
def displacement_x(points):
    if len(points) < 2:
        return 0
    x_start = points[0][0]
    x_end = points[-1][0]
    return x_end - x_start

'''
Implementacja cechy 8: przesunięcie w osi Y
Implemenation of feature 8: displacement in Y axis
'''
def displacement_y(points):
    if len(points) < 2:
        return 0
    y_start = points[0][1]
    y_end = points[-1][1]
    return y_end - y_start

'''
Implementacja cechy 9: czas podpisu
Implemenation of feature 9: signing time
'''
def total_signing_time(times):
    if not times:
        return 0
    total_time = (times[-1] - times[0]) / 1000000
    return total_time

'''
Implementacja cechy 10: kąt między startowym a końcowym punktem
Implemenation of feature 10: angle between start and end point
'''
def angle_between_start_end(points):
    if len(points) < 2:
        return 0
    
    x_start, y_start = points[0]
    x_end, y_end = points[-1]

    angle = np.arctan2(y_end - y_start, x_end - x_start)

    return angle

'''
Funkcja dzieląca podpis na równe fragmenty czasowe.
Liczba fragmentów jest określana przez num_parts.
W tym przypadku, num_parts = 20.

Function splitting the signature into equal time parts.
The number of parts is determined by num_parts.
In this case, num_parts = 20.
'''
def split_points_and_times(xy_list, times_list, num_parts=20):
    if len(xy_list) != len(times_list):
        raise ValueError("xy_list and times_list must have the same length")

    # Obliczanie liczby punktów w podpisie np. 505
    # Calculate the number of points in the signature f.ex. 505
    N = len(xy_list)

    # Liczba punktów w każdym fragmencie np. 10
    # Number of points in each part f.ex. 10
    base_size = N // num_parts

    # Reszta z dzielenia na fragmenty np. 5, które będą dodane do pierwszych fragmentów czasowych
    # Excess of division into parts f.ex. 5 which will be added to the first time parts
    remainder = N % num_parts
    
    sublists = []
    start = 0
    
    for i in range(num_parts):
        # Obliczanie końca fragmentu czasowego
        # Calculate the end of the time part
        end = start + base_size

        # Dodanie jednego elementu do pierwszych fragmentów czasowych
        if i < remainder:
            end += 1
        
        # Dodanie punktów i czasów do podlisty
        # Add points and times to the sublist
        sublists.append((xy_list[start:end], times_list[start:end]))
        
        # Przesunięcie początku na koniec
        # Move the start to the end
        start = end
    
    return sublists

def process_directory(directory, selected_numbers):
    parent_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "signatures-database",
    )
    subject_dir = os.path.join(parent_dir, f"subject{directory}", "normalized-signs")
    extracted_features_dir = os.path.join(parent_dir, f"subject{directory}", "extracted-features")
    os.makedirs(extracted_features_dir, exist_ok=True)

    # Pobieranie wszystkich nazwy plików z katalogu normalized-signs
    # Get all filenames from the normalized-signs directory
    all_filenames = []
    for f in os.listdir(subject_dir):
        if os.path.isfile(os.path.join(subject_dir, f)):
            all_filenames.append(f)

    # Sortowanie rosnąco nazw plików według numerów w ich nazwach
    # Sort the filenames in ascending order by the numbers in their names
    sorted_filenames = sorted(all_filenames, key=lambda x: int(re.search(r'\d+', x).group()))

    # Wybranie 10 plików do policzenia profilu użytkownika zgodnie z wylosowanymi numerami podpisów od 1 do 15
    # Select 10 files to calculate the user profile according to the randomly selected signature numbers from 1 to 15
    selected_filenames = []
    for f in sorted_filenames:
        file_number = int(re.search(r'\d+', f).group())
        if file_number in selected_numbers:
            selected_filenames.append(f)

    '''
    Definicja list głównych dla każdej z cech zawierających metryczki wyliczone na bazie punktów z danego przedziału t w każdym wylosowanym podpisie.
    Struktura: [Podpis1: [cecha1_t1, cecha1_t2, cecha1_t3, ...], Podpis2: [cecha1_t1, cecha1_t2, cecha1_t3, ...], ...]

    Definition of main lists for each feature containing metrics calculated based on points from a given time interval t in each randomly selected signature.
    Structure: [Signature1: [feature1_t1, feature1_t2, feature1_t3, ...], Signature2: [feature1_t1, feature1_t2, feature1_t3, ...], ...]
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

        # Usunięcie przerw przed podziałem
        # Remove breaks before splitting
        xy_list = [point for point in xy_list if point != "BREAK"]
        times_list = [time for time in times_list if time != "BREAK"]

        # Podział punktów i czasów na fragmenty czasowe
        # Split points and times into time parts
        sublists = split_points_and_times(xy_list, times_list)

        '''
        Definicja podlist dla każdej z cech zawierających metryczki wyliczone na bazie punktów z danego przedziału t.
        Struktura: [cecha1_t1, cecha1_t2, cecha1_t3, ...]

        Definition of sublists for each feature containing metrics calculated based on points from a given time interval t.
        Structure: [feature1_t1, feature1_t2, feature1_t3, ...]
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

        '''
        Zapis cech do pliku z danymi wyliczonymi na podstawie punktów z danego przedziału t.
        Struktura pliku na przykładzie cechy średniej szybkości V:

        T/C (Czas na cechy)
           C1 C2 C3 C4 ... CN
        T1 V1 
        T2 V2
        T3 V3
        .  .
        .  .
        .  .
        TN VN

        Save features to a file with data calculated based on points from a given time interval t.
        File structure for the average speed V feature example:

        T/C (Time for features)
           C1 C2 C3 C4 ... CN
        T1 V1 
        T2 V2
        T3 V3
        .  .
        .  .
        .  .
        TN VN       
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
        
    # Transpozycja list cech dla wylosowanych podpisów, aby wierszami były czasy, a kolumnami cechy
    # Transposition of feature lists for randomly selected signatures so that the rows are times and the columns are features
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

    '''
    Zapis średnich i odchyleń standardowych dla poszczególnych cech dla wylosowanych 10 podpisów do pliku według podziału na fragmenty czasowe
    Struktura pliku:

    T/C (Czas na cechy)
        C1 C2 C3 C4 ... CN
    T1 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    T2 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    T3 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    .  .
    .  .
    .  .
    TN mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10

    Save the means and standard deviations for each feature for the randomly selected 10 signatures to a file according to the division into time fragments.
    File structure:

    T/C (Time for features)
        C1 C2 C3 C4 ... CN
    T1 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    T2 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    T3 mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
    .  .
    .  .
    .  .
    TN mean1, std1, mean2, std2, mean3, std3, ..., mean10, std10
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

    # Zapisanie danych o średnich i odchyleniach cech jako profilu danego użytkownika
    # Save the data about the means and standard deviations of features as the profile of the given user
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

# Wylosowanie 10 numerów od 1 do 15
# Draw 10 numbers from 1 to 15
selected_numbers = sorted(random.sample(range(1, 16), 10))
selected_numbers_file = os.path.join(parent_dir, "selected_numbers.txt")
with open(selected_numbers_file, "w") as f:
    f.write(", ".join(map(str, selected_numbers)) + "\n")

if choice == 'range':
    start = int(input("Enter start subject number: "))
    end = int(input("Enter end subject number: "))
    for directory in range(start, end + 1):
        process_directory(directory, selected_numbers)
elif choice == 'all':
    for directory in os.listdir(parent_dir):
        if directory.startswith("subject") and directory[7:].isdigit():
            process_directory(int(directory[7:]), selected_numbers)
else:
    print("Invalid choice. Please enter 'range' or 'all'.")
