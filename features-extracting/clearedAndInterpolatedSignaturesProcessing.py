import re
import numpy as np
import os

def process_directory(directory):
    subject_dir = os.path.join(parent_dir, f"subject{directory}", "fixed-signs")

    for filename in os.listdir(subject_dir):
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

        # Podział punktów na grupy punktów oznaczające przerwy w pisaniu
        # Przykład: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]
        # Divide points into groups of points representing breaks in writing
        # Example: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]
        partial_xy_list = []
        full_xy_list = []

        for xy in xy_list:
            if xy == "BREAK":
                # Sprawdzenie, czy lista nie jest pusta przed dodaniem
                # Check if the list is not empty before adding
                if partial_xy_list:
                    full_xy_list.append(partial_xy_list)
                    # Resetowanie listy częściowej
                    # Reset the partial list
                    partial_xy_list = []
            else:
                # Dodawanie elementów, które nie są "BREAK"
                # Add elements that are not "BREAK"
                partial_xy_list.append(xy)

        if partial_xy_list:  # Dodanie ostatniej listy, jeśli nie była pusta
            full_xy_list.append(partial_xy_list)

        # Podział czasów na grupy czasów oznaczające przerwy w pisaniu
        # Przykład: [[t1, t2], [t3, t4]]
        # Divide times into groups of times representing breaks in writing
        # Example: [[t1, t2], [t3, t4]]
        partial_times_list = []
        full_times_list = []

        for time in times_list:
            if time == "BREAK":
                if partial_times_list:
                    full_times_list.append(partial_times_list)
                    partial_times_list = []
            else:
                partial_times_list.append(time)

        if partial_times_list:
            full_times_list.append(partial_times_list)

        xy_list2 = []
        times_list2 = []

        # Przetwarzanie punktów i czasów w celu usnięcia nadmiarowych danych
        # Process points and times to remove redundant data
        for xy_points, time_points in zip(full_xy_list, full_times_list):
            cleaned_points, cleaned_times = remove_close_points(xy_points, time_points)
            xy_list2.append(cleaned_points)
            times_list2.append(cleaned_times)

        # Wyodrębnienie nazwy pliku z obiektu pliku
        # Extract the filename from the file object
        output_filename = os.path.join(parent_dir, f"subject{directory}", "cleared-signs", f"cleared-{filename}")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Zapis przeczyszczonych danych do nowego pliku txt
        # Save cleaned data to a new txt file
        with open(output_filename, "w") as f:
            for points, times in zip(xy_list2, times_list2):
                for (x, y), time in zip(points, times):
                    f.write(f"x: {x}, y: {y}, t: {time}\n")
                f.write(f"BREAK\n")
        
        xy_list3 = []
        times_list3 = []

        # Przetwarzanie punktów i czasów w celu interpolacji danych
        # Process points and times to interpolate data
        for xy_points, time_points in zip(xy_list2, times_list2):
            interpolated_points, interpolated_times = interpolate_points(xy_points, time_points)
            xy_list3.append(interpolated_points)
            times_list3.append(interpolated_times)

        # Wyodrębnienie nazwy pliku z obiektu pliku
        # Extract the filename from the file object
        output_filename2 = os.path.join(parent_dir, f"subject{directory}", "interpolated-signs", f"interpolated-{filename}")
        os.makedirs(os.path.dirname(output_filename2), exist_ok=True)

        # Zapis zinterpolowanych danych do nowego pliku txt
        # Save interpolated data to a new txt file
        with open(output_filename2, "w") as f:
            for points, times in zip(xy_list3, times_list3):
                for (x, y), time in zip(points, times):
                    f.write(f"x: {x}, y: {y}, t: {time}\n")
                f.write(f"BREAK\n")


# Usuwanie punktów, które są zbyt blisko siebie
# Remove points that are too close to each other
def remove_close_points(points, times, min_distance=5):
    '''
    Funkcja usuwająca punkty, które są zbyt blisko siebie.
    Manimalna odległość między punktami jest określona przez min_distance.
    W tym przypadku, min_distance = 5.

    # Function that removes points that are too close to each other.
    # The minimum distance between points is determined by min_distance.
    # In this case, min_distance = 5.
    '''
    
    cleaned_points = []
    cleaned_times = []

    points = np.array(points)
    times = np.array(times)
    
    for point, time in zip(points, times):
        if cleaned_points:
            # Sprawdzenie, czy odległość euklidesowa między punktem a wszystkimi punktami w cleaned_points jest większa niż min_distance
            # Check if the euclidean distance between the point and all points in cleaned_points is greater than min_distance
            if all(
                np.linalg.norm(point - np.array(cleaned_points), axis=1) >= min_distance
            ):
                cleaned_points.append(point.tolist())
                cleaned_times.append(time.tolist())
        else:
            # Dodanie pierwszego punktu (x, y) i czasu t do odpowiednich list
            cleaned_points.append(point.tolist()) 
            cleaned_times.append(time.tolist())

    return cleaned_points, cleaned_times

# Interpolowanie punktów tam, gdzie ich brakuje
# Interpolate points where they are missing
def interpolate_points(points, times, min_distance=10, max_distance=100):
    '''
    Funkcja interpolująca brakujące punkty.
    Punkty są interpolowane, jeżeli dystans między istniejącymi punktami jest większy niż min_distance i mniejszy niż max_distance.
    W tym przypadku, min_distance = 10 i max_distance = 100.

    # Function that interpolates missing points.
    # Points are interpolated if the distance between existing points is greater than min_distance and less than max_distance.
    # In this case, min_distance = 10 and max_distance = 100.
    '''
    interpolated_points = []
    interpolated_times = []

    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]

        start_time = times[i]
        end_time = times[i + 1]

        # Obliczenie odległości euklidesowej między punktami
        # Calculate the euclidean distance between points
        distance = np.linalg.norm(np.array(start_point) - np.array(end_point))

        # Dodanie zaokrąglonego do części całkowitych początkowego punktu i czasu
        # Add the rounded to integers start point and time
        interpolated_points.append([round(start_point[0]), round(start_point[1])])
        interpolated_times.append(round(start_time))

        if distance > min_distance and distance < max_distance:
            
            # Obliczanie liczby punktów do interpolacji
            # Calculate the number of points to interpolate
            num_points = int(distance // min_distance)

            for j in range(1, num_points + 1):
                # Interpolacja liniowa dla punktów
                # Linear interpolation for points
                interp_point = [
                    start_point[0] + j * (end_point[0] - start_point[0]) / (num_points + 1),
                    start_point[1] + j * (end_point[1] - start_point[1]) / (num_points + 1),
                ]

                # Interpolacja liniowa dla czasu w mikrosekundach
                # Linear interpolation for time in microseconds
                interp_time = start_time + j * (end_time - start_time) / (num_points + 1)

                # Dodanie zinterpolowanych punktów i czasów do odpowiednich list
                # Add interpolated points and times to the appropriate lists
                interpolated_points.append([round(interp_point[0]), round(interp_point[1])])
                interpolated_times.append(round(interp_time))

    # Dodanie zaokrąglonego do części całkowitych ostatniego punktu i czasu do odpowiednich list
    # Add the rounded to integers last point and time to the appropriate lists
    interpolated_points.append([round(points[-1][0]), round(points[-1][1])])
    interpolated_times.append(round(times[-1]))

    return interpolated_points, interpolated_times

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
