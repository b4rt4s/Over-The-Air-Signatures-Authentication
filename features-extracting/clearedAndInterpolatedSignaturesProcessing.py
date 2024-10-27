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
        partial_xy_list = []
        full_xy_list = []

        for xy in xy_list:
            if xy != "BREAK":
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
            if time != "BREAK":
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

        # Wyodrębnienie nazwy pliku z obiektu pliku
        output_filename = os.path.join(parent_dir, f"subject{directory}", "cleared-signs", f"cleared-{filename}")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Zapis do pliku txt
        with open(output_filename, "w") as f:
            for points, times in zip(xy_list2, times_list2):
                for (x, y), time in zip(points, times):
                    f.write(f"x: {x}, y: {y}, t: {time}\n")
                f.write(f"BREAK\n")
        
        xy_list3 = []
        times_list3 = []

        for xy_points, time_points in zip(xy_list2, times_list2):
            interpolated_points, interpolated_times = interpolate_points(xy_points, time_points)
            xy_list3.append(interpolated_points)
            times_list3.append(interpolated_times)

        # Wyodrębnienie nazwy pliku z obiektu pliku
        output_filename2 = os.path.join(parent_dir, f"subject{directory}", "interpolated-signs", f"interpolated-{filename}")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filename2), exist_ok=True)

        # Zapis do pliku txt
        with open(output_filename2, "w") as f:
            for points, times in zip(xy_list3, times_list3):
                for (x, y), time in zip(points, times):
                    f.write(f"x: {x}, y: {y}, t: {time}\n")
                f.write(f"BREAK\n")


# Usuwanie punktów, które są zbyt blisko siebie
def remove_close_points(points, times, min_distance=4):
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
    interpolated_points = []
    interpolated_times = []

    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]

        start_time = times[i]
        end_time = times[i + 1]

        # Obliczenie odległości między punktami
        distance = np.linalg.norm(np.array(start_point) - np.array(end_point))

        # Dodanie początkowego punktu i czasu
        interpolated_points.append([round(start_point[0]), round(start_point[1])])
        interpolated_times.append(round(start_time))

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
                interp_time = start_time + j * (end_time - start_time) / (num_points + 1)

                interpolated_points.append([round(interp_point[0]), round(interp_point[1])])
                interpolated_times.append(round(interp_time))

    # Dodaj ostatni punkt i czas
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
