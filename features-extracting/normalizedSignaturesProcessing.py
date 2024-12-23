import re
import os

# Normalizowanie punktów i czasów
# Normalize points and times
def normalize_data(directory):
    '''
    Funkcja normalizuje punkty i czasy do (0, 0) i 0 mikrosekundy.
    Dane, które normalizujemy to punkty i czasy z plików 'interpolated-signs'.

    Function normalizes points and times to (0, 0) and 0 microseconds.
    Data we normalize are points and times from 'interpolated-signs' files.
    '''
    subject_dir = os.path.join(parent_dir, f"subject{directory}", "interpolated-signs")

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

        # Normalizacja punktów do (0, 0) oraz czasów do 0 mikrosekund
        # Normalization of points to (0, 0) and times to 0 microseconds
        if xy_list and xy_list[0] != "BREAK":
            x_offset, y_offset = xy_list[0]
            t_offset = times_list[0]

            normalized_xy_list = []
            normalized_times_list = []

            for point, time in zip(xy_list, times_list):
                if point == "BREAK":
                    normalized_xy_list.append("BREAK")
                    normalized_times_list.append("BREAK")
                else:
                    x, y = point

                    # Odejmowanie od punktów i czasów wartości pierwszego punktu i czasu
                    # Subtracting the values of the first point and time from the points and times
                    normalized_xy_list.append((x - x_offset, y - y_offset)) 
                    normalized_times_list.append(time - t_offset)
        else:
            normalized_xy_list = xy_list
            normalized_times_list = times_list

        # Wyodrębnienie nazwy pliku z obiektu pliku i zamiana 'interpolated-sign' na 'normalized-sign'
        # Extracting the file name from the file object and replacing 'interpolated-sign' with 'normalized-sign'
        file_number = os.path.splitext(filename)[0].replace('interpolated-sign', 'normalized-sign')
        output_filename = os.path.join(parent_dir, f"subject{directory}", "normalized-signs", f"{file_number}.txt")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Zapis znoramlizowanych danych do nowego pliku txt
        # Save normalized data to a new txt file
        with open(output_filename, "w") as f:
            for points, times in zip(normalized_xy_list, normalized_times_list):
                if points == "BREAK":
                    f.write("BREAK\n")
                else:
                    x, y = points
                    f.write(f"x: {x}, y: {y}, t: {times}\n")

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

choice = input("Enter 'range' to specify a range of subjects or 'all' to process all subjects: ").strip().lower()

if choice == 'range':
    start = int(input("Enter start subject number: "))
    end = int(input("Enter end subject number: "))
    for directory in range(start, end + 1):
        normalize_data(directory)
elif choice == 'all':
    for directory in os.listdir(parent_dir):
        if directory.startswith("subject") and directory[7:].isdigit():
            normalize_data(int(directory[7:]))
else:
    print("Invalid choice. Please enter 'range' or 'all'.")
