import os
import re

def process_directory(directory):
    directory_path = os.path.join(parent_dir, f"subject{directory}")

    # Create subdirectories "signature-pngs", "original-signs", and "fixed-signs"
    signature_pngs_dir = os.path.join(directory_path, "signature-pngs")
    original_signs_dir = os.path.join(directory_path, "original-signs")
    fixed_signs_dir = os.path.join(directory_path, "fixed-signs")

    os.makedirs(signature_pngs_dir, exist_ok=True)
    os.makedirs(original_signs_dir, exist_ok=True)
    os.makedirs(fixed_signs_dir, exist_ok=True)

    # Move PNG files to "signature-pngs" and process text files
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith('.png'):
                os.rename(file_path, os.path.join(signature_pngs_dir, filename))
            elif filename.lower().endswith('.txt'):
                process_text_file(file_path, original_signs_dir, fixed_signs_dir)

def process_text_file(filename, original_signs_dir, fixed_signs_dir):
    xy_list = []
    times_list = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            match = re.search(
                r"id: (-?\d+), x: (-?\d+), y: (-?\d+), t: (\d+:\d+:\d+)", line
            )
            if match:
                x_coord = int(match.group(2))
                y_coord = int(match.group(3))
                xy_list.append((x_coord, y_coord))

                time_str = match.group(4)
                minutes, seconds, microseconds = map(int, time_str.split(":"))
                time_tuple = (minutes, seconds, microseconds)
                times_list.append(time_tuple)
            else:
                xy_list.append((-1, -1))
                times_list.append((-1, -1, -1))

    # Move the original file to "original-signs"
    os.rename(filename, os.path.join(original_signs_dir, os.path.basename(filename)))

    # Write the processed data to a new file in "fixed-signs"
    fixed_filename = os.path.join(fixed_signs_dir, os.path.basename(filename))
    with open(fixed_filename, "w") as file:
        previous_minutes = None
        hour = 0
        consecutive_invalid_count = 0
        for (x, y), (m, s, us) in zip(xy_list, times_list):
            if previous_minutes is not None and previous_minutes == 59 and m == 0:
                hour += 1
            previous_minutes = m

            total_microseconds = ((hour * 60 + m) * 60 + s) * 1000000 + us

            if x == -1 and y == -1:
                consecutive_invalid_count += 1
                if consecutive_invalid_count > 1:
                    continue
            else:
                consecutive_invalid_count = 0

            if x == -1 and y == -1:
                file.write(f"BREAK\n")
            else:
                file.write(f"x: {x}, y: {-y}, t: {total_microseconds}\n")

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
