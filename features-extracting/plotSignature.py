import re
import matplotlib.pyplot as plt
import os

directory = int(input("Enter directory number to read: "))
parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)
directory_path = os.path.join(parent_dir, f"subject{directory}")

# Find the subfolder starting with 'fixed-signs'
fixed_subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f)) and f.startswith('fixed-signs')]
if len(fixed_subfolders) == 0:
    print("No subfolders starting with 'fixed-signs' found.")
    exit(1)
fixed_subfolder = fixed_subfolders[0]

fixed_subfolder_path = os.path.join(directory_path, fixed_subfolder)
file_num = int(input("Enter file number to read: "))
fixed_filename = os.path.join(fixed_subfolder_path, f"sign_{file_num}.txt")

if not os.path.isfile(fixed_filename):
    print(f"File {fixed_filename} does not exist.")
    exit(1)

fixed_xy_list = []

with open(fixed_filename, "r") as file:
    for line in file:
        line = line.strip()
        if line != "BREAK":
            match = re.search(r"x: (-?\d+), y: (-?\d+)", line)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
                fixed_xy_list.append((x_coord, y_coord))

# Find the subfolder starting with 'cleared-signs'
cleared_subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f)) and f.startswith('cleared-signs')]
if len(cleared_subfolders) == 0:
    print("No subfolders starting with 'cleared-signs' found.")
    exit(1)
cleared_subfolder = cleared_subfolders[0]

cleared_subfolder_path = os.path.join(directory_path, cleared_subfolder)
cleared_filename = os.path.join(cleared_subfolder_path, f"cleared-sign_{file_num}.txt")

if not os.path.isfile(cleared_filename):
    print(f"File {cleared_filename} does not exist.")
    exit(1)

cleared_xy_list = []

with open(cleared_filename, "r") as file:
    for line in file:
        line = line.strip()
        if line != "BREAK":
            match = re.search(r"x: (-?\d+), y: (-?\d+)", line)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
                cleared_xy_list.append((x_coord, y_coord))

# Find the subfolder starting with 'interpolated-signs'
interpolated_subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f)) and f.startswith('interpolated-signs')]
if len(interpolated_subfolders) == 0:
    print("No subfolders starting with 'interpolated-signs' found.")
    exit(1)
interpolated_subfolder = interpolated_subfolders[0]

interpolated_subfolder_path = os.path.join(directory_path, interpolated_subfolder)
interpolated_filename = os.path.join(interpolated_subfolder_path, f"interpolated-sign_{file_num}.txt")

if not os.path.isfile(interpolated_filename):
    print(f"File {interpolated_filename} does not exist.")
    exit(1)

interpolated_xy_list = []

with open(interpolated_filename, "r") as file:
    for line in file:
        line = line.strip()
        if line != "BREAK":
            match = re.search(r"x: (-?\d+), y: (-?\d+)", line)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
                interpolated_xy_list.append((x_coord, y_coord))

# Find the subfolder starting with 'normalized-signs'
normalized_subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f)) and f.startswith('normalized-signs')]
if len(normalized_subfolders) == 0:
    print("No subfolders starting with 'normalized-signs' found.")
    exit(1)
normalized_subfolder = normalized_subfolders[0]

normalized_subfolder_path = os.path.join(directory_path, normalized_subfolder)
normalized_filename = os.path.join(normalized_subfolder_path, f"normalized-sign_{file_num}.txt")

if not os.path.isfile(normalized_filename):
    print(f"File {normalized_filename} does not exist.")
    exit(1)

normalized_xy_list = []

with open(normalized_filename, "r") as file:
    for line in file:
        line = line.strip()
        if line != "BREAK":
            match = re.search(r"x: (-?\d+), y: (-?\d+)", line)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
                normalized_xy_list.append((x_coord, y_coord))

# Create figure and axes
fig, axs = plt.subplots(
    5, 1, figsize=(10, 25)
)  # 5 plots vertically, 1 column, figure size 10x25 inches

# Plot 1: Original points from fixed-signs
x_vals_fixed, y_vals_fixed = zip(*fixed_xy_list)  # Unpacking list into x and y
axs[0].scatter(x_vals_fixed, y_vals_fixed, marker="o", color="blue", s=1)
axs[0].set_title("Original Points from fixed-signs")
axs[0].set_xlabel("x - axis")
axs[0].set_ylabel("y - axis")

# Plot 2: Original points from cleared-signs
x_vals_cleared, y_vals_cleared = zip(*cleared_xy_list)  # Unpacking list into x and y
axs[1].scatter(x_vals_cleared, y_vals_cleared, marker="o", color="green", s=1)
axs[1].set_title("Original Points from cleared-signs")
axs[1].set_xlabel("x - axis")
axs[1].set_ylabel("y - axis")

# Plot 3: Original points from interpolated-signs
x_vals_interpolated, y_vals_interpolated = zip(*interpolated_xy_list)  # Unpacking list into x and y
axs[2].scatter(x_vals_interpolated, y_vals_interpolated, marker="o", color="red", s=1)
axs[2].set_title("Original Points from interpolated-signs")
axs[2].set_xlabel("x - axis")
axs[2].set_ylabel("y - axis")

# Plot 4: Original points from normalized-signs
x_vals_normalized, y_vals_normalized = zip(*normalized_xy_list)  # Unpacking list into x and y
axs[3].scatter(x_vals_normalized, y_vals_normalized, marker="o", color="purple", s=1)
axs[3].set_title("Original Points from normalized-signs")
axs[3].set_xlabel("x - axis")
axs[3].set_ylabel("y - axis")

# Plot 5: Combined points from all folders
axs[4].scatter(x_vals_fixed, y_vals_fixed, marker="o", color="blue", s=1, label="fixed-signs")
axs[4].scatter(x_vals_cleared, y_vals_cleared, marker="o", color="green", s=1, label="cleared-signs")
axs[4].scatter(x_vals_interpolated, y_vals_interpolated, marker="o", color="red", s=1, label="interpolated-signs")
axs[4].scatter(x_vals_normalized, y_vals_normalized, marker="o", color="purple", s=1, label="normalized-signs")
axs[4].set_title("Combined Points from fixed-signs, cleared-signs, interpolated-signs, and normalized-signs")
axs[4].set_xlabel("x - axis")
axs[4].set_ylabel("y - axis")
axs[4].legend()

# Set appropriate spacing between plots
plt.tight_layout()

# Display plots
plt.show()
