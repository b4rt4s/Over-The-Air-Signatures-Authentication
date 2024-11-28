import re
import matplotlib.pyplot as plt
import os

def split_points(xy_list, num_parts):
    N = len(xy_list)
    base_size = N // num_parts
    remainder = N % num_parts
    sublists = []
    start = 0

    for i in range(num_parts):
        end = start + base_size
        if i < remainder:
            end += 1
        sublists.append(xy_list[start:end])
        start = end
    return sublists

directory = int(input("Enter directory number to read: "))
parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)
directory_path = os.path.join(parent_dir, f"subject{directory}")

# Przygotowanie podpisu z folderu fixed-signs do wyświetlenia na wykresie.
# Prepare signature from fixed-signs folder to display on the plot.
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

# Przygotowanie podpisu z folderu cleared-signs do wyświetlenia na wykresie.
# Prepare signature from cleared-signs folder to display on the plot.
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


# Przygotowanie podpisu z folderu interpolated-signs do wyświetlenia na wykresie.
# Prepare signature from interpolated-signs folder to display on the plot.
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

# Przygotowanie podpisu z folderu normalized-signs do wyświetlenia na wykresie.
# Prepare signature from normalized-signs folder to display on the plot.
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

# Przygotowanie jednej wspólnej przestrzeni do wyświetlenia 5 wykresów.
# Prepare one common space to display 5 plots.
fig, axs = plt.subplots(6, 1, figsize=(10, 10))

# Wykres nr 1 - punkty z folderu fixed-signs.
# Plot 1 - points from fixed-signs.
x_vals_fixed, y_vals_fixed = zip(*fixed_xy_list)
axs[0].scatter(x_vals_fixed, y_vals_fixed, marker="o", color="blue", s=1)
axs[0].set_title("Points from fixed-signs")
axs[0].set_xlabel("x - axis")
axs[0].set_ylabel("y - axis")

# Wykres nr 2 - punkty z folderu cleared-signs.
# Plot 2 - points from cleared-signs.
x_vals_cleared, y_vals_cleared = zip(*cleared_xy_list)
axs[1].scatter(x_vals_cleared, y_vals_cleared, marker="o", color="green", s=1)
axs[1].set_title("Points from cleared-signs")
axs[1].set_xlabel("x - axis")
axs[1].set_ylabel("y - axis")

# Wykres nr 3 - punkty z folderu interpolated-signs.
# Plot 3 - points from interpolated-signs.
x_vals_interpolated, y_vals_interpolated = zip(*interpolated_xy_list)
axs[2].scatter(x_vals_interpolated, y_vals_interpolated, marker="o", color="red", s=1)
axs[2].set_title("Points from interpolated-signs")
axs[2].set_xlabel("x - axis")
axs[2].set_ylabel("y - axis")

# Wykres nr 4 - punkty z folderu normalized-signs.
# Plot 4 - points from normalized-signs.
x_vals_normalized, y_vals_normalized = zip(*normalized_xy_list)
axs[3].scatter(x_vals_normalized, y_vals_normalized, marker="o", color="purple", s=1)
axs[3].set_title("Points from normalized-signs")
axs[3].set_xlabel("x - axis")
axs[3].set_ylabel("y - axis")

# Podział normalized_xy_list na sublisty
# Splitting normalized_xy_list into sublists
num_parts = 20
sublists = split_points(normalized_xy_list, num_parts)

# Wykres nr 5 - punkty z folderu normalized-signs z podziałami czasowymi
# Plot 5 - points from normalized-signs with time splits
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', '+', 'x', '|', '_']
colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'brown']

for idx, sublist in enumerate(sublists):
    x_vals_split, y_vals_split = zip(*sublist)
    color = colors[idx % len(colors)]
    marker_style = markers[(idx // len(colors)) % len(markers)]
    axs[4].scatter(x_vals_split, y_vals_split, marker=marker_style, color=color, s=1)

# Obliczenie indeksów punktów, gdzie następują podziały
split_indices = []
start = 0
N = len(normalized_xy_list)
base_size = N // num_parts
remainder = N % num_parts

for i in range(num_parts):
    end = start + base_size
    if i < remainder:
        end += 1
    if end < N:
        split_indices.append(end)
    start = end

# Dodanie linii pionowych w miejscach podziałów
for idx in split_indices:
    x_split = normalized_xy_list[idx][0]
    axs[4].axvline(x=x_split, color='red', linestyle='--', linewidth=0.5)

def split_points(xy_list, num_parts):
    N = len(xy_list)
    base_size = N // num_parts
    remainder = N % num_parts
    sublists = []
    start = 0

    for i in range(num_parts):
        end = start + base_size
        if i < remainder:
            end += 1
        sublists.append(xy_list[start:end])
        start = end
    return sublists

axs[4].set_title("Points from normalized-signs with splits")
axs[4].set_xlabel("x - axis")
axs[4].set_ylabel("y - axis")

# Wykres nr 6 - naniesienie na siebie punktów z folderów fixed-signs, cleared-signs, interpolated-signs, normalized-signs.
# Plot 6 - overlaying points from fixed-signs, cleared-signs, interpolated-signs, normalized-signs.
axs[5].scatter(x_vals_fixed, y_vals_fixed, marker="o", color="blue", s=1, label="fixed-signs")
axs[5].scatter(x_vals_cleared, y_vals_cleared, marker="o", color="green", s=1, label="cleared-signs")
axs[5].scatter(x_vals_interpolated, y_vals_interpolated, marker="o", color="red", s=1, label="interpolated-signs")
axs[5].scatter(x_vals_normalized, y_vals_normalized, marker="o", color="purple", s=1, label="normalized-signs")
axs[5].scatter(x_vals_split, y_vals_split, marker="o", color="purple", s=1, label="split-normalized-signs")
axs[5].set_title("Combined points from fixed-signs, cleared-signs, interpolated-signs, and normalized-signs")
axs[5].set_xlabel("x - axis")
axs[5].set_ylabel("y - axis")

# Wyświetlenie wykresów.
# Display plots.
plt.tight_layout()
plt.show()
