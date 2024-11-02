import os
import numpy as np

def is_signature_genuine(profile_data, signature_data):
    genuine_features_count = 0
    for feature_index in range(10):  # Assuming 10 features
        matches = []
        for t in range(len(profile_data)):
            mean = profile_data[t][feature_index * 2]
            std_dev = profile_data[t][feature_index * 2 + 1]
            value = signature_data[t][feature_index]
            if std_dev == 0:
                continue  # Avoid division by zero
            if abs(value - mean) <= std_dev:
                matches.append(1)
            else:
                matches.append(0)
        if np.mean(matches) >= 0.60: # od 0.01 do 0.99 
            genuine_features_count += 1
    return genuine_features_count >= 7 # od 1 do 9 i mamy 1000 ustawien, dla tego wyliczyc far i frr

def process_directory(directory):
    # Etap 1: wczytywanie danych profilu
    for profile_num in range(1, 57):
        profile_filename = os.path.join(parent_dir, "profiles", f"profile-{profile_num}.txt")
        with open(profile_filename, "r") as profile_file:
            profile_data = []
            for line in profile_file:
                profile_data.append([float(x) for x in line.strip().split(", ")])

        # Etap 2: wczytywanie danych podpisów
        extracted_features_dir = os.path.join(parent_dir, f"subject{directory}", "extracted-features")
        for filename in os.listdir(extracted_features_dir):
            file_path = os.path.join(extracted_features_dir, filename)
            with open(file_path, "r") as feature_file:
                signature_data = []
                for line in feature_file:
                    signature_data.append([float(x) for x in line.strip().split(", ")])
            if is_signature_genuine(profile_data, signature_data):
                print(f"{filename}: Genuine signature with profile {profile_num}")
            else:
                pass
                #print(f"{filename}: Forged signature with profile {profile_num}")

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
