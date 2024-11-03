import os
import numpy as np

def is_signature_genuine(profile_data, signature_data):
    # Inicjacja progu minimalnego dla liczby zgodnych cech
    # Initialize the minimum threshold for the number of matching features
    genuine_features_count = 0

    # Sprawdzanie zgodności cech dla każdej z 10 cech
    # Checking feature compatibility for each of the 10 features
    for feature_index in range(10):
        matches = []

        # Inicjacja pętli o długości podziału czasowego w pliku profilu
        # Initialize a loop of the length of the time division in the profile file
        for t in range(len(profile_data)):
            mean = profile_data[t][feature_index * 2]
            std_dev = profile_data[t][feature_index * 2 + 1]
            value = signature_data[t][feature_index]

            if std_dev == 0:
                continue

            # Sprawdzenie odległości wartości cechy od średniej na poziomie 1 odchylenia standardowego
            # Check the distance of the feature value from the mean at the level of 1 standard deviation
            if abs(value - mean) <= std_dev:
                # Uznanie zgodności cechy
                # Recognition of feature compatibility
                matches.append(1)
            else:
                # Odrzucenie zgodności cechy
                # Rejection of feature compatibility
                matches.append(0)
        
        # Jeżeli średnia zgodność metryk dla danej cechy zmierzona na poszczególnych odcinkach czasu wynosi co najmniej 60%, to cecha jest uznawana za zgodną
        # If the average metric compatibility for a given feature measured at individual time intervals is at least 60%, the feature is considered compatible
        if np.mean(matches) >= 0.60:
            genuine_features_count += 1

    # Jeżeli liczba zgodnych cech wynosi co najmniej 7, to podpis jest uznawany za autentyczny
    # If the number of matching features is at least 7, the signature is considered genuine
    return genuine_features_count >= 7

def process_directory(directory):
    # Etap 1: wczytywanie profilów
    # Stage 1: loading profiles
    for profile_num in range(1, 57):
        profile_filename = os.path.join(parent_dir, "profiles", f"profile-{profile_num}.txt")
        with open(profile_filename, "r") as profile_file:
            profile_data = []
            for line in profile_file:
                profile_data.append([float(x) for x in line.strip().split(", ")])

        # Etap 2: wczytywanie podpisów
        # Stage 2: loading signatures
        extracted_features_dir = os.path.join(parent_dir, f"subject{directory}", "extracted-features")
        for filename in os.listdir(extracted_features_dir):
            print(profile_num, directory, filename)
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
