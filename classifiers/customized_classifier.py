import os
import numpy as np

# Funkcja sprawdzająca autentyczność podpisu według zdefiniowanych thresholdów
# Function checking the authenticity of the signature according to the defined thresholds
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

# Funkcja wczytująca wszystkie profile z plików
# Function loading all profiles from files
def load_all_profiles(parent_dir):
    profiles = {}
    for profile_num in range(1, 57):
        profile_filename = os.path.join(parent_dir, "profiles", f"profile-{profile_num}.txt")
        with open(profile_filename, "r") as profile_file:
            profile_data = [[float(x) for x in line.strip().split(", ")] for line in profile_file]
        profiles[profile_num] = profile_data
    return profiles

# Funkcja zwracająca liczbę błędnie uznanych prób autentycznych i nieautentycznych podpisów
# Function returning the number of incorrectly recognized genuine and impostor attempts
def process_user(user_num, profiles, parent_dir):
    # Etap 1: wczytywanie profilów 
    # Stage 1: loading profiles
    profile_data = profiles[user_num]

    # Etap 2: wczytywanie podpisów
    # Stage 2: loading signatures
    extracted_features_dir = os.path.join(parent_dir, f"subject{user_num}", "extracted-features")
    signature_filenames = os.listdir(extracted_features_dir)

    # Porównanie podpisów użytkownika względem jego własnego profilu - sprawdzenie błędu FRR
    # Comparing user signatures to their own profile - checking the FRR error
    FRR_count = 0
    N_genuine_attempts = 0

    for filename in signature_filenames:
        file_path = os.path.join(extracted_features_dir, filename)

        with open(file_path, "r") as feature_file:
            signature_data = [[float(x) for x in line.strip().split(", ")] for line in feature_file]

        N_genuine_attempts += 1

        if not is_signature_genuine(profile_data, signature_data):
            FRR_count += 1

    # Porównanie podpisów użytkownika względem profili innych użytkowników - sprawdzenie błędu FAR
    # Comparing user signatures to profiles of other users - checking the FAR error
    FAR_count = 0
    N_impostor_attempts = 0
    
    for filename in signature_filenames:
        file_path = os.path.join(extracted_features_dir, filename)

        with open(file_path, "r") as feature_file:
            signature_data = [[float(x) for x in line.strip().split(", ")] for line in feature_file]

        for other_profile_num, other_profile_data in profiles.items():
            # Pominięcie porównania podpisów użytkownika względem jego własnego profilu
            # Skipping the comparison of user signatures to their own profile
            if other_profile_num == user_num:
                continue

            N_impostor_attempts += 1

            if is_signature_genuine(other_profile_data, signature_data):
                FAR_count += 1

    return FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

profiles = load_all_profiles(parent_dir)

total_FRR_count = 0
total_FAR_count = 0
total_N_genuine_attempts = 0
total_N_impostor_attempts = 0

choice = input("Enter 'range' to specify a range of subjects or 'all' to process all subjects: ").strip().lower()

if choice == 'range':
    start = int(input("Enter start subject number: "))
    end = int(input("Enter end subject number: "))
    for user_num in range(start, end + 1):
        FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts = process_user(user_num, profiles, parent_dir)
        total_FRR_count += FRR_count
        total_FAR_count += FAR_count
        total_N_genuine_attempts += N_genuine_attempts
        total_N_impostor_attempts += N_impostor_attempts
elif choice == 'all':
    for directory in os.listdir(parent_dir):
        if directory.startswith("subject") and directory[7:].isdigit():
            user_num = int(directory[7:])
            FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts = process_user(user_num, profiles, parent_dir)
            total_FRR_count += FRR_count
            total_FAR_count += FAR_count
            total_N_genuine_attempts += N_genuine_attempts
            total_N_impostor_attempts += N_impostor_attempts
else:
    print("Invalid choice. Please enter 'range' or 'all'.")

# Obliczenie wartości średniego błędu FRR i FAR dla całego systemu
# Calculating the average FRR and FAR error values for the entire system
if total_N_genuine_attempts > 0:
    FRR_rate = (total_FRR_count / total_N_genuine_attempts) * 100
else:
    FRR_rate = 0.0

if total_N_impostor_attempts > 0:
    FAR_rate = (total_FAR_count / total_N_impostor_attempts) * 100
else:
    FAR_rate = 0.0

print(f"Total FRR: {FRR_rate:.2f}%")
print(f"Total FAR: {FAR_rate:.2f}%")
