import os
import numpy as np

# Funkcja sprawdzająca autentyczność podpisu według zdefiniowanych thresholdów
def is_signature_genuine(profile_data, signature_data, mean_matches_threshold, genuine_features_threshold):
    # Inicjacja progu minimalnego dla liczby zgodnych cech
    genuine_features_count = 0

    # Sprawdzanie zgodności cech dla każdej z 10 cech
    for feature_index in range(10):
        matches = []

        # Inicjacja pętli o długości podziału czasowego w pliku profilu
        for t in range(len(profile_data)):
            mean = profile_data[t][feature_index * 2]
            std_dev = profile_data[t][feature_index * 2 + 1]
            value = signature_data[t][feature_index]

            if std_dev == 0:
                continue

            # Sprawdzenie odległości wartości cechy od średniej na poziomie 1 odchylenia standardowego
            if abs(value - mean) <= std_dev:
                # Uznanie zgodności cechy
                matches.append(1)
            else:
                # Odrzucenie zgodności cechy
                matches.append(0)
        
        # Sprawdzenie, czy średnia zgodność dla danej cechy przekracza próg mean_matches_threshold
        if len(matches) > 0 and np.mean(matches) >= mean_matches_threshold:
            genuine_features_count += 1

    # Jeżeli liczba zgodnych cech wynosi co najmniej genuine_features_threshold, to podpis jest uznawany za autentyczny
    return genuine_features_count >= genuine_features_threshold

# Funkcja wczytująca wszystkie profile z plików
def load_all_profiles(parent_dir):
    profiles = {}
    for profile_num in range(1, 57):
        profile_filename = os.path.join(parent_dir, "profiles", f"profile-{profile_num}.txt")
        with open(profile_filename, "r") as profile_file:
            profile_data = [[float(x) for x in line.strip().split(", ")] for line in profile_file]
        profiles[profile_num] = profile_data
    return profiles

# Funkcja zwracająca liczbę błędnie uznanych prób autentycznych i nieautentycznych podpisów
def process_user(user_num, profiles, parent_dir, mean_matches_threshold, genuine_features_threshold):
    # Etap 1: wczytywanie profilów 
    profile_data = profiles[user_num]

    # Etap 2: wczytywanie podpisów
    extracted_features_dir = os.path.join(parent_dir, f"subject{user_num}", "extracted-features")
    signature_filenames = os.listdir(extracted_features_dir)

    # Porównanie podpisów użytkownika względem jego własnego profilu - sprawdzenie błędu FRR
    FRR_count = 0
    N_genuine_attempts = 0

    for filename in signature_filenames:
        file_path = os.path.join(extracted_features_dir, filename)

        with open(file_path, "r") as feature_file:
            signature_data = [[float(x) for x in line.strip().split(", ")] for line in feature_file]

        N_genuine_attempts += 1

        if not is_signature_genuine(profile_data, signature_data, mean_matches_threshold, genuine_features_threshold):
            FRR_count += 1

    # Porównanie podpisów użytkownika względem profili innych użytkowników - sprawdzenie błędu FAR
    FAR_count = 0
    N_impostor_attempts = 0
    
    for filename in signature_filenames:
        file_path = os.path.join(extracted_features_dir, filename)

        with open(file_path, "r") as feature_file:
            signature_data = [[float(x) for x in line.strip().split(", ")] for line in feature_file]

        for other_profile_num, other_profile_data in profiles.items():
            # Pominięcie porównania podpisów użytkownika względem jego własnego profilu
            if other_profile_num == user_num:
                continue
            
            N_impostor_attempts += 1

            if is_signature_genuine(other_profile_data, signature_data, mean_matches_threshold, genuine_features_threshold):
                FAR_count += 1

    return FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts

# Główna część programu
parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signatures-database",
)

profiles = load_all_profiles(parent_dir)

# Definicja zakresów progów
mean_matches_thresholds = np.arange(0.1, 1.0, 0.1)
genuine_features_thresholds = range(1, 11)

choice = input("Enter 'range' to specify a range of subjects or 'all' to process all subjects: ").strip().lower()

# Zapis wyników FAR i FRR do pliku w celu późniejszego wykorzystania tych wartości do stworzenia wykresów
with open('far_frr_results.txt', 'w') as result_file:
    result_file.write('far,frr,mean_matches_threshold,genuine_features_threshold\n')

    if choice == 'range':
        start = int(input("Enter start subject number: "))
        end = int(input("Enter end subject number: "))

        for mean_matches_threshold in mean_matches_thresholds:
            for genuine_features_threshold in genuine_features_thresholds:
                total_FRR_count = 0
                total_FAR_count = 0
                total_N_genuine_attempts = 0
                total_N_impostor_attempts = 0

                for user_num in range(start, end + 1):
                    FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts = process_user(
                        user_num, profiles, parent_dir, mean_matches_threshold, genuine_features_threshold)
                    total_FRR_count += FRR_count
                    total_FAR_count += FAR_count
                    total_N_genuine_attempts += N_genuine_attempts
                    total_N_impostor_attempts += N_impostor_attempts

                # Obliczenie wartości średniego błędu FRR i FAR dla całego systemu
                if total_N_genuine_attempts > 0:
                    FRR_rate = (total_FRR_count / total_N_genuine_attempts) * 100
                else:
                    FRR_rate = 0.0

                if total_N_impostor_attempts > 0:
                    FAR_rate = (total_FAR_count / total_N_impostor_attempts) * 100
                else:
                    FAR_rate = 0.0

                print(f"Mean Matches Threshold: {mean_matches_threshold:.1f}, Genuine Features Threshold: {genuine_features_threshold}")
                print(f"Total FAR: {FAR_rate:.2f}%, Total FRR: {FRR_rate:.2f}%")

                # Zapisz wyniki do pliku, łącząc progi w jednym polu
                result_file.write(f"{FAR_rate:.2f},{FRR_rate:.2f},[{mean_matches_threshold:.1f},{genuine_features_threshold}]\n")

    elif choice == 'all':
        for mean_matches_threshold in mean_matches_thresholds:
            for genuine_features_threshold in genuine_features_thresholds:
                total_FRR_count = 0
                total_FAR_count = 0
                total_N_genuine_attempts = 0
                total_N_impostor_attempts = 0

                for directory in os.listdir(parent_dir):
                    if directory.startswith("subject") and directory[7:].isdigit():
                        user_num = int(directory[7:])
                        FRR_count, FAR_count, N_genuine_attempts, N_impostor_attempts = process_user(
                            user_num, profiles, parent_dir, mean_matches_threshold, genuine_features_threshold)
                        total_FRR_count += FRR_count
                        total_FAR_count += FAR_count
                        total_N_genuine_attempts += N_genuine_attempts
                        total_N_impostor_attempts += N_impostor_attempts

                # Obliczenie wartości średniego błędu FRR i FAR dla całego systemu
                if total_N_genuine_attempts > 0:
                    FRR_rate = (total_FRR_count / total_N_genuine_attempts) * 100
                else:
                    FRR_rate = 0.0

                if total_N_impostor_attempts > 0:
                    FAR_rate = (total_FAR_count / total_N_impostor_attempts) * 100
                else:
                    FAR_rate = 0.0

                print(f"Mean Matches Threshold: {mean_matches_threshold:.1f}, Genuine Features Threshold: {genuine_features_threshold}")
                print(f"Total FAR: {FAR_rate:.2f}%, Total FRR: {FRR_rate:.2f}%")

                # Zapisz wyniki do pliku, łącząc progi w jednym polu
                result_file.write(f"{FAR_rate:.2f},{FRR_rate:.2f},[{mean_matches_threshold:.1f},{genuine_features_threshold}]\n")

    else:
        print("Invalid choice. Please enter 'range' or 'all'.")
