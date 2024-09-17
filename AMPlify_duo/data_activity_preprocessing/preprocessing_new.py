from Bio import SeqIO
import os
import random

def remove_duplicates(sequences):
    unique_seqs = {}
    for seq_record in sequences:
        if str(seq_record.seq) not in unique_seqs:
            unique_seqs[str(seq_record.seq)] = seq_record
    return list(unique_seqs.values())

def split_train_test(data, train_ratio=0.8):
    """Funkcja dzieląca dane na zbiory treningowy i testowy w zadanej proporcji."""
    train_size = int(len(data) * train_ratio)
    train_set = random.sample(data, train_size)
    
    # Zamiast porównywać obiekty SeqRecord, porównujemy atrybuty .seq
    train_seqs = set([str(record.seq) for record in train_set])
    test_set = [record for record in data if str(record.seq) not in train_seqs]
    return train_set, test_set

# Ścieżki do plików dla Staphylococcus aureus
active_file_path_saureus = "./data_activity_dbaasp/staphylococcusaureus_active_32.fa"
inactive_file_path_saureus = "./data_activity_dbaasp/staphylococcusaureus_inactive_128.fa"
active_file_path_ecoli = "./data_activity_dbaasp/escherichiacoli_active_32.fa"
inactive_file_path_ecoli = "./data_activity_dbaasp/escherichiacoli_inactive_128.fa"

# Odczyt sekwencji
active_sequences_saureus = list(SeqIO.parse(active_file_path_saureus, "fasta"))
inactive_sequences_saureus = list(SeqIO.parse(inactive_file_path_saureus, "fasta"))
active_sequences_ecoli = list(SeqIO.parse(active_file_path_ecoli, "fasta"))
inactive_sequences_ecoli = list(SeqIO.parse(inactive_file_path_ecoli, "fasta"))

# Usunięcie duplikatów
active_sequences_saureus = remove_duplicates(active_sequences_saureus)
inactive_sequences_saureus = remove_duplicates(inactive_sequences_saureus)
active_sequences_ecoli = remove_duplicates(active_sequences_ecoli)
inactive_sequences_ecoli = remove_duplicates(inactive_sequences_ecoli)


# Listy do przechowywania sekwencji i rekordów w celu znalezienia wspolnych sekwencji
escherichia_records = {'active': [], 'inactive': []}
staphylo_records = {'active': [], 'inactive': []}
all_sequences_escherichia = set()
all_sequences_staphylo = set()

# Uzupełnienie sekwencji dla E. coli
for record in active_sequences_ecoli:
    escherichia_records['active'].append(record)
    all_sequences_escherichia.add(str(record.seq))

for record in inactive_sequences_ecoli:
    escherichia_records['inactive'].append(record)
    all_sequences_escherichia.add(str(record.seq))

# Uzupełnienie sekwencji dla S. aureus
for record in active_sequences_saureus:
    staphylo_records['active'].append(record)
    all_sequences_staphylo.add(str(record.seq))

for record in inactive_sequences_saureus:
    staphylo_records['inactive'].append(record)
    all_sequences_staphylo.add(str(record.seq))

# Znajdowanie wspólnych sekwencji (zbiór C)
union_sequences = all_sequences_escherichia & all_sequences_staphylo
print(f"Liczba wspólnych sekwencji: {len(union_sequences)}")


# Opcjonalnie możesz filtrować oryginalne rekordy, aby zawierały tylko wspólne sekwencje
filtered_escherichia_active = [record for record in escherichia_records['active'] if str(record.seq) in union_sequences]
filtered_escherichia_inactive = [record for record in escherichia_records['inactive'] if str(record.seq) in union_sequences]
filtered_staphylo_active = [record for record in staphylo_records['active'] if str(record.seq) in union_sequences]
filtered_staphylo_inactive = [record for record in staphylo_records['inactive'] if str(record.seq) in union_sequences]


# Podział każdego zbioru na treningowy (80%) i testowy (20%)
union_escherichia_active_train, union_escherichia_active_test = split_train_test(filtered_escherichia_active)
union_escherichia_inactive_train, union_escherichia_inactive_test = split_train_test(filtered_escherichia_inactive)
union_staphylo_active_train, union_staphylo_active_test = split_train_test(filtered_staphylo_active)
union_staphylo_inactive_train, union_staphylo_inactive_test = split_train_test(filtered_staphylo_inactive)

# Pliki do trenowania amplify bez czesci do testowania
escherichia_active_test_seqs = set(str(record.seq) for record in union_escherichia_active_test)
escherichia_inactive_test_seqs = set(str(record.seq) for record in union_escherichia_inactive_test)
staphylo_active_test_seqs = set(str(record.seq) for record in union_staphylo_active_test)
staphylo_inactive_test_seqs = set(str(record.seq) for record in union_staphylo_inactive_test)


escherichia_active = [record for record in active_sequences_ecoli if str(record.seq) not in escherichia_active_test_seqs]
escherichia_inactive = [record for record in inactive_sequences_ecoli if str(record.seq) not in escherichia_inactive_test_seqs]
staphylo_active = [record for record in active_sequences_saureus if str(record.seq) not in staphylo_active_test_seqs]
staphylo_inactive = [record for record in inactive_sequences_saureus if str(record.seq) not in staphylo_inactive_test_seqs]


# Zapisanie przefiltrowanych sekwencji do plików (jeśli potrzebne)
output_folder = './data_union/'
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, 'union_ecoli_active_train.fa'), 'w') as output_handle:
    SeqIO.write(union_escherichia_active_train, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_ecoli_inactive_train.fa'), 'w') as output_handle:
    SeqIO.write(union_escherichia_inactive_train, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_saureus_activ_train.fa'), 'w') as output_handle:
    SeqIO.write(union_staphylo_active_train, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_saureus_inactive_train.fa'), 'w') as output_handle:
    SeqIO.write(union_staphylo_inactive_train, output_handle, 'fasta')

print("Pliki union do trenowania zostały utworzone.")

# Zapisanie przefiltrowanych sekwencji do plików (jeśli potrzebne)
output_folder = './data_test/'
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, 'union_ecoli_active_test.fa'), 'w') as output_handle:
    SeqIO.write(filtered_escherichia_active, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_ecoli_inactive_test.fa'), 'w') as output_handle:
    SeqIO.write(filtered_escherichia_inactive, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_saureus_active_test.fa'), 'w') as output_handle:
    SeqIO.write(filtered_staphylo_active, output_handle, 'fasta')

with open(os.path.join(output_folder, 'union_saureus_inactive_test.fa'), 'w') as output_handle:
    SeqIO.write(filtered_staphylo_inactive, output_handle, 'fasta')

print("Pliki union do testowania zostały utworzone.")

# Zapisanie przefiltrowanych sekwencji do plików (jeśli potrzebne)
output_folder = './data_activity/'
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, 'ecoli_active.fa'), 'w') as output_handle:
    SeqIO.write(escherichia_active, output_handle, 'fasta')

with open(os.path.join(output_folder, 'ecoli_inactive.fa'), 'w') as output_handle:
    SeqIO.write(escherichia_inactive, output_handle, 'fasta')

with open(os.path.join(output_folder, 'saureus_active.fa'), 'w') as output_handle:
    SeqIO.write(staphylo_active, output_handle, 'fasta')

with open(os.path.join(output_folder, 'saureus_inactive.fa'), 'w') as output_handle:
    SeqIO.write(staphylo_inactive, output_handle, 'fasta')

print("Pliki do trenowania amplify (z wyłączeniem zbioru do testowania) zostały utworzone.")
print("ilość seq w ecoli_active.fa:", len(escherichia_active))
print("ilość seq w ecoli_inactive.fa:", len(escherichia_inactive))
print("ilość seq w saureus_active.fa:", len(staphylo_active))
print("ilość seq w saureus_inactive.fa:", len(staphylo_inactive))

