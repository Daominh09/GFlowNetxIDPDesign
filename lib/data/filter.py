import csv

# File paths
input_file = 'dataset.csv'  # Change to your input file path
output_file = 'dataset_filtered.csv'  # Output file name

# Read and filter sequences
filtered_rows = []
total_count = 0
filtered_count = 0

with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    
    for row in reader:
        total_count += 1
        seq_length = len(row['sequence'])
        
        if seq_length <= 1024:
            filtered_rows.append(row)
            filtered_count += 1
            print(f"Keeping: {row['uniprot_id']} (length: {seq_length})")
        else:
            print(f"Filtering out: {row['uniprot_id']} (length: {seq_length})")

# Write filtered data to new CSV
with open(output_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

# Print summary
print("\n" + "=" * 50)
print("FILTERING SUMMARY")
print("=" * 50)
print(f"Total sequences: {total_count}")
print(f"Sequences with length <= 1024: {filtered_count}")
print(f"Sequences filtered out (length > 1024): {total_count - filtered_count}")
print(f"\nFiltered data saved to: {output_file}")