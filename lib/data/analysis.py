import csv
import statistics

# Read the CSV file
filename = 'dataset.csv'  # Change this to your file path

sequences = []
scores = []

with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        sequences.append(len(row['sequence']))
        scores.append(float(row['score']))

# Calculate statistics for sequence lengths
print("=" * 50)
print("SEQUENCE LENGTH STATISTICS")
print("=" * 50)
print(f"Count: {len(sequences)}")
print(f"Min length: {min(sequences)}")
print(f"Max length: {max(sequences)}")
print(f"Mean length: {statistics.mean(sequences):.2f}")
print(f"Median length: {statistics.median(sequences):.2f}")
print(f"Std deviation: {statistics.stdev(sequences):.2f}")

print("\n" + "=" * 50)
print("SCORE STATISTICS")
print("=" * 50)
print(f"Count: {len(scores)}")
print(f"Min score: {min(scores):.6f}")
print(f"Max score: {max(scores):.6f}")
print(f"Mean score: {statistics.mean(scores):.6f}")
print(f"Median score: {statistics.median(scores):.6f}")
print(f"Std deviation: {statistics.stdev(scores):.6f}")

