import pandas as pd

# Read the TSV file
df = pd.read_csv('datasets/dataset.tsv', sep='\t')

# Select and rename columns
df = df.rename(columns={'UniProt.Acc': 'uniprot_id', 'Full.seq': 'sequence'})
cols = ['uniprot_id', 'sequence']

# Filter for Client and Driver (C_D)
cd_df = df[df['Datasets'].str.contains('C_D', na=False)][cols]
cd_df.to_csv('datasets/client_and_driver.csv', index=False)
print(f"Client and Driver (C_D): {len(cd_df)} records")

# Filter for Client Exclusive (CE)
ce_df = df[df['Datasets'].str.contains('CE', na=False)][cols]
ce_df.to_csv('datasets/client_exclusive.csv', index=False)
print(f"Client Exclusive (CE): {len(ce_df)} records")

# Filter for Driver Exclusive (DE)
de_df = df[df['Datasets'].str.contains('DE', na=False)][cols]
de_df.to_csv('datasets/driver_exclusive.csv', index=False)
print(f"Driver Exclusive (DE): {len(de_df)} records")

print("\nDone! Created 3 CSV files with uniprot_id and sequence columns.")

# Calculate max sequence length for each file
cd_max = cd_df['sequence'].str.len().max()
ce_max = ce_df['sequence'].str.len().max()
de_max = de_df['sequence'].str.len().max()

print(f"\nMax sequence lengths:")
print(f"  Client and Driver (C_D): {cd_max}")
print(f"  Client Exclusive (CE): {ce_max}")
print(f"  Driver Exclusive (DE): {de_max}")