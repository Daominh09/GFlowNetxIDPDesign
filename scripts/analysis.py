import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt

# The provided CSV data as a string
data_csv = "/Users/minhdao/Documents/University/USF_Academic/UR2PHD/GFlowNetxIDPDesign/datasets/filtered_csat.csv"

def analyze_protein_correlations(csv_data: str):
    """
    Loads protein data, calculates the correlation matrix for 'deescore', 'dg', and 'csat',
    and displays both the numerical matrix and a heatmap visualization.

    Args:
        csv_data: A string containing the protein data in CSV format.
    """
    try:
        # Use io.StringIO to read the string data as if it were a file
        df = pd.read_csv(csv_data)

        # Select the columns for correlation analysis
        correlation_cols = ['deescore', 'dg', 'csat']
        data_for_corr = df[correlation_cols]

        # Calculate the Pearson correlation matrix
        correlation_matrix = data_for_corr.corr(method='pearson')

        # --- Output Numerical Results ---
        print("="*60)
        print("Pearson Correlation Matrix (deescore, dg, csat)")
        print("="*60)
        print(correlation_matrix)
        print("\n" + "="*60)

        # --- Visualization (Heatmap) ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,              # Show the correlation values on the map
            cmap='coolwarm',         # Color map: 'cool' for positive, 'warm' for negative
            fmt=".2f",               # Format annotations to 2 decimal places
            linewidths=.5,           # Lines between cells
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Correlation Heatmap of Protein Features')
        plt.show()

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("Please ensure the CSV data is correctly formatted and contains the 'deescore', 'dg', and 'csat' columns.")

# Execute the analysis function with the provided data
if __name__ == "__main__":
    analyze_protein_correlations(data_csv)