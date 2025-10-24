import pandas as pd
import numpy as np

# --- Configuration ---
INITIAL_FILE_NAME = '/kaggle/input/amazon-ml-dataset-csv/dataset/train.csv' # Placeholder name for the generated input file
NEW_TRAIN_FILE_NAME = '/kaggle/working/train.csv'        # Output file (85% of data)
VAL_FILE_NAME = '/kaggle/working/val.csv'                # Output file (15% of data)
N_ROWS = 75000
SPLIT_RATIO = 0.85
SEED = 42 # For reproducible results

def perform_split(df, ratio, new_train_name, val_name):
    """Splits the DataFrame into training and validation sets and saves them."""
    total_size = len(df)
    train_size = int(total_size * ratio)
    val_size = total_size - train_size

    print(f"\nTotal rows: {total_size}")
    print(f"Calculated 85% train size: {train_size} rows")
    print(f"Calculated 15% val size: {val_size} rows")

    # 1. Randomly sample the rows for the new training set (85%)
    # The 'random_state' ensures the sampling is the same every time the script is run
    train_df = df.sample(n=train_size, random_state=SEED)

    # 2. The validation set (15%) is everything that was NOT sampled for training
    val_df = df.drop(train_df.index)

    # --- Save the results ---
    # Saving without index since 'sample_id' serves as the row identifier
    train_df.to_csv(new_train_name, index=False)
    val_df.to_csv(val_name, index=False)

    print(f"\nSplit successful:")
    print(f"  -> Training data ({len(train_df)} rows) saved to '{new_train_name}'")
    print(f"  -> Validation data ({len(val_df)} rows) saved to '{val_name}'")

# --- Main execution block ---
if __name__ == "__main__":
    try:
        # Step 1: Load the existing file (original_train.csv)
        print(f"Attempting to load data from '{INITIAL_FILE_NAME}'...")
        original_df = pd.read_csv(INITIAL_FILE_NAME)
        print(f"Successfully loaded {len(original_df)} rows.")
    except FileNotFoundError:
        print(f"Error: Required file '{INITIAL_FILE_NAME}' not found.")
        print("Please ensure your original data file is available in the same directory.")
        exit()
    
    # Step 2: Perform the 85-15 split and save the new files
    perform_split(original_df, SPLIT_RATIO, NEW_TRAIN_FILE_NAME, VAL_FILE_NAME)