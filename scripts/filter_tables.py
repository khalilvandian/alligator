import pandas as pd
import os
from tqdm import tqdm

# Load the dataframe (modify this to read your dataframe, assuming it's named 'df')
df = pd.read_pickle("B:/Projects/alligator/Data/filtered_all_datasets_gt.pkl")

# The folder containing the CSV files
folder_path = 'B:/Projects/Innograph/Training_Data/tables/unfiltered/tables'

# Get the unique table names from column 0 of the dataframe
table_names_to_keep = df[0].unique()

# Get a list of all CSV files in the folder
csv_files = os.listdir(folder_path)

# Initialize a progress bar for the processing of CSV files
with tqdm(total=len(csv_files), desc="Processing Tables") as pbar:
    # Iterate through all the CSV files in the folder
    for csv_file in csv_files:
        table_name = csv_file.replace('.csv', '')  # Assuming table names match the CSV file names without '.csv'
        
        if table_name in table_names_to_keep:
            # Load the table
            table = pd.read_csv(os.path.join(folder_path, csv_file))
            
            # Get the relevant rows from the dataframe for this table
            relevant_rows = df[df[0] == table_name]
            
            # Subtract 1 from the row numbers in the dataframe to account for the header row in the table
            rows_to_keep = relevant_rows[1].values - 1
            
            # Keep only the rows in the table that have a matching row number from column 1 of the dataframe
            table = table.loc[table.index.isin(rows_to_keep)]
            
            # If the resulting table is empty, delete the CSV file, otherwise save the filtered table
            if table.empty:
                os.remove(os.path.join(folder_path, csv_file))
            else:
                table.to_csv(os.path.join(folder_path, csv_file), index=False)
        else:
            # If the table is not in the list of tables to keep, delete the file
            os.remove(os.path.join(folder_path, csv_file))
        
        # Update the progress bar
        pbar.update(1)

print("Process completed.")
