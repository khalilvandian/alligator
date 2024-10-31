import pandas as pd
import os
from tqdm import tqdm

# Load the ground truth dataframe from the pickle file
df = pd.read_pickle("B:/Projects/alligator/Data/filtered_all_datasets_gt.pkl")
## TODO: change after testing
# df = pd.read_csv("B:/Projects/Alligator-2/alligator/Tests/Trainingdata_sampling/gt_test.csv", header=None)
# sort data frame with table name, row and column for making the filtering easier
df = df.sort_values(by=[0, 1, 2])

# The folder containing the original CSV files
folder_path = 'B:/Projects/Innograph/Training_Data/tables/unfiltered/backup/tables'
# ## TODO: change after testing
# folder_path = 'B:/Projects/Alligator-2/alligator/Tests/Trainingdata_sampling/tables'

# The folder to save selected and edited tables
output_folder = 'B:/Projects/Alligator-2/alligator/Training_Data/tables'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get the unique table names from column 0 of the dataframe
table_names_to_keep = df[0].unique()

# Get a list of all CSV files in the original folder
csv_files = os.listdir(folder_path)
csv_files = csv_files[:2000]  # For testing purposes, only process the first 10 files
counter = 0

# Initialize a progress bar for the processing of CSV files
with tqdm(total=len(csv_files), desc="Processing Tables") as pbar:
    # Iterate through all the CSV files in the folder
    for csv_file in csv_files:
        table_name = csv_file.replace('.csv', '')  # Assuming table names match the CSV file names without '.csv'

        ## TODO: Remove this block of code after testing
        flag = False

        if table_name in table_names_to_keep and table_name == "0A1DO6N4":
            # Load the table with headers
            table = pd.read_csv(os.path.join(folder_path, csv_file))
            
            # Get the relevant rows from the dataframe for this table
            relevant_rows = df[df[0] == table_name]

            # Subtract 1 from the row numbers in the dataframe to account for the header row in the table
            rows_to_keep = relevant_rows[1].values - 1

            ## TODO: change to involve column mismatches too
            ## TODO: Remove this block of code after testing
            if set(rows_to_keep) != set(table.index):
                flag = True
                print("=====================================")
                print("relevant rows: ", set(rows_to_keep))
                print("Table Rows", set(table.index))
                counter += 1
                print(f"Processing table: {table_name}")
                
                print("The tableis loaded:")
                print(table.to_string())
                print("\n \n")
                print("The relevant rows are:")
                print(relevant_rows.to_string())

                # compare the row numbers in the dataframe to the row numbers in the table
                print("\n \n")
                # print(f"Row numbers in the dataframe: {set(relevant_rows[1].values)}")
                # print(f"Row numbers in the table: {set(table.index.values)}")
            else:
                continue

            
            # Create a list of all original indices
            original_indices = list(table.index)
            
            # Keep only the rows in the table that have a matching row number from column 1 of the dataframe
            table = table.loc[table.index.isin(rows_to_keep)]
            
            # Create a mapping of old index to new index after rows are removed
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(table.index)}
            
            # Update the row numbers in the ground truth dataframe for the current table
            df.loc[df[0] == table_name, 1] = df.loc[df[0] == table_name, 1].apply(
                lambda x: index_mapping.get(x - 1, None) + 1 if (x - 1) in index_mapping else None)
            
            ## TODO: Remove this block of code after testing
            if flag:
                print("The updated table is:")
                print(table.reset_index().drop(columns=["index"]).to_string())
                print("\n \n")
                print("The index mapping is:")
                print(index_mapping)
                print("\n \n")
                print("The updated ground truth dataframe is:")
                print(df[df[0] == table_name].to_string())
                print("=====================================")


            # Drop any rows in the ground truth dataframe where the mapping failed (those rows were removed)
            df = df.dropna(subset=[1])
            df[1] = df[1].astype(int)  # Ensure the row column is of type int

            # If the resulting table is empty, skip saving it; otherwise, save to the output folder
            if not table.empty:
                # output_path = os.path.join(output_folder, csv_file)
                # table.to_csv(output_path, index=False)
                pass
        else:
            # print(f"\nTable {table_name} is not in the ground truth dataframe. Skipping.\n")
            pass
            
        
        # Update the progress bar
        pbar.update(1)

# Save the updated ground truth dataframe
# gt_save_path = "B:/Projects/Alligator-2/alligator/Training_Data/gt/updated_filtered_all_datasets_gt.pkl"
# os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
# df.to_pickle(gt_save_path)

print(f"counter: {counter}")
print("Process completed.")
