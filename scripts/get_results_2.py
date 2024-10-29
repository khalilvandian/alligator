import argparse
import os
import json
import requests
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch paginated results from Alligator API and save as JSON"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Endpoint to get the results from",
        default="http://localhost:5043",
    )
    parser.add_argument("--dataset_name", type=str, default="github-testset-fixed")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory to save the JSON results",
        default="results",
    )
    args = parser.parse_args()
    args.output_path = os.path.expanduser(args.output_path)

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # List of table names to fetch
    tables_names = ["Github_Testset"]  # Replace with actual table names

    for table_name in tables_names:
        page = 1
        # Initialize the progress bar for page-by-page progress
        with tqdm.tqdm(desc=f"Fetching pages for {table_name}", unit="page") as progress_bar:
            while True:
                # Form the URL with the page parameter
                url = f"http://localhost:5042/dataset/{args.dataset_name}/table/{table_name}?token=alligator_demo_2023&page={page}"
                try:
                    response = requests.get(
                        url,
                        headers={
                            "accept": "application/json",
                            "Content-Type": "application/json",
                        },
                        timeout=300  # Long timeout for handling large data
                    )
                    response.raise_for_status()  # Raise an error if the request failed

                    # Parse the JSON response
                    data = response.json()
                    
                    # Stop if there are no results for this page
                    if not data or len(data) == 0:
                        print(f"No more data on page {page} for {table_name}.")
                        break

                    # Save JSON response to a file per page
                    output_file = os.path.join(args.output_path, f"alligator_annotations_{table_name}_page_{page}.json")
                    with open(output_file, "w") as f:
                        json.dump(data, f)
                    print(f"Saved annotations for {table_name} (Page {page}) to JSON.")

                    # Update progress bar for each page
                    progress_bar.update(1)

                    # Move to the next page
                    page += 1

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for {table_name} on page {page}: {e}")
                    break  # Stop on error
