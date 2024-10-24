import requests
from urllib.parse import urlparse
import re
import pandas as pd
from tqdm import tqdm  # Progress bar

def extract_qid(input_str):
    """
    Extracts the QID from a Wikidata URL or validates if the input is already a QID.
    
    Args:
        input_str (str): The input string, either a full Wikidata URL, 'NIL', or a QID.
    
    Returns:
        str: The extracted or validated QID (e.g., 'Q3046730'), or 'NIL'.
    
    Raises:
        ValueError: If the input does not contain a valid QID or URL with a QID, showing the invalid value.
    """
    input_str = input_str.strip()

    # Handle NIL cases explicitly
    if input_str.upper() == 'NIL':
        return 'NIL'

    # Check if the input is already a QID using regex (case-insensitive)
    qid_match = re.fullmatch(r'Q\d+', input_str, re.IGNORECASE)
    if qid_match:
        return qid_match.group(0).upper()  # Ensure QID is uppercase

    # If not a QID, attempt to extract from URL
    parsed_url = urlparse(input_str.lower())  # Convert URL to lowercase for validation
    if 'wikidata.org' not in parsed_url.netloc:
        raise ValueError(f"Error: The input '{input_str}' is neither a valid QID nor a valid Wikidata URL.")

    # Use regex to find the QID in the URL path (case-insensitive)
    match = re.search(r'\/(wiki|entity)\/(Q\d+)', parsed_url.path, re.IGNORECASE)
    if match:
        return match.group(2).upper()  # Extract QID and return it as uppercase
    else:
        raise ValueError(f"Error: No valid QID found in the URL '{input_str}'.")

def get_instance_of(qid):
    """
    Retrieves the 'instance of' (P31) properties for a given QID from Wikidata.
    """
    sparql_query = f"""
    SELECT ?instanceOf ?instanceOfLabel WHERE {{
      wd:{qid} wdt:P31 ?instanceOf.
      
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
      }}
    }}
    """

    url = 'https://query.wikidata.org/sparql'
    headers = {
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'WikidataInstanceOfFetcher/1.0 (https://yourdomain.example)'
    }

    response = requests.get(url, params={'query': sparql_query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"SPARQL query failed with status code {response.status_code}: {response.text}")

    data = response.json()
    results = []
    for item in data['results']['bindings']:
        instance_of_id = item['instanceOf']['value'].split('/')[-1]
        instance_of_label = item.get('instanceOfLabel', {}).get('value', 'No label')
        results.append((instance_of_id, instance_of_label))

    return results

def fetch_instance_of(input_str):
    """
    Given a Wikidata URL or QID, fetches the 'instance of' properties for the entity.
    """
    try:
        qid = extract_qid(input_str)
        if qid == 'NIL':
            return 'NIL'  # If the input is 'NIL', return 'NIL' directly
        instances = get_instance_of(qid)
        return instances
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

def batch_fetch_instance_of(qid_series):
    """
    Given a Pandas Series of Wikidata URLs or QIDs, fetches the 'instance of' properties for each entity in batches.

    Args:
        qid_series (pd.Series): A Pandas Series of Wikidata URLs or QIDs.

    Returns:
        pd.Series: A Pandas Series where each entry contains a list of tuples, each tuple being (instance_id, instance_label), or 'NIL'.
    """
    # Extract QIDs from the input series, ensuring 'NIL' cases are handled
    qids = qid_series.apply(lambda x: extract_qid(x))

    # Define the batch size (Wikidata allows a maximum of 50 entities per query)
    batch_size = 50

    def query_batch(batch_qids):
        # Filter out 'NIL' values from the batch before querying
        valid_qids = [qid for qid in batch_qids if qid != 'NIL']

        if not valid_qids:
            return {}  # If no valid QIDs to query, return an empty result

        # Create SPARQL query using VALUES clause for batch of QIDs
        qid_values = " ".join([f"wd:{qid}" for qid in valid_qids])
        sparql_query = f"""
        SELECT ?item ?instanceOf ?instanceOfLabel WHERE {{
          VALUES ?item {{ {qid_values} }}
          ?item wdt:P31 ?instanceOf.
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
          }}
        }}
        """
        url = 'https://query.wikidata.org/sparql'
        headers = {
            'Accept': 'application/sparql-results+json',
            'User-Agent': 'WikidataBatchInstanceFetcher/1.0 (https://yourdomain.example)'
        }

        response = requests.get(url, params={'query': sparql_query}, headers=headers)

        if response.status_code != 200:
            raise Exception(f"SPARQL query failed with status code {response.status_code}: {response.text}")

        data = response.json()
        results = {}
        for item in data['results']['bindings']:
            item_qid = item['item']['value'].split('/')[-1]
            instance_of_id = item['instanceOf']['value'].split('/')[-1]
            instance_of_label = item.get('instanceOfLabel', {}).get('value', 'No label')
            if item_qid not in results:
                results[item_qid] = []
            results[item_qid].append((instance_of_id, instance_of_label))
        return results

    all_results = {}

    # Process in batches and use tqdm for progress bar
    total_batches = (len(qids) + batch_size - 1) // batch_size  # Calculate total number of batches
    for i in tqdm(range(0, len(qids), batch_size), total=total_batches, desc="Fetching batches"):
        batch_qids = qids[i:i+batch_size]
        batch_results = query_batch(batch_qids)
        all_results.update(batch_results)

    # Map the results back to the original QIDs in the input series, handling 'NIL' cases
    result_series = qid_series.apply(lambda x: 'NIL' if extract_qid(x) == 'NIL' else all_results.get(extract_qid(x), []))

    return result_series

def main():
    # Example usage of batch_fetch_instance_of
    qids = pd.Series([
        'https://www.wikidata.org/wiki/Q42',  # Douglas Adams
        'https://www.wikidata.org/wiki/Q1',   # Universe
        'Q3046730',                           # Some random entity
        'NIL',                                # NIL input
        'HTTP://WWW.WIKIDATA.ORG/ENTITY/Q7581076',  # Mixed case input (to be handled correctly)
        'InvalidInput',                       # Invalid input
        # Add more QIDs or URLs here
    ])

    # Fetch 'instance of' properties in batch
    instances_series = batch_fetch_instance_of(qids)

    # Print results
    for input_str, instances in zip(qids, instances_series):
        print(f"\n'Instance of' (P31) for {input_str}:")
        if instances == 'NIL':
            print("NIL")
        elif instances:
            for idx, (instance_id, label) in enumerate(instances, start=1):
                print(f"{idx}. {label} ({instance_id})")
        else:
            print("No 'instance of' properties found or an error occurred.")

if __name__ == "__main__":
    main()
