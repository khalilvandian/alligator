import json
import pandas as pd
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    alligatior_annotations_path = "B:/Projects/alligator/Data/exampleResponse.json"
    gt_file_path = 'B:/Projects/alligator/gh/gt/cea_gt.csv'

    include_nil = False
    threshold = 0.2

    # Load Ground Truth for Correct QIDs
    gt = pd.read_csv(gt_file_path, header=None)
    gt.columns = ["table_name", "row", "col", "qid"]
    tables_names = gt["table_name"].unique().tolist()
    url_regex = re.compile(r"http(s)?\:////www/.wikidata/.org\/(wiki|entity)\/")
    gt["qid"] = gt["qid"].map(lambda x: url_regex.sub("", x))

    # Load Alligator Annotations
    with open(alligatior_annotations_path) as f:
        alligator_annotations = json.load(f)
        alligator_annotations = alligator_annotations["semanticAnnotations"]["cea"]

    for include_nil in [False, True]:
        # Create Mapping for Ground Truth
        if not include_nil:
            gt_mapping = {
                f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid}
                for row in gt.itertuples()
                if row.qid.startswith("Q")
            }
            gt_mapping_nil = {
                f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid}
                for row in gt.itertuples()
                if row.qid.lower() == "nil"
            }
        else:
            gt_mapping = {f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid} for row in gt.itertuples()}
            gt_mapping_nil = {}   


        # Set initial parameters
        performance_metrics = []
        current_table = None
        current_table_name = 'Github_Testset'

        # Calculate Performance Metrics for Different Thresholds
        for threshold in np.arange(0.0, 1.001, 0.001):
            tp = 0
            all_gt = len(gt) - len(gt_mapping_nil)
            all_predicted = 0

            for annotation in alligator_annotations:
                key = "{}-{}-{}".format(current_table_name, annotation["idRow"], annotation["idColumn"])
                # if key in gt_mapping_nil:
                #     continue
                if key not in gt_mapping:
                    continue
                predicted_qid = ""
                if len(annotation["entity"]) > 0:
                    if annotation["entity"][0]["score"] >= threshold:
                        all_predicted += 1
                        predicted_qid = annotation["entity"][0]["id"]
                if predicted_qid != "" and predicted_qid in gt_mapping[key]["target"]:
                    tp += 1

            precision = tp / all_predicted if all_predicted > 0 else 0
            recall = tp / all_gt if all_gt > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            performance_metrics.append([threshold, precision, recall, f1])

            # if threshold*10 % 1 == 0:
            #     print("Number of mentions to be linked:", all_gt)
            #     print("Number of mentions linked:", all_predicted)
            #     print("Precision: {:.4f}".format(precision))
            #     print("Recall: {:.4f}".format(recall))
            #     print("F1: {:.4f}".format(f1))
            #     print("Threshold: ", threshold)
            #     print("=====================================")

        # Save Performance Metrics to a DataFrame
        performance_metrics_df = pd.DataFrame(performance_metrics, columns=["threshold", "precision", "recall", "f1"])

        # Visualize Performance Metrics vs. Threshold
        thresholds = performance_metrics_df["threshold"]
        precision = performance_metrics_df['precision']
        recall = performance_metrics_df['recall']
        f1_score = performance_metrics_df['f1']

        # Plotting
        plt.figure(figsize=(10, 6))

        plt.plot(thresholds, precision, label='Precision', color='blue')
        plt.plot(thresholds, recall, label='Recall', color='green')
        plt.plot(thresholds, f1_score, label='F1 Score', color='red')

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        if include_nil:
            plt.title('Precision, Recall, and F1 Score vs. Threshold (Including NIL)')
        else:
            plt.title('Precision, Recall, and F1 Score vs. Threshold (Excluding NIL)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Create Mapping for Ground Truth
    gt_mapping = {
        f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid}
        for row in gt.itertuples()
        if row.qid.startswith("Q")
    }
    gt_mapping_nil = {
        f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid}
        for row in gt.itertuples()
        if row.qid.lower() == "nil"
    }

    # Score Distribution for NIL Entities
    NIL_entity_Scores = []

    # extract top score for each mention
    for mention in alligator_annotations:
        key = "{}-{}-{}".format(current_table_name, mention["idRow"], mention["idColumn"])
        if key in gt_mapping_nil:
            NIL_entity_Scores.append(mention["entity"][0]["score"] if len(mention["entity"]) > 0 else 0)

    # Plotting the histogram for the 'Score' column
    plt.figure(figsize=(10, 6))
    plt.hist(NIL_entity_Scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of NIL Entity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Score Distribution for Non-NIL Entities
    NonNil_entity_Scores = []

    # extract top score for each mention
    for mention in alligator_annotations:
        predicted_qid = ""
        key = "{}-{}-{}".format(current_table_name, mention["idRow"], mention["idColumn"])

        if key in gt_mapping:
            if len(mention["entity"]) > 0:
                NonNil_entity_Scores.append(mention["entity"][0]["score"])
            else:
                NonNil_entity_Scores.append(0)

    # Plotting the histogram for the 'Score' column
    # Plotting the histogram for the 'Score' column
    plt.figure(figsize=(10, 6))
    plt.hist(NonNil_entity_Scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Non NIL Entity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()