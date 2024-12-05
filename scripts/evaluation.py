import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def load_ground_truth(gt_file_path):
    """Load and preprocess the ground truth data."""
    gt = pd.read_csv(gt_file_path, header=None)
    gt.columns = ["table_name", "row", "col", "qid"]
    url_regex = re.compile(r"http(s)?\:////www/.wikidata/.org\/(wiki|entity)\/")
    gt["qid"] = gt["qid"].map(lambda x: url_regex.sub("", x))
    return gt

def create_gt_mapping(gt, include_nil):
    """Create mappings for ground truth based on whether to include NIL values."""
    if include_nil:
        return {f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid} for row in gt.itertuples()}, {}
    else:
        gt_mapping = {f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid} for row in gt.itertuples() if row.qid.startswith("Q")}
        gt_mapping_nil = {f"{row.table_name}-{row.row}-{row.col}": {"target": row.qid} for row in gt.itertuples() if row.qid.lower() == "nil"}
        return gt_mapping, gt_mapping_nil

def calculate_metrics(alligator_annotations, gt_mapping, gt_mapping_nil, include_nil, current_table_name, threshold=0.0):
    """Calculate precision, recall, and F1 scores for a specific threshold."""
    fn, fp, tn, tp = 0, 0, 0, 0

    for annotation in alligator_annotations:
        key = f"{current_table_name}-{annotation['idRow']}-{annotation['idColumn']}"

        # Skip if the target is NIL if we are not including NIL values
        if not include_nil and key not in gt_mapping:
            continue

        # Is it a NIL prediction?
        if len(annotation["entity"]) == 0 or annotation["entity"][0]["score"] < threshold:
            predicted_qid = "NIL"
        else:
            predicted_qid = annotation["entity"][0]["id"]

        target = gt_mapping.get(key, {}).get("target", [])

        if predicted_qid == "NIL" and target == "NIL":
            tn += 1
        elif predicted_qid == "NIL" and target != "NIL":
            fn += 1
        elif predicted_qid != "NIL" and (target == "NIL" or predicted_qid != target):
            fp += 1
        elif predicted_qid != "NIL" and target != "NIL" and predicted_qid == target:
            tp += 1
        else:
            raise ValueError("case leftout")

    precision = (tp / (tp + fp)) if tp + fp > 0 else 0
    recall = (tp / (tp + fn)) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def plot_metrics(performance_metrics_df, include_nil):
    """Plot precision, recall, and F1 scores against thresholds."""
    plt.figure(figsize=(10, 6))
    plt.plot(performance_metrics_df["threshold"], performance_metrics_df["precision"], label='Precision')
    plt.plot(performance_metrics_df["threshold"], performance_metrics_df["recall"], label='Recall')
    plt.plot(performance_metrics_df["threshold"], performance_metrics_df["f1"], label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Performance Metrics vs. Threshold ({"Including" if include_nil else "Excluding"} NIL)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_score_distribution(scores, title):
    """Plot a histogram for score distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def extract_scores(alligator_annotations, gt_mapping, current_table_name, include_nil=False):
    """Extract scores for NIL or non-NIL entities."""
    scores = []
    for mention in alligator_annotations:
        key = f"{current_table_name}-{mention['idRow']}-{mention['idColumn']}"
        if include_nil and key in gt_mapping:
            scores.append(mention["entity"][0]["score"] if mention["entity"] else 0)
        elif not include_nil and key in gt_mapping:
            scores.append(mention["entity"][0]["score"] if mention["entity"] else 0)
    return scores

if __name__ == "__main__":
    alligator_annotations_path = "./Results/github testset - previous processes/exampleResponse.json"
    gt_file_path = './gh/gt/cea_gt.csv'
    current_table_name = 'Github_Testset'

    gt = load_ground_truth(gt_file_path)

    with open(alligator_annotations_path) as f:
        alligator_annotations = json.load(f)["semanticAnnotations"]["cea"]

    for include_nil in [False, True]:
        gt_mapping, gt_mapping_nil = create_gt_mapping(gt, include_nil)
        performance_metrics = []

        for threshold in np.arange(0.0, 1.001, 0.01):
            precision, recall, f1 = calculate_metrics(alligator_annotations, gt_mapping, gt_mapping_nil, include_nil, current_table_name, threshold)
            performance_metrics.append([threshold, precision, recall, f1])

        performance_metrics_df = pd.DataFrame(performance_metrics, columns=["threshold", "precision", "recall", "f1"])
        plot_metrics(performance_metrics_df, include_nil)

    # Plot score distributions for non-NIL and NIL entities
    gt_mapping, gt_mapping_nil = create_gt_mapping(gt, include_nil=False)
    nil_scores = extract_scores(alligator_annotations, gt_mapping_nil, current_table_name, include_nil=True)
    non_nil_scores = extract_scores(alligator_annotations, gt_mapping, current_table_name, include_nil=False)
    plot_score_distribution(nil_scores, 'Distribution of NIL Entity Scores')
    plot_score_distribution(non_nil_scores, 'Distribution of Non-NIL Entity Scores')
