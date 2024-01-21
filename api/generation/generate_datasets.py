import sys 
sys.path.insert(0, '..')
import os
import pandas as pd
import utils.utils as utils
from tqdm import tqdm
from process.wrapper import mongodb_conn


dataset = {
    "Round1_T2D": {
        "tables": "../data/Dataset/Round1_T2D/tables",
        "cea":"../data/Dataset/Round1_T2D/gt/CEA_Round1_gt_WD.csv",
        "cpa":"../data/Dataset/Round1_T2D/gt/CPA_Round1_gt.csv",
        "cta": "../data/Dataset/Round1_T2D/gt/CTA_Round1_gt.csv"
    },
    "HardTableR2-2021": {
        "tables":"../data/Dataset/HardTablesR2/tables/", 
        "cea":"../data/Dataset/HardTablesR2/gt/cea.csv",
        "cpa":"../data/Dataset/HardTablesR2/gt/cpa.csv",
        "cta": "../data/Dataset/HardTablesR2/gt/cta.csv"
    },
    "HardTableR3-2021": {
        "tables":"../data/Dataset/HardTablesR3/tables/", 
        "cea":"../data/Dataset/HardTablesR3/gt/cea.csv",
        "cpa":"../data/Dataset/HardTablesR3/gt/cpa.csv",
        "cta": "../data/Dataset/HardTablesR3/gt/cta.csv"
    },
    "Round3": {
        "tables": "../data/Dataset/Round3_2019/tables/",
        "cea":"../data/Dataset/Round3_2019/gt/CEA_Round3_gt_WD.csv",
        "cpa":"../data/Dataset/Round3_2019/gt/CPA_Round3_gt.csv",
        "cta": "../data/Dataset/Round3_2019/gt/CTA_Round3_gt.csv"
    },
    "2T-2020": {
        "tables":"../data/Dataset/2T_Round4/tables/", 
        "cea":"../data/Dataset/2T_Round4/gt/cea.csv",
        "cpa": None,
        "cta": "../data/Dataset/2T_Round4/gt/cta.csv"
    },
    "Round4": {
        "tables": "../data/Dataset/Round4_2020/tables/",
        "cea":"../data/Dataset/Round4_2020/gt/cea.csv",
        "cpa":"../data/Dataset/Round4_2020/gt/cpa.csv",
        "cta": "../data/Dataset/Round4_2020/gt/cta.csv"
    }
}


def get_samples(candidates, cea_gt, table_name):
    positive_sample = []
    negative_sample = []
    found_in_gt = False
    n_sample = 0
    TOTAL_SAMPLES_TO_COLLECT = 2
    for candidate in candidates:
        candidate = candidate[1]
        features = {feature:round(candidate["features"][feature], 3) for feature in candidate["features"]}
        #del features["cta"]
        #del features["ctaMax"]
        #del features["cpa"]
        #del features["cpaMax"]
        #features["diff"] = candidates[0][1]["features"]["cea"] - features["cea"]
        #features["diff"] = round(features["diff"], 3)
        del features["cea"]
        features["neural1_score"] = candidate["neural1_score"]
        features["neural1_score_diff"] = candidates[0][1]["neural1_score"] - candidate["neural1_score"]
        
        features["neural2_score"] = candidate["neural2_score"]
        features["neural2_score_diff"] = candidates[0][1]["neural2_score"] - candidate["neural2_score"]
        
        features["logistic1_score"] = candidate["logistic1_score"]
        features["logistic1_score_diff"] = candidates[0][1]["logistic1_score"] - candidate["logistic1_score"]
        
        features["logistic2_score"] = candidate["logistic2_score"]
        features["logistic2_score_diff"] = candidates[0][1]["logistic2_score"] - candidate["logistic2_score"]
        
        if candidate["id"].lower() in cea_gt:
            sample = dict(**{"tableName": table_name}, **features, **{"target": 1})
            found_in_gt = True
            positive_sample.append(sample)
            if n_sample == TOTAL_SAMPLES_TO_COLLECT:
                break
        elif n_sample < TOTAL_SAMPLES_TO_COLLECT:
            sample = dict(**{"tableName": table_name}, **features, **{"target": 0})
            n_sample += 1
            negative_sample.append(sample)
    samples = positive_sample + negative_sample        
    return found_in_gt, samples      


def fill_zero_in_samples(samples):
    new_samples = []
    for sample in samples:
        sample["neural1_score_diff"] = round(sample["neural1_score_diff"], 3)
        sample["neural2_score_diff"] = round(sample["neural2_score_diff"], 3)
        sample["logistic1_score_diff"] = round(sample["logistic1_score_diff"], 3)
        sample["logistic2_score_diff"] = round(sample["logistic2_score_diff"], 3)
        new_samples.append(sample)
        new_sample = sample.copy()
        for key in ["cta", "ctaMax", "cpa", "cpaMax", 
                    "neural1_score", "neural2_score", "logistic1_score", "logistic2_score",
                    "neural1_score_diff", "neural2_score_diff", "logistic1_score_diff", "logistic2_score_diff"]:
            new_sample[key] = 0
        new_samples.append(new_sample)
    return new_samples   



cea_c = mongodb_conn.get_collection("candidateScored")
total_wrong = 0
BUFFER_SIZE = 10000
for id_dataset in dataset:
    tables_path, cea_target_path, cpa_target_path, cta_target_path = list(dataset[id_dataset].values())
    cea_gt = utils.get_cea_ann(cea_target_path)
    name = f"{id_dataset}_100_m1"
    path = f"data/ml_with_type/2/{name}.csv"
    buffer = []
    results = cea_c.find({"datasetName": name})
    total = cea_c.count_documents({"datasetName": name})
   
    for result in tqdm(results, total=total):
        id_table = result["tableName"]
        id_row = result["row"]
        for id_col, item in enumerate(result["candidates"]):
            key = f"{id_table} {id_row} {id_col}"
            if key not in cea_gt or cea_gt.get(key) is None:
                continue
            temp = item
            if len(temp) > 0:
                found_in_gt, samples = get_samples(temp,  cea_gt.get(key, []), id_table)
                if found_in_gt:
                    samples = fill_zero_in_samples(samples)
                    buffer += samples
        
        if len(buffer) >= BUFFER_SIZE:
            pd.DataFrame(buffer).to_csv(path, mode='a', index=False, header=not os.path.exists(path))
            buffer = []
    
    if len(buffer) > 0:
        pd.DataFrame(buffer).to_csv(path, mode='a', index=False, header=not os.path.exists(path))
        buffer = []