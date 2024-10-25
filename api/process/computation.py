import asyncio
import os
import sys
import time
import traceback
import tracemalloc

import psutil
import redis
from keras.models import load_model

import utils.utils as utils
from phases.data_preparation import DataPreparation
from phases.decision import Decision
from phases.featuresExtraction import FeauturesExtraction
from phases.featuresExtractionRevision import FeaturesExtractionRevision
from phases.lookup import Lookup
from phases.prediction import Prediction
from wrapper.Database import MongoDBWrapper  # MongoDB database wrapper
from wrapper.lamAPI import LamAPI


async def main():
    proc = psutil.Process()
    rss = proc.memory_info().rss  # Resident Set Size (in bytes)
    vms = proc.memory_info().vms  # Virtual Memory Size (in bytes)
    tracemalloc.start()
    start = time.perf_counter()

    pn_neural_path = "./process/ml_models/Linker_PN_100.h5"
    rn_neural_path = "./process/ml_models/Linker_RN_100.h5"
    # rn_neural_path = "./process/ml_models/RN_model_from_scratch_turl-120k-correct-qid-filtered.h5"
    # rn_neural_path = "./process/ml_models/RN_model_from_scratch_turl-120k-no-correct-qid-filtered.h5"

    pn_model = load_model(pn_neural_path)
    rn_model = load_model(rn_neural_path)

    REDIS_ENDPOINT = os.environ["REDIS_ENDPOINT"]
    REDIS_JOB_DB = int(os.environ["REDIS_JOB_DB"])
    LAMAPI_HOST = os.environ["LAMAPI_ENDPOINT"]
    LAMAPI_TOKEN = os.environ["LAMAPI_TOKEN"]

    job_active = redis.Redis(host=REDIS_ENDPOINT, db=REDIS_JOB_DB)

    # Initialize MongoDB wrapper and get collections for different data models
    mongoDBWrapper = MongoDBWrapper()
    log_c = mongoDBWrapper.get_collection("log")
    row_c = mongoDBWrapper.get_collection("row")
    candidate_scored_c = mongoDBWrapper.get_collection("candidateScored")
    cea_c = mongoDBWrapper.get_collection("cea")
    cpa_c = mongoDBWrapper.get_collection("cpa")
    cta_c = mongoDBWrapper.get_collection("cta")
    cea_prelinking_c = mongoDBWrapper.get_collection("ceaPrelinking")

    data = row_c.find_one_and_update({"status": "TODO"}, {"$set": {"status": "DOING"}})

    if data is None:
        print("No data to process", flush=True)
        job_active.set("STOP", "")
        sys.exit(0)

    rows_data = data["rows"]
    column_metadata = data["column"]
    target = data["target"]
    _id = data["_id"]
    dataset_name = data["datasetName"]
    table_name = data["tableName"]
    page = data["page"]
    header = data["header"]
    lamapi_kwargs = data.get("lamapi_kwargs", {"kg": "wikidata", "limit": 50})
    kg_reference = lamapi_kwargs.get("kg", "wikidata")

    # Instantiate the LAMAPI object
    lamAPI = LamAPI(LAMAPI_HOST, LAMAPI_TOKEN, mongoDBWrapper, kg=kg_reference)

    obj_row_update = {"status": "DONE", "time": None}
    dp = DataPreparation(header, rows_data, lamAPI)

    try:
        column_metadata, target = await dp.compute_datatype(column_metadata, target)
        if target["SUBJ"] is not None:
            column_metadata[str(target["SUBJ"])] = "SUBJ"
        obj_row_update["column"] = column_metadata
        obj_row_update["metadata"] = {
            "column": [
                {"idColumn": int(id_col), "tag": column_metadata[id_col]}
                for id_col in column_metadata
            ]
        }
        obj_row_update["target"] = target

        metadata = {
            "datasetName": dataset_name,
            "tableName": table_name,
            "kgReference": kg_reference,
            "page": page,
        }

        collections = {
            "ceaPrelinking": cea_prelinking_c,
            "cea": cea_c,
            "cta": cta_c,
            "cpa": cpa_c,
            "candidateScored": candidate_scored_c,
        }
        dp.rows_normalization()
        l = Lookup(data, lamAPI, target, log_c)
        await l.generate_candidates(lamapi_kwargs=lamapi_kwargs)
        rows = l.get_rows()
        features = await FeauturesExtraction(rows, lamAPI).compute_feautures()
        Prediction(rows, features, pn_model).compute_prediction("rho")
        cea_preliking_data = utils.get_cea_pre_linking_data(metadata, rows)
        revision = FeaturesExtractionRevision(rows)
        features = revision.compute_features()
        Prediction(rows, features, rn_model).compute_prediction("rho'")
        storage = Decision(
            metadata,
            cea_preliking_data,
            rows,
            revision._cta,
            revision._cpa_pair,
            collections,
        )
        storage.store_data()
        end = time.perf_counter()
        execution_time = round(end - start, 2)
        mem_size, mem_peak = tracemalloc.get_traced_memory()
        rss = proc.memory_info().rss - rss
        vms = proc.memory_info().vms - vms
        obj_row_update["time"] = execution_time
        obj_row_update["memory_size"] = mem_size
        obj_row_update["memory_peak"] = mem_peak
        obj_row_update["rss"] = rss
        obj_row_update["vms"] = vms
        obj_row_update["time"] = execution_time
        row_c.update_one({"_id": _id}, {"$set": obj_row_update})
        print("End", flush=True)
    except Exception as e:
        log_c.insert_one(
            {
                "datasetName": dataset_name,
                "tableName": table_name,
                "error": str(e),
                "stackTrace": traceback.format_exc(),
            }
        )


# Run the asyncio event loop
asyncio.run(main())
