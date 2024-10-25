import asyncio
import os
import traceback
from typing import List, Dict

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient
from wrapper.URLs import URLs

headers = {"accept": "application/json"}

LAMAPI_TOKEN = os.environ["LAMAPI_TOKEN"]


class LamAPI:
    def __init__(
        self,
        LAMAPI_HOST,
        client_key,
        database,
        response_format="json",
        max_concurrent_requests=50,
        kg: str = "wikidata",
    ) -> None:
        self.format = response_format
        self.database = database
        base_url = LAMAPI_HOST
        self._url = URLs(base_url, response_format=response_format)
        self.client_key = client_key
        self.kg = kg
        # Initialize the semaphore with the max_concurrent_requests limit
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def __to_format(self, response):
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            if self.format == "json":
                result_json = await response.json()
                for kg in ["wikidata", "dbpedia", "crunchbase"]:
                    if result_json and kg in result_json:
                        return result_json[kg]
                if result_json is None:
                    result_json = {}
                return result_json  # If none of the keys are found, return the original JSON data
            else:
                raise Exception("Sorry, Invalid format!")

        return {}

    async def __submit_get(self, url, params):
        try:
            retry_options = ExponentialRetry(
                attempts=3, start_timeout=3, max_timeout=10
            )
            timeout = aiohttp.ClientTimeout(total=1000)  # Adjusted timeout
            async with self.semaphore:
                async with RetryClient(
                    connector=aiohttp.TCPConnector(ssl=False),
                    retry_options=retry_options,
                ) as session:
                    async with session.get(
                        url, headers=headers, params=params, timeout=timeout
                    ) as response:
                        return await self.__to_format(response)
        except Exception as e:
            self.__log_error("GET", url, params, str(e))
            return {"error": str(e)}  # Return a structured error message.

    async def __submit_post(self, url, params, json_data):
        try:
            retry_options = ExponentialRetry(
                attempts=3, start_timeout=3, max_timeout=10
            )
            timeout = aiohttp.ClientTimeout(total=1000)  # Adjusted timeout
            async with self.semaphore:
                async with RetryClient(
                    connector=aiohttp.TCPConnector(ssl=False),
                    retry_options=retry_options,
                ) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        params=params,
                        json=json_data,
                        timeout=timeout,
                    ) as response:
                        return await self.__to_format(response)
        except Exception as e:
            self.__log_error("POST", url, params, str(e), json_data)
            return {"error": str(e)}  # Return a structured error message.

    def __log_error(self, method, url, params, error_message, json_data=None):
        # Use a generic or specific error type based on the exception.
        error_type = "timeout" if "TimeoutError" in error_message else "generic"
        traceback_info = traceback.format_exc()

        self.database.get_collection("log").insert_one(
            {
                "type": error_type,
                "method": method,
                "url": url,
                "params": params,
                "json_data": json_data,
                "error_message": error_message,
                "stack_trace": traceback_info,
            }
        )

    async def literal_recognizer(self, column):
        json_data = {"json": column}
        params = {"token": self.client_key}
        result = await self.__submit_post(
            self._url.literal_recognizer_url(), params, json_data
        )
        freq_data = {}
        for cell in result:
            item = result[cell]
            if (
                item["datatype"] == "STRING"
                and item["datatype"] == item["classification"]
            ):
                datatype = "ENTITY"
            else:
                datatype = item["classification"]
            if datatype not in freq_data:
                freq_data[datatype] = 0
            freq_data[datatype] += 1

        return freq_data

    async def column_analysis(
        self, columns: List[List[str]], model_type: str = "fast"
    ) -> List[List[str]]:
        # The input is a list of lists, where every list is a column in the table
        json_data = {"json": [columns]}
        params = {"token": self.client_key, model_type: model_type}

        # The resutls are a list of dictionaries, where:
        # Every dictionary has a key in the form `table_idx`, where `idx` ranges in the number of tables processed
        # For every `table_idx`, the value is a dictionary with the following keys:
        # - `column_idx`: The index of the column in the table, with the following keys:
        #   - `index_column`: The datatype of the column
        #   - `tag`: The tag assigned to the column (LIT/NE)
        #   - `datatype`: The datatype of the column
        #   - `classification`: The classification of the column
        #   - `probabilities`: a dictionary containing the probabilities of every datatype returned
        response: List[Dict[str, Dict[str, Dict[str, str]]]] = await self.__submit_post(
            self._url.column_analysis_url(), params, json_data
        )
        assert len(response) == 1, "The response should contain only one table"
        return response[0]["table_1"]

    async def labels(self, entities):
        params = {"token": self.client_key, "kg": self.kg}
        json_data = {"json": entities}
        return await self.__submit_post(self._url.entities_labels(), params, json_data)

    async def objects(self, entities):
        params = {"token": self.client_key, "kg": self.kg}
        json_data = {"json": entities}
        return await self.__submit_post(
            self._url.entities_objects_url(), params, json_data
        )

    async def predicates(self, entities):
        params = {"token": self.client_key, "kg": self.kg}
        json_data = {"json": entities}
        return await self.__submit_post(
            self._url.entities_predicates_url(), params, json_data
        )

    async def types(self, entities):
        params = {"token": self.client_key, "kg": self.kg}
        json_data = {"json": entities}
        return await self.__submit_post(
            self._url.entities_types_url(), params, json_data
        )

    async def literals(self, entities):
        params = {"token": self.client_key, "kg": self.kg}
        json_data = {"json": entities}
        return await self.__submit_post(
            self._url.entities_literals_url(), params, json_data
        )

    async def lookup(
        self, string, ids=None, lamapi_kwargs={"kg": "wikidata", "limit": 50}
    ):
        # Convert boolean values to strings
        if isinstance(ids, list):
            ids_str = " ".join(ids)
        elif ids is None:
            ids_str = ""
        elif isinstance(ids, str):
            ids_str = ids
        else:
            raise ValueError("ids must be a list, a string, or None")

        params = {
            "token": LAMAPI_TOKEN,
            "name": string,
            "ids": ids_str,
        }
        params.update(lamapi_kwargs)
        for k, v in params.items():
            if isinstance(v, bool):
                params[k] = str(v).lower()

        return await self.__submit_get(self._url.lookup_url(), params)
