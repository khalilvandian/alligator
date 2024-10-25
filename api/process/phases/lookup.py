import asyncio
import traceback
from typing import List

from model.row import Row


class Lookup:
    def __init__(self, data: object, lamAPI, target, log_c):
        self._header = data.get("header", [])
        self._dataset_name = data["datasetName"]
        self._table_name = data["tableName"]
        self._types = data.get("types", {})
        self._lamAPI = lamAPI
        self._target = target
        self._log_c = log_c
        self._rows_data = data["rows"]
        self._rows = []
        self._cache = {}

    async def generate_candidates(self, lamapi_kwargs={"kg": "wikidata", "limit": 50}):
        tasks = []
        for row in self._rows_data:
            tasks.append(
                asyncio.create_task(
                    self._build_row(
                        row["data"],
                        row["idRow"],
                        row.get("ids", None),
                        lamapi_kwargs=lamapi_kwargs,
                    )
                )
            )
        results = await asyncio.gather(*tasks)
        for row in results:
            self._rows.append(row)

    async def _build_row(
        self, cells, id_row, ids=None, lamapi_kwargs={"kg": "wikidata", "limit": 50}
    ):
        row = Row(id_row, len(cells))
        cells_as_strings = [str(cell) for cell in cells]
        row_text = " ".join(cells_as_strings)
        for i, cell in enumerate(cells):
            if i in self._target["NE"]:
                qid = ids[i] if ids is not None else None
                types = self._types.get(str(i))

                if cell in self._cache:
                    candidates = self._cache.get(cell, [])
                else:
                    candidates = await self._get_candidates(
                        cell, id_row, types, qid, lamapi_kwargs=lamapi_kwargs
                    )
                    self._cache[cell] = candidates
                is_subject = i == self._target["SUBJ"]
                row.add_ne_cell(cell, row_text, candidates, i, is_subject, qid=qid)
            elif i in self._target["LIT"]:
                row.add_lit_cell(cell, i, self._target["LIT_DATATYPE"][str(i)])
            else:
                row.add_notag_cell(cell, i)
        return row

    async def _get_candidates(
        self,
        cell,
        id_row,
        types,
        qid=None,
        lamapi_kwargs={"kg": "wikidata", "limit": 50},
    ):
        candidates = []
        try:
            candidates = await self._lamAPI.lookup(
                cell, ids=qid, lamapi_kwargs=lamapi_kwargs
            )
        except Exception as e:
            self._log_c.insert_one(
                {
                    "datasetName": self._dataset_name,
                    "tableName": self._table_name,
                    "idRow": id_row,
                    "cell": cell,
                    "types": types,
                    "error": str(e),
                    "stackTrace": traceback.format_exc(),
                    "result": candidates,
                }
            )
        return candidates

    def get_rows(self) -> List[Row]:
        return self._rows
