"""
Milvus Lite memory backend using Remote Ollama embeddings.
Replaces the original MilvusMemory (which requires local sentence-transformers + Docker Milvus).
"""

import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from time import strftime, localtime
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

from ollama_client import OllamaClient

FIXED_SUBTRACT = 1721761000
EMBEDDING_DIM = 768  # nomic-embed-text output dimension


@dataclass
class MemoryItem:
    caption: str
    time: float
    position: list
    theta: float
    image_path: str = ""

    def __post_init__(self):
        if self.caption is None:
            self.caption = ""


class LiteMemory:

    def __init__(
        self,
        db_path: str = "./remembr_memory.db",
        collection_name: str = "robot_memory",
        ollama_host: str = "192.168.31.63",
        ollama_port: int = 11434,
        embed_model: str = "nomic-embed-text",
        time_offset: float = FIXED_SUBTRACT,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.time_offset = time_offset
        self.working_memory = []

        self.ollama = OllamaClient(host=ollama_host, port=ollama_port)
        self.client = MilvusClient(db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            print(f"[LiteMemory] Using existing collection: {self.collection_name}")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="position", dtype=DataType.FLOAT_VECTOR, dim=3),
            FieldSchema(name="theta", dtype=DataType.FLOAT),
            FieldSchema(name="time", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
        ]
        schema = CollectionSchema(fields=fields)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="text_embedding", index_type="FLAT", metric_type="L2")
        index_params.add_index(field_name="position", index_type="FLAT", metric_type="L2")
        index_params.add_index(field_name="time", index_type="FLAT", metric_type="L2")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[LiteMemory] Created collection: {self.collection_name}")

    def insert(self, item: MemoryItem):
        embedding = self.ollama.embed(item.caption, model=self.embed_model)
        data = {
            "id": str(time.time()),
            "text_embedding": embedding,
            "position": list(item.position),
            "theta": float(item.theta),
            "time": [float(item.time) - self.time_offset, 0.0],
            "caption": item.caption,
            "image_path": item.image_path or "",
        }
        self.client.insert(collection_name=self.collection_name, data=[data])

    def search_by_text(self, query: str, limit: int = 5) -> str:
        embedding = self.ollama.embed(query, model=self.embed_model)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="text_embedding",
            limit=limit * 6,
            output_fields=["caption", "position", "time", "theta", "image_path"],
        )
        deduped = self._deduplicate_by_position(results[0], limit)
        self.working_memory.extend(deduped)
        return self._format_results(deduped)

    def search_by_position(self, query: tuple, limit: int = 5) -> str:
        pos = list(np.array(query).astype(float))
        if len(pos) == 2:
            pos.append(0.0)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[pos],
            anns_field="position",
            limit=limit,
            output_fields=["caption", "position", "time", "theta", "image_path"],
        )
        self.working_memory.extend(results[0])
        return self._format_results(results[0])

    def search_by_time(self, hms_time: str, limit: int = 5) -> str:
        import datetime

        hms_time = hms_time.strip()
        t = localtime(self.time_offset)
        mdy_date = strftime("%m/%d/%Y", t)
        template = "%m/%d/%Y %H:%M:%S"

        try:
            datetime.datetime.strptime(hms_time, template)
            full_time = hms_time
        except ValueError:
            full_time = mdy_date + " " + hms_time

        query_time = time.mktime(
            datetime.datetime.strptime(full_time, template).timetuple()
        ) - self.time_offset

        results = self.client.search(
            collection_name=self.collection_name,
            data=[[query_time, 0.0]],
            anns_field="time",
            limit=limit,
            output_fields=["caption", "position", "time", "theta", "image_path"],
        )
        self.working_memory.extend(results[0])
        return self._format_results(results[0])

    def _deduplicate_by_position(self, results: list, limit: int, radius: float = 0.5) -> list:
        """Keep only the best-scored result per spatial cluster (within radius meters)."""
        kept = []
        for r in results:
            pos = np.array(r["entity"]["position"][:2])
            too_close = False
            for k in kept:
                kpos = np.array(k["entity"]["position"][:2])
                if np.linalg.norm(pos - kpos) < radius:
                    too_close = True
                    break
            if not too_close:
                kept.append(r)
            if len(kept) >= limit:
                break
        return kept

    def _format_results(self, results: list) -> str:
        out = ""
        for r in results:
            entity = r["entity"]
            t_val = entity["time"]
            if isinstance(t_val, list) and len(t_val) == 2:
                t_val = t_val[0]
            t_val += self.time_offset
            t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(t_val))
            pos = np.array(entity["position"]).round(3).tolist()
            out += (
                f"At time={t_str}, the robot was at position {pos}. "
                f"The robot saw: {entity['caption']}\n\n"
            )
        return out

    def reset(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"[LiteMemory] Dropped collection: {self.collection_name}")
        self._ensure_collection()
        self.working_memory = []

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)

    def close(self):
        self.client.close()
