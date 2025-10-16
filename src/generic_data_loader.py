from __future__ import annotations

import csv
import json
import logging
import os
from typing import Generator

from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class GenericDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")
        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must have extension {ext}")

    # -----------------------------
    # 💾 EAGER LOADING (as before)
    # -----------------------------
    def load_custom(
        self,
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        self.check(self.corpus_file, "jsonl")
        self.check(self.query_file, "jsonl")
        self.check(self.qrels_file, "tsv")

        if not self.corpus:
            logger.info("Loading Corpus (eager)...")
            self.corpus = {doc["_id"]: {"text": doc["text"], "title": doc["title"]}
                           for doc in self.iter_corpus()}
            logger.info("Loaded %d Documents.", len(self.corpus))

        if not self.queries:
            logger.info("Loading Queries (eager)...")
            self.queries = {q["_id"]: q["text"] for q in self.iter_queries()}

        if os.path.exists(self.qrels_file):
            logger.info("Loading Qrels (eager)...")
            self.qrels = {}
            for qrel in self.iter_qrels():
                qid, cid, score = qrel
                self.qrels.setdefault(qid, {})[cid] = score

            # Filter queries by available qrels
            self.queries = {qid: self.queries[qid] for qid in self.qrels if qid in self.queries}
            logger.info("Loaded %d Queries with Qrels.", len(self.queries))

        return self.corpus, self.queries, self.qrels

    def load(self, split="test"):
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        return self.load_custom()

    # -----------------------------
    # 🐢 LAZY LOADING METHODS
    # -----------------------------
    def iter_corpus(self) -> Generator[dict, None, None]:
        """Stream corpus.jsonl lazily, yielding one document at a time."""
        self.check(self.corpus_file, "jsonl")
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in fIn:
                data = json.loads(line)
                yield {
                    "_id": data.get("_id"),
                    "text": data.get("text"),
                    "title": data.get("title"),
                }

    def iter_queries(self) -> Generator[dict, None, None]:
        """Stream queries.jsonl lazily, yielding one query at a time."""
        self.check(self.query_file, "jsonl")
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                data = json.loads(line)
                yield {
                    "_id": data.get("_id"),
                    "text": data.get("text"),
                }

    def iter_qrels(self) -> Generator[tuple[str, str, int], None, None]:
        """Stream qrels.tsv lazily, yielding (query_id, corpus_id, score)."""
        self.check(self.qrels_file, "tsv")
        with open(self.qrels_file, encoding="utf-8") as fIn:
            reader = csv.reader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                yield row[0], row[1], int(row[2])

    # -----------------------------
    # 📦 Utility methods (for compatibility)
    # -----------------------------
    def load_corpus(self):
        if not self.corpus:
            self.corpus = {doc["_id"]: {"text": doc["text"], "title": doc["title"]}
                           for doc in self.iter_corpus()}
        return self.corpus
