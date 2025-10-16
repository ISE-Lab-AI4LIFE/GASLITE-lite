import os
from typing import List

from beir import util
# Use our custom GenericDataLoader with lazy loading support
from src.generic_data_loader import GenericDataLoader
import logging
import random

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_dataset(
        dataset_name: str,
        data_split: str = "train",  # or "test"
        data_portion: float = 1.0,  # portion of data to use; 1.0 means all
        random_seed: int = 42,
        embedder_model_name: str = None,  # some models require specific input format (e.g., `e5-base-v2`), we inject it here

        filter_in_qids: List[str] = None,  # filter only these qids
):
    random.seed(random_seed)  # for reproducibility
    filter_concepts = False

    if dataset_name == 'msmarco' and data_split == 'gen_qs':
        # in case of synthetic-queries dataset
        import json
        with open("data/concept-queries-gen.json", encoding='utf8') as f:
            gen_qs_dict = json.load(f)
            concept_name = filter_in_qids[0]
            queries = gen_qs_dict[concept_name]['gen_qs']

        # build hf dataset
        qp_pairs_dataset = Dataset.from_dict({'query': queries, 'query_id': list(range(len(queries)))})

        return None, queries, None, qp_pairs_dataset
    # In case we need to filter only concepts from the training queries
    if data_split.startswith("train-concepts"):
        filter_concepts = True
        data_split = "train"

    # Choose the split
    if data_split == "test":
        data_split = {
            'scifact': 'test',  # ~300
            # MSMARCO's evaluated set is actually the dev-split (e.g., per MTEB: https://github.com/embeddings-benchmark/mteb/blob/aa32a26604dda19ef39b4b0491f064d790cba735/scripts/run_mteb_english.py#L114)
            'msmarco': 'dev',  # ~7K
            'nq': 'test',
            'climate-fever': 'test',
            'trec-covid': 'test',
            'trec-news': 'test',
            'hotpotqa': 'test',
            'quora': 'test',
            'dbpedia-entity': 'test',
            'fiqa': 'test',
        }[dataset_name]
        if "nq" in dataset_name: dataset_name = "nq"
    else:  # "train"
        data_split = {
            'scifact': 'train',
            'msmarco': 'train',
            'fiqa': 'train',
            'nq': 'train',
        }[dataset_name]
        if "nq" in dataset_name: dataset_name = "nq-train"

    # Download dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    out_dir = os.path.join(os.getcwd(), "data")
    data_path = util.download_and_unzip(url, out_dir)

    # MEMORY OPTIMIZATION: Use lazy loading to avoid loading entire corpus
    # Step 1: Load queries and qrels first (lightweight - just metadata)
    logger.info("Loading queries and qrels using lazy loading...")
    loader = GenericDataLoader(data_folder=data_path)
    loader.qrels_file = os.path.join(loader.qrels_folder, data_split + ".tsv")
    
    # Load queries using lazy iterator
    queries = {q["_id"]: q["text"] for q in loader.iter_queries()}
    
    # Load qrels using lazy iterator
    qrels = {}
    for qid, cid, score in loader.iter_qrels():
        qrels.setdefault(qid, {})[cid] = score
    
    # Filter queries by available qrels (same as GenericDataLoader.load())
    queries = {qid: queries[qid] for qid in qrels if qid in queries}
    
    logger.info(f"Initial load: queries={len(queries)}, qrels={len(qrels)}")

    # [OPTIONAL] Filter only relevant concepts from the queries
    if filter_concepts and not filter_in_qids:
        relevant_concept_qids = []
        # iterate over all the yamls that starts with `concept-*` and fetch the concept qids
        import yaml
        concept_filenames = [f for f in os.listdir("config/cover_alg") if f.startswith("concept-")]

        for concept_filename in concept_filenames:
            # fetch qids relevant to the concept from the attack config
            with open(f"config/cover_alg/{concept_filename}", "r") as f:
                concept_config = yaml.safe_load(f)
                relevant_concept_qids.extend(concept_config['concept_qids'])

        # Update `queries` and `qrels` respectively
        queries = {qid: queries[qid] for qid in relevant_concept_qids}
        qrels = {qid: qrel for qid, qrel in qrels.items() if qid in queries.keys()}

        logger.info(f"Filtered concepts: {len(concept_filenames)=}, {len(queries)=}, {len(qrels)=}")

    # [OPTIONAL] Filter only specific qids
    if filter_in_qids:
        filtered_queries = {qid: q for qid, q in queries.items() if qid in filter_in_qids}

        # Update `queries` and `qrels` respectively
        queries = filtered_queries
        qrels = {qid: qrel for qid, qrel in qrels.items() if qid in queries.keys()}

        logger.info(f"Filtered specific qids: {filter_in_qids=}, {len(queries)=}, {len(qrels)=}")

    # [OPTIONAL] Reduce amount of queries according to `data_portion` (by random sampling)
    if data_portion < 1.0:
        _items = list(queries.items())
        n_queries = min(max(int(len(queries) * data_portion), 10_000), len(queries))
        _sampled_items = random.sample(_items, n_queries)

        # Update queries to the sampling
        queries = dict(_sampled_items)

        # Update `qrels` respectively (to contain only the sampled queries)
        qrels = {qid: qrel for i, (qid, qrel) in enumerate(qrels.items()) if qid in queries.keys()}

    # CRITICAL MEMORY OPTIMIZATION: Identify which passages we need BEFORE loading corpus
    relevant_pids = set()
    for qid, pids_dict in qrels.items():
        relevant_pids.update(pids_dict.keys())
    
    logger.info(f"Memory optimization: Need to load {len(relevant_pids)} relevant passages (out of potentially millions)")
    
    # Step 2: Load ONLY the relevant corpus documents using lazy iterator
    # This is the key optimization - we stream corpus.jsonl and only keep what we need
    corpus = {}
    logger.info(f"Streaming corpus and loading only relevant passages...")
    
    for doc in loader.iter_corpus():
        pid = doc.get('_id')
        
        # Only load documents we actually need
        if pid in relevant_pids:
            title = doc.get('title', '') or ''
            text = doc.get('text', '') or ''
            corpus[pid] = {'text': (title + ' ' + text).strip()}
            
            # Early exit if we've loaded all needed documents
            if len(corpus) >= len(relevant_pids):
                logger.info(f"Found all {len(relevant_pids)} needed documents, stopping corpus scan")
                break
    
    logger.info(f"Loaded data: {dataset_name=}, {len(corpus)=}, {len(queries)=}, {len(qrels)=}")
    
    # [OPTIONAL] Format input text for specific models
    # MEMORY OPTIMIZATION: Modify in-place where possible to avoid creating full copies
    if embedder_model_name == "intfloat/e5-base-v2":
        # https://huggingface.co/intfloat/e5-base-v2#faq
        for pid in corpus:
            corpus[pid]['text'] = 'passage: ' + corpus[pid]['text']
        queries = {qid: ('query: ' + text) for qid, text in queries.items()}
    elif embedder_model_name == "Salesforce/SFR-Embedding-Mistral":
        task_desc = 'Given a web search query, retrieve relevant passages that answer the query'
        queries = {qid: f'Instruct: {task_desc}\nQuery: {text}' for qid, text in queries.items()}
    elif embedder_model_name == "nomic-ai/nomic-embed-text-v1":
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage
        for pid in corpus:
            corpus[pid]['text'] = 'search_document: ' + corpus[pid]['text']
        queries = {qid: ('search_query: ' + text) for qid, text in queries.items()}
    elif embedder_model_name in ['BAAI/bge-base-en-v1.5', 'Snowflake/snowflake-arctic-embed-m']:
        # https://huggingface.co/BAAI/bge-base-en-v1.5#frequently-asked-questions
        # https://huggingface.co/Snowflake/snowflake-arctic-embed-m#using-huggingface-transformers
        queries = {qid: ('Represent this sentence for searching relevant passages:' + text)
                   for qid, text in queries.items()}
    elif embedder_model_name == "dunzhang/stella_en_1.5B_v5":
        # https://huggingface.co/dunzhang/stella_en_1.5B_v5
        task_desc = 'Given a web search query, retrieve relevant passages that answer the query.'
        queries = {qid: f'Instruct: {task_desc}\nQuery: {text}' for qid, text in queries.items()}

    # Build HF dataset
    qp_pairs_dataset = _build_hf_dataset(corpus, queries, qrels)

    return corpus, queries, qrels, qp_pairs_dataset


def _build_hf_dataset(corpus, queries, qrels) -> Dataset:
    # Build queries and passages pairs
    q_texts, q_ids, p_ids, p_texts = [], [], [], []
    
    # PERFORMANCE OPTIMIZATION: Move import outside loop and set seed once
    random.seed(0)

    for qid, pids in qrels.items():
        pids: List[str] = list(pid for pid, rank in pids.items() if rank == 1)

        # Get query
        q_text = queries[qid]

        # Get passage; we only need one positive passage per query, other can be found via `qrels`.
        gold_pid = random.choice(pids)
        p_text = corpus[gold_pid]['text']
        q_texts.append(q_text)
        q_ids.append(qid)
        p_texts.append(p_text)
        p_ids.append(gold_pid)

    qp_pairs_dataset = Dataset.from_dict({'query': q_texts, 'query_id': q_ids,
                                          'passage': p_texts, 'pid': p_ids})

    return qp_pairs_dataset
