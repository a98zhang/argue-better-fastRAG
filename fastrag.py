import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import json

from fastrag.stores import PLAIDDocumentStore
from fastrag.retrievers.colbert import ColBERTRetriever

from haystack import Pipeline



def main():
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    
    dataroot = 'data'
    dataset = 'effective'
    datasplit = 'train'

    queries_path = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection_path = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = pd.read_csv(queries_path, sep='\t', header=None)

    nbits = 2
    create = True if args.index else False
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    
    store = PLAIDDocumentStore(
        index_path=index_name,
        checkpoint_path="Intel/ColBERT-NQ",
        collection_path=collection_path,
        create=create,
        nbits=nbits,
        gpus=args.gpus,
        ranks=args.ranks,
        doc_maxlen=120,
        query_maxlen=60,
        kmeans_niters=4,
    )
    
    p = Pipeline()
    retriever = ColBERTRetriever(store)
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])

    res = dict()
    for i in range(len(queries)):
        q = queries[1][i]
        d = p.run(query=q)
        res[q] = d["documents"][:3]
    
    with open("data/effective/res.json", "w") as outfile:
        json.dump(res, outfile)
   

    #from haystack.nodes import BM25Retriever, SentenceTransformersRanker
    #from fastrag.readers import T5Reader

    #retriever = BM25Retriever(document_store=document_store, top_k=100)
    #reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=10)
    #reader = T5Reader(model_name_or_path="google/flan-t5-base", input_converter_mode="summarization", input_converter_tokenizer_max_len=16300,  min_length=10, max_length=100, num_beams=4, top_k=1, use_gpu=False)


if __name__ == "__main__":
    main()
