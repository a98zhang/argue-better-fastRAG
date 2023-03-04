import os
import sys
import argparse
import logging
from pathlib import Path
import random
import pandas as pd

from fastrag.stores import PLAIDDocumentStore
from fastrag.retrievers.colbert import ColBERTRetriever

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
    
    queries = pd.read_csv(queries_path, sep='\t', header=None)
    sampled_q = queries[1][random.randint(0, len(queries))]
    print(sampled_q)

    print('****** Retrieve!')
    retriever = ColBERTRetriever(store)
    res = retriever.retrieve(sampled_q, 3)
   
    for r in res:
        print(r)



if __name__ == "__main__":
    main()
