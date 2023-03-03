import os
import sys
import argparse
import logging
from pathlib import Path

from fastrag.stores import PLAIDDocumentStore

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher



def main():
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    
    dataroot = 'data'
    dataset = 'effective'
    datasplit = 'train'

    queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = Queries(path=queries)
    collection = Collection(path=collection)

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    nbits = 2
    index_name = f'{dataset}.{datasplit}.{nbits}bits'

    with Run().context(RunConfig(nranks=5, experiment='notebook')):

        config = ColBERTConfig(
            nbits=nbits, 
            gpus=args.gpus
    
        )
        print("initialize indexer")
        indexer = Indexer(checkpoint="Intel/ColBERT-NQ", config=config)
        print("start indexing")
        indexer.index(name=index_name, collection=collection, overwrite=True)

    with Run().context(RunConfig(experiment='notebook')):
        print("initialize searcher")
        searcher = Searcher(index=index_name)

if __name__ == "__main__":
    main()