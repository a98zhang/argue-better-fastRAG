import os
from tqdm import tqdm
import argparse
import pandas as pd
import json

from fastrag.stores import PLAIDDocumentStore
from fastrag.retrievers.colbert import ColBERTRetriever
from haystack.nodes import SentenceTransformersRanker
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate
from haystack.nodes import Shaper
from fastrag.readers import T5Reader
from fastrag.readers.FiD import FiDReader
from fastrag.utils import get_timing_from_pipeline
from haystack import Pipeline

def jsonify(res, reader=1, custom=0, k=1, j=3):

    output = {
        'query': res['query'],
        'ans': dict(),
        'docs': dict()
    }
        
    if custom:
        output['ans']["0"] = {
            'answer': res['results'],
            'score': '', 
            'context': ''
        }

    if reader & (not custom):
        ans = res['answers'][:k]
        for m, a in enumerate(ans):
            output['ans'][m] = {
                'answer': a.answer,
                'score': a.score,
                'context': a.context
            }

    docs = res['documents'][:j]
    if custom:
        docs = res['texts'][:j]

    for n, d in enumerate(docs):
        output['docs'][n] = {
            'content': d.content,
            'id': d.id,
            'score': d.score,
            'meta': d.meta 
        }

    return output


def main():
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--generative", type=int, default=1)
    parser.add_argument("--custom", type=int, default=1)
    args = parser.parse_args()
    
    dataroot = 'data'
    dataset = 'effective'
    datasplit = 'train'

    #queries_path = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    queries_path = os.path.join(dataroot, dataset, datasplit, 'questions.search.test.tsv')
    collection_path = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = pd.read_csv(queries_path, sep='\t', header=None)

    nbits = 2
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    checkpoint_path = "Intel/ColBERT-NQ"
    if args.index:
        create = True
        checkpoint_path = "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        index_name = f'{dataset}.{datasplit}.{nbits}bits.msmarco'
    else:
        create = False

    #---------------------#
    #  create components  #
    #---------------------#


    store = PLAIDDocumentStore(
        index_path=index_name,
        checkpoint_path=checkpoint_path,
        collection_path=collection_path,
        create=create,
        nbits=nbits,
        gpus=args.gpus,
        ranks=args.ranks,
        doc_maxlen=4099,
        query_maxlen=4099,
        kmeans_niters=4,
    )
    
    retriever = ColBERTRetriever(
        store, 
        top_k=100 if args.generative else 10
    )
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", 
        top_k=10
    )

    reader = T5Reader(
        model_name_or_path="google/flan-t5-base", 
        input_converter_tokenizer_max_len=16000,  
        min_length=100, 
        max_length=512, 
        num_beams=4, 
        top_k=1, 
        use_gpu=True
    )

    # custom reader 
    example_generator = PromptTemplate(
        name="eg",
        prompt_text="""Give a better example for this ineffective argument. The example generated should reference similar effective arguments 
               \n\n Similar effective arguments: $texts \n\nIneffective argument:$query \n\n Answer:"""
    )
    prompt_node = PromptNode(model_name_or_path="google/flan-t5-base", default_prompt_template=example_generator, max_length=1000)

    # shaper
    shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["texts"])

    # reader = FiDReader(
    #     input_converter_tokenizer_max_len=250,
    #     max_length=20,
    #     model_name_or_path="path/to/fid",
    #     use_gpu=True
    # )

    #--------------------#
    #   build pipeline   #
    #--------------------#

    p = Pipeline()

    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    if args.generative:
        p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
        if args.custom:
            p.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
            p.add_node(component=prompt_node, name="prompt_node", inputs=["Shaper"])
        else:
            p.add_node(component=reader, name="Reader", inputs=["Reranker"])


    #------------------#
    #   run pipeline   #
    #------------------#

    # store top 3 retrieved docs
    results = dict()
    tmstp = pd.Timestamp.now()  
    for i in tqdm(range(len(queries))):
        res = p.run(query=queries[1][i])
        print(res)
        results[i] = jsonify(res, reader=args.generative, custom=args.custom)
    
    # output results into json file
    json_output = json.dumps(results, indent=4) 
    with open(f"data/effective/res_t5_{tmstp}.json", "w") as outfile:
        outfile.write(json_output)

if __name__ == "__main__":
    main()
