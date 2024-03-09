import gradio
import os
import jsonpickle

store_file_path = "my_faiss_index.faiss"
store_config_file_path = "my_faiss_index.json"
document_external_location = "https://github.com/OldManUmby/DND.SRD.Wiki/archive/refs/heads/master.zip"
# document_external_location = "https://github.com/koganei/fLLaMingo/archive/refs/heads/main.zip"

"""Setting logging to info"""

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

"""Set Document Store to be In Memory"""

# from haystack.document_stores import InMemoryDocumentStore

# document_store = InMemoryDocumentStore(use_bm25=True)

from haystack.document_stores import FAISSDocumentStore

should_process_files = False

if os.path.exists(store_file_path):
    print("loading path")
    document_store = FAISSDocumentStore(faiss_index_path=store_file_path, faiss_config_path=store_config_file_path)
else:
    document_store = FAISSDocumentStore()
    should_process_files = True

"""Load Data"""

if should_process_files:
    print("processing files")
    from haystack.utils import fetch_archive_from_http

    doc_dir = "data/suitecrm_docs"

    # fetch_archive_from_http(
    #     url=document_external_location,
    #     output_dir=doc_dir
    # )

    """Add Data to the DocumentStore"""

    from haystack.pipelines.standard_pipelines import TextIndexingPipeline

    files_to_index = []

    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                files_to_index.append(file_path)

    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)


"""Initialize the retriever

"""

# from haystack.nodes import BM25Retriever

# retriever = BM25Retriever(document_store=document_store)

from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
)

if should_process_files:
    # Important:
    # Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation.
    # While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
    # At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
    document_store.update_embeddings(retriever)

    print("saved to" + store_file_path)
    document_store.save(index_path=store_file_path)

"""Initialize the reader"""

#  from haystack.nodes import FARMReader

# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

"""Create pipeline to hook up retriever and reader"""

from haystack.nodes import PromptNode
from haystack.nodes.prompt import PromptTemplate
from ExtractiveQAWithPromptNodePipeline import ExtractiveQAWithPromptNodePipeline

pn = PromptNode(
        "gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
        max_length=256,
        default_prompt_template=PromptTemplate(
            name="question-answering-with-document-scores",
            prompt_text="The following is a question about SuiteCRM. Please answer the following question using the paragraphs below from the documentation in Markdown format.\n"
            "An answer should be two paragraphs long and should quote the novel to support the argument.\n"
            "---\n"
            "Paragraphs:\n{join(documents)}\n"
            "---\n"
            "Question: {query}\n\n"
            "Instructions: Consider all the paragraphs above and their corresponding scores to generate "
            "the answer. While a single paragraph may have a high score, it's important to consider all "
            "paragraphs for the same answer candidate to answer accurately.\n\n"
            "After having considered all possibilities, the final answer is:\n",
        ),
    )

pipe = ExtractiveQAWithPromptNodePipeline(
    retriever,
    prompt_node=pn
    )

def my_inference_function(question):


    prediction = pipe.run(
        query=question,
        params={
            "Retriever": {"top_k": 10}
        }
    )

    # json_prediction = jsonpickle.encode(prediction, unpicklable=False)

    return prediction



# gradio_interface = gradio.Interface(
#   fn = my_inference_function,
#   inputs = "text",
#   outputs = "json",
# )
# gradio_interface.launch()


result = my_inference_function("What is a role?")
print(result["results"])
print("\n\n====================\n\n")
print(jsonpickle.encode(result, unpicklable=False))