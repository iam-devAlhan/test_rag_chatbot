from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
chroma_api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")
hf_token = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="test-collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=chroma_api_key,
    tenant=tenant,
    database=database
)

llm = HuggingFaceEndpoint(
    endpoint_url="https://api.friendli.ai/dedicated/depr5042pti8evf",
    repo_id="mistralai/MistralAI-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    top_k=10,
    top_p=0.95,
    temperature=0.7,
    repetition_penalty=1.03,
    max_new_tokens=512
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """You are a helpful assistant, Answer the following question from the context below concisely
context:
{context}

question:
{question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

doc_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

q = "What role has AI done in leveraging technology?"

answer = retrieval_chain.invoke(input=q)
print(answer)