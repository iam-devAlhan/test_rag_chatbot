from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from friendli import Friendli
from langchain_core.language_models import LLM
from pydantic import Field, PrivateAttr
import os

load_dotenv()
chroma_api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")

class FriendliLLM(LLM):
    endpoint_id: str = Field(...)  # Pydantic field

    # Private attribute (not validated by Pydantic)
    _client: Friendli = PrivateAttr()

    def __init__(self, endpoint_id: str, **kwargs):
        super().__init__(endpoint_id=endpoint_id, **kwargs)
        self._client = Friendli()  # picks up FRIENDLI_TOKEN from env

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.endpoint_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"endpoint_id": self.endpoint_id}

    @property
    def _llm_type(self):
        return "friendli-llm"

# --- Load environment variables ---

# --- Setup embeddings + Chroma ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="test-collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=chroma_api_key,
    tenant=tenant,
    database=database
)

# --- Use Friendli endpoint ---
llm = FriendliLLM(endpoint_id="depr5042pti8evf")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Prompt ---
template = """You are a helpful assistant. Answer the following question from the context below concisely.

Context:
{context}

Question:
{input}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "input"])

# --- Build chain ---
doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

# --- Run query ---
def get_answer(q):
    answer = retrieval_chain.invoke({"input": q})
    return answer["answer"]
