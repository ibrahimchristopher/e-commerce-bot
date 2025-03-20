"""
vectorstore.py contains the implementation  of the class Vectorstore

The  Vectorstore Class Addresses FAQ and Policy Indexing and Searchable vector database for Agent to consult
    ==> Vectorstore is as a result of embeddings created from OPen AI
"""
###setting up Vectorstore class
import numpy as np

class VectorStoreRetriever:
    """
    A class for setting up and querying a vector store to retrieve relevant sections of policies and FAQs.

    This class stores document embeddings and allows efficient similarity-based retrieval of relevant 
    content using OpenAI embeddings.

    Methods:
    --------
    __init__(docs: list, vectors: list, oai_client):
        Initializes the retriever with documents, their vector representations, and an OpenAI client.

    from_docs(cls, docs, oai_client):
        Class method to create an instance from a list of documents by generating embeddings using OpenAI.

    query(query: str, k: int = 5) -> list[dict]:
        Retrieves the top-k most relevant documents based on similarity to the given query.

        Parameters:
        -----------
        query : str
            The input query for which relevant documents need to be retrieved.
        k : int, optional (default=5)
            The number of top relevant documents to return.

        Returns:
        --------
        list[dict]
            A list of the top-k most relevant documents, each including its similarity score.
    """

    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        # Extract the text content for embedding
        input_texts = [doc["page_content"] for doc in docs]
        
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=input_texts
        )
        
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]
