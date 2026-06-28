import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class Database:
    def __init__(self, name: str, database_path: str = "./chroma", embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.client = chromadb.PersistentClient(path=database_path)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection = self.client.get_or_create_collection(name=name)
        
        self.bm25_index = None
        self.all_chunks = []
        
        self._update_bm25_index()

    def _update_bm25_index(self) -> None:
        all_data = self.collection.get()
        if all_data and all_data.get("documents"):
            self.all_chunks = all_data["documents"]
            tokenized_corpus = [doc.lower().split() for doc in self.all_chunks]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.all_chunks = []
            self.bm25_index = None

    def add(self, text: str, id: str) -> None:
        embedding = self.embedding_model.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[id]
        )
        self._update_bm25_index()
        
    def add_chunks(self, chunks: list[tuple[str, str]], id: str) -> None:
        texts = [chunk[0] for chunk in chunks]
        metadatas = [{"page": chunk[1]} for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"{id}-{i}" for i in range(len(chunks))]
        )
        self._update_bm25_index()

    def search_vector(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        formatted_results = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                formatted_results.append((doc, dist))
                
        return formatted_results

    def search_bm25(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        if not self.bm25_index:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        doc_score_pairs = list(zip(self.all_chunks, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:k]

    def search_hybrid(self, query: str, k: int = 4) -> list[str]:
        vector_results = self.search_vector(query, k=k)
        bm25_results = self.search_bm25(query, k=k)
        
        vector_docs = [doc for doc, _ in vector_results]
        bm25_docs = [doc for doc, _ in bm25_results]
        
        combined_docs = []
        for v_doc, b_doc in zip(vector_docs + [None]*k, bm25_docs + [None]*k):
            if v_doc and v_doc not in combined_docs:
                combined_docs.append(v_doc)
            if b_doc and b_doc not in combined_docs:
                combined_docs.append(b_doc)
            
            if len(combined_docs) >= k:
                break
                
        return combined_docs[:k]