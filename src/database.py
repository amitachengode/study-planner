import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pdf_processor import generate_chunks_from_pdf

class Database:
    def __init__(self, name: str, database_path: str = "./chroma", embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.client = chromadb.PersistentClient(path=database_path)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection = self.client.get_or_create_collection(name=name)

    def add(self, text: str, id: str) -> None:
        embedding = self.embedding_model.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[id]
        )
        
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

    def search(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        # Safely unpack Chroma's nested results structure
        formatted_results = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                formatted_results.append((doc, dist))
                
        return formatted_results