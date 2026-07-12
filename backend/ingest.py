import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Configuration: Map the exact frontend dropdown roles to the specific PDFs
LOCAL_QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "textbook_knowledge"

ROLE_DOCS = {
    "AI/ML Engineering Intern": "aiml_book.pdf",
    "Backend Engineering Intern": "backend_book.pdf"
}

def ingest_all_knowledge():
    print("💾 Connecting to Qdrant Vector Engine...")
    client = QdrantClient(path=LOCAL_QDRANT_PATH)

    # Wipe the old development database to ensure a clean, fresh architecture
    if client.collection_exists(COLLECTION_NAME):
        print("🧹 Dropping old collection...")
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print("🧠 Loading HuggingFace Embedding pipeline (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Advanced chunking: overlapping text ensures context isn't lost mid-sentence
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_processed_chunks = []

    # 2. Process each textbook and inject Metadata
    for role_name, pdf_filename in ROLE_DOCS.items():
        print(f"\n📄 Processing tracking pipeline for: {role_name}")
        if not os.path.exists(pdf_filename):
            print(f"⚠️ Skipping: '{pdf_filename}' not found in backend folder.")
            continue
            
        # Load the PDF
        loader = PyPDFLoader(pdf_filename)
        pages = loader.load()
        print(f"   ↳ Loaded {len(pages)} pages successfully.")
        
        # Split into chunks
        chunks = text_splitter.split_documents(pages)
        print(f"   ↳ Generated {len(chunks)} text chunks.")
        
        # METADATA INJECTION: Tag every single chunk with the target role
        for chunk in chunks:
            chunk.metadata["target_role"] = role_name
            
        all_processed_chunks.extend(chunks)

    if not all_processed_chunks:
        print("\n❌ Error: No textbook data chunks were processed. Place the PDFs in the folder.")
        return

    # 3. Commit everything to the Vector Database
    print(f"\n⏳ Embedding and writing {len(all_processed_chunks)} total chunks to Qdrant. This takes a minute...")
    QdrantVectorStore.from_documents(
        documents=all_processed_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=LOCAL_QDRANT_PATH,
    )

    print("\n🚀 Advanced Metadata-Filtered RAG Ingestion Complete!")

if __name__ == "__main__":
    ingest_all_knowledge()