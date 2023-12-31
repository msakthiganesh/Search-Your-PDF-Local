from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
from envs import SENTENCE_TRANSFORMER_MODEL_DIR

PERSIST_DIR = "db"
# SENTENCE_TRANSFORMER_MODEL = '<MODEL_NAME>'
# SENTENCE_TRANSFORMER_MODEL_DIR = '<MODEL_DIR>'  # Currently fetched from envs file


def main():
    for root, dirs, files in os.walk("docs"):
        print(root, dir, files)
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL_DIR)

    # Create Vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == '__main__':
    main()
