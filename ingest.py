from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

# Store embeddings in temp directory (auto-cleared on Hugging Face restart)
INDEX_DIR = Path("/tmp/vector_data")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# load a text/markdown doc (example - replace with uploaded file path)
loader = TextLoader("docs/my_doc.txt")
docs = loader.load()

# split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# create embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# save ONLY to /tmp, never to repo
vectorstore.save_local(str(INDEX_DIR / "vector_index"))
