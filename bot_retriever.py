from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Получаем vectorstore из заранее загруженных файлов
vectorstore = FAISS.load_local("faiss_db_full", embeddings, allow_dangerous_deserialization=True)


# Создаем на его базе ретривер
retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=5,
    score_threshold=None
)