from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # популярная модель
    model_kwargs={'device': 'cpu'},  # если используете GPU
    encode_kwargs={'normalize_embeddings': True}  # нормализация эмбендингов
)
# Укажите модель векторов или другие параметры, если требуется  
vectorstore = FAISS.load_local("faiss_db_full", embeddings, allow_dangerous_deserialization=True)

# Теперь вы можете использовать vectorstore для поиска или взаимодействия

retriever = vectorstore.as_retriever(
    search_type="similarity",  # тип поиска похожих документов
    k=5,  # количество возвращаемых документов (Default: 4)
    score_threshold=None,  # минимальный порог для поиска "similarity_score_threshold"
)