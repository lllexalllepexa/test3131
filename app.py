import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import pysrt
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time
import shutil

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.absolute()

@dataclass
class AppConfig:
    DB_PATH: str = str(SCRIPT_DIR / "chroma_db")
    COLLECTION_NAME: str = "anime_transcripts"
    
    MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # -----------------------
    
    SUBS_FOLDER_NAME: str = "aot_subs"
    BATCH_SIZE: int = 2000 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CORE LOGIC ---

class SubtitleProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'[^\w\s\.\!\?а-яА-ЯёЁ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    @staticmethod
    def _process_subs_object(subs, file_identifier: str) -> List[Dict[str, Any]]:
        transcripts = []
        for i, sub in enumerate(subs):
            if len(sub.text.strip()) < 3:
                continue
            transcripts.append({
                'episode': file_identifier,
                'text': SubtitleProcessor.clean_text(sub.text),
                'raw_text': sub.text,
                'start': str(sub.start),
                'end': str(sub.end),
                'duration': sub.duration.seconds,
                'index': i,
                'id': f"{file_identifier}_{i}"
            })
        return transcripts

    @staticmethod
    def parse_file_path(file_path: Path) -> List[Dict[str, Any]]:
        try:
            subs = pysrt.open(str(file_path), encoding='utf-8')
            return SubtitleProcessor._process_subs_object(subs, file_path.stem)
        except Exception as e:
            try:
                subs = pysrt.open(str(file_path), encoding='cp1251')
                return SubtitleProcessor._process_subs_object(subs, file_path.stem)
            except:
                logger.error(f"Error parsing file {file_path}: {e}")
                return []

    @staticmethod
    def parse_uploaded_file(uploaded_file) -> List[Dict[str, Any]]:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            subs = pysrt.from_string(content)
            file_id = Path(uploaded_file.name).stem
            return SubtitleProcessor._process_subs_object(subs, file_id)
        except Exception as e:
            st.error(f"Ошибка чтения файла {uploaded_file.name}: {e}")
            return []

class VectorSearchService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )

    def reset_collection(self):
        try:
            self.client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            logger.error(f"Reset error: {e}")

    def index_data(self, data: List[Dict[str, Any]]) -> int:
        if not data:
            return 0
            
        ids = [item['id'] for item in data]
        documents = [item['text'] for item in data]
        metadatas = [{
            'episode': item['episode'],
            'start': item['start'],
            'end': item['end'],
            'raw_text': item['raw_text']
        } for item in data]

        total = len(ids)
        batch_size = self.config.BATCH_SIZE
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
        return total

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        clean_query = SubtitleProcessor.clean_text(query)
        try:
            results = self.collection.query(
                query_texts=[clean_query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            formatted = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': (1 - results['distances'][0][i]) * 100
                    })
            return formatted
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_stats(self) -> int:
        return self.collection.count()

# --- UI ---

def render_result(result: Dict[str, Any]):
    meta = result['metadata']
    score = result['score']
    with st.container(border=True):
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.markdown(f"**🎬 {meta['episode']}** `({meta['start']} -> {meta['end']})`")
            st.markdown(f"### \"{meta['raw_text']}\"")
        with c2:
            st.metric("Score", f"{score:.0f}%")

def main():
    st.set_page_config(page_title="Subtitle Search", page_icon="🔎", layout="wide")
    
    if 'config' not in st.session_state:
        st.session_state.config = AppConfig()
    
    @st.cache_resource
    def get_service():
        return VectorSearchService(st.session_state.config)
    
    service = get_service()

    with st.sidebar:
        st.header("⚙️ Источник данных")
        
        mode = st.radio("Откуда брать субтитры?", ["📁 Демо (AoT)", "📤 Загрузить свои (.srt)"])
        
        if mode == "📁 Демо (AoT)":
            full_path = SCRIPT_DIR / st.session_state.config.SUBS_FOLDER_NAME
            st.info(f"Папка на сервере: `{st.session_state.config.SUBS_FOLDER_NAME}`")
            
            if st.button("🔄 Индексировать Демо", type="primary"):
                with st.status("Индексация...", expanded=True) as status:
                    if not full_path.exists():
                        st.error("Папка демо не найдена.")
                        st.stop()
                    
                    files = list(full_path.glob("*.srt"))
                    all_transcripts = []
                    service.reset_collection() 
                    
                    prog = st.progress(0)
                    for i, f in enumerate(files):
                        all_transcripts.extend(SubtitleProcessor.parse_file_path(f))
                        prog.progress((i+1)/len(files))
                    
                    service.index_data(all_transcripts)
                    status.update(label="✅ Демо данные загружены!", state="complete", expanded=False)
                    st.rerun()

        else:
            uploaded_files = st.file_uploader("Перетащите .srt файлы сюда", type=["srt"], accept_multiple_files=True)
            if uploaded_files:
                if st.button(f"🚀 Индексировать {len(uploaded_files)} файлов", type="primary"):
                    with st.status("Обработка файлов...", expanded=True) as status:
                        all_transcripts = []
                        service.reset_collection()
                        
                        prog = st.progress(0)
                        for i, file in enumerate(uploaded_files):
                            transcripts = SubtitleProcessor.parse_uploaded_file(file)
                            all_transcripts.extend(transcripts)
                            prog.progress((i+1)/len(uploaded_files))
                        
                        if all_transcripts:
                            st.write(f"Загрузка {len(all_transcripts)} строк в БД...")
                            service.index_data(all_transcripts)
                            status.update(label="✅ Ваши файлы проиндексированы!", state="complete", expanded=False)
                            st.rerun()
                        else:
                            st.error("Не удалось прочитать файлы.")

        st.markdown("---")
        st.metric("В базе строк:", service.get_stats())

    # Main Area
    st.title("🔎 Семантический поиск по субтитрам")
    
    if service.get_stats() == 0:
        st.warning("👈 База пуста. Выберите источник слева и нажмите кнопку Индексации.")
    else:
        q = st.text_input("Запрос (можно на русском)", placeholder="О чем говорят герои?", label_visibility="collapsed")
        
        limit = st.selectbox(
            "Количество результатов", 
            options=range(1, 100),
            index=4
        )
        
        if q:
            with st.spinner("Поиск..."):
                res = service.search(q, limit)
            if res:
                for r in res:
                    render_result(r)
            else:
                st.info("Ничего не найдено.")

if __name__ == "__main__":
    main()
