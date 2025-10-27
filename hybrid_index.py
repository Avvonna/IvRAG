import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from embedders import (
    BaseEmbedder,
    EmbeddingConfig,
    create_embedder,
    validate_embedder_compatibility,
)

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def build_faiss_ip_index(
    emb: np.ndarray, 
    ids: Optional[np.ndarray] = None
) -> faiss.Index:
    """
    Построение FAISS индекса с нормализацией эмбеддингов
    
    Args:
        emb: Матрица эмбеддингов (не модифицируется)
        ids: ID документов
    
    Returns:
        index: FAISS IndexIDMap2 с нормализованными векторами
    """
    if emb.shape[0] == 0:
        raise ValueError("Cannot build index from empty embeddings")
    
    d = emb.shape[1]
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(base)
    
    # ИСПРАВЛЕНО: Копируем массив перед нормализацией
    emb_normalized = emb.astype(np.float32, copy=True)
    faiss.normalize_L2(emb_normalized)
    
    if ids is None:
        ids = np.arange(emb.shape[0], dtype=np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    
    index.add_with_ids(emb_normalized, ids) # type: ignore
    return index


def build_bm25(texts: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized = [tokenize_bm25(t) for t in texts]
    return BM25Okapi(tokenized), tokenized


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Ранговая нормализация скоров (более устойчива к выбросам)
    
    Args:
        scores: Исходные скоры
    
    Returns:
        normalized: Нормализованные скоры в диапазоне [0, 1]
    """
    if scores.size == 0:
        return scores
    
    # Сортируем индексы по убыванию скоров
    sorted_idx = np.argsort(scores)[::-1]
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(len(scores))
    
    # Нормализация рангов в [0, 1]
    if len(scores) > 1:
        return 1.0 - (ranks / (len(scores) - 1))
    return np.ones_like(scores)


def softmax_normalize(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax нормализация скоров
    
    Args:
        scores: Исходные скоры
        temperature: Температурный параметр (меньше = более резкое распределение)
    
    Returns:
        normalized: Нормализованные вероятности
    """
    if scores.size == 0:
        return scores
    
    scores_scaled = scores / temperature
    exp_scores = np.exp(scores_scaled - np.max(scores_scaled))
    return exp_scores / exp_scores.sum()


class HybridIndex:
    """
    Гибридный индекс с BM25 и FAISS
    
    Attributes:
        texts: Список текстов в корпусе
        ids: Внутренние ID документов
        emb: Матрица эмбеддингов (не нормализованная)
        faiss: FAISS индекс (содержит нормализованные векторы)
        embedder: Эмбеддер для создания векторов
        embedding_config: Конфигурация модели эмбеддингов
        use_bm25: Флаг использования BM25
        bm25: BM25 индекс (если включен)
        tokenized: Токенизированные тексты (если BM25 включен)
        bm25_normalization: Метод нормализации BM25 ('rank' или 'softmax')
    """
    
    def __init__(
        self,
        texts: pd.Series,
        embedder: BaseEmbedder,
        use_bm25: bool = True,
        bm25_normalization: str = "rank",
    ):
        """       
        Args:
            texts: Pandas Series с текстами для индексации
            embedder: Экземпляр эмбеддера для создания векторов
            use_bm25: Использовать ли BM25
            bm25_normalization: Метод нормализации BM25 ('rank' или 'softmax')
        """
        clean_texts = texts.dropna().astype("string")
        if clean_texts.empty:
            raise ValueError("Cannot build index from empty corpus")
        
        if bm25_normalization not in ("rank", "softmax"):
            raise ValueError(f"Invalid bm25_normalization: {bm25_normalization}. Use 'rank' or 'softmax'")
        
        self.texts = clean_texts.tolist()
        self.ids = np.arange(len(self.texts), dtype=np.int64)
        self.bm25_normalization = bm25_normalization
        
        # Сохраняем эмбеддер и его конфигурацию
        self.embedder = embedder
        self.embedding_config = embedder.config
        
        # Построение векторного индекса
        print(f"Building embeddings with {embedder.config.model_name}...")
        self.emb, _ = embedder.embed_series(clean_texts, treat_as="passage")
        
        # ИСПРАВЛЕНО: build_faiss_ip_index теперь не модифицирует self.emb
        self.faiss = build_faiss_ip_index(self.emb, self.ids)
        
        # Построение BM25 индекса
        self.use_bm25 = bool(use_bm25)
        if self.use_bm25:
            print("Building BM25 index...")
            self.bm25, self.tokenized = build_bm25(self.texts)
        else:
            self.bm25, self.tokenized = None, None
        
        print(f"Index built: {len(self.texts)} documents")
    
    def search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.6,
        bm25_k: Optional[int] = None,
        vec_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Гибридный поиск по запросу
        
        Комбинирует результаты BM25 и векторного поиска с весами:
        final_score = alpha * vec_score + (1 - alpha) * bm25_score
        
        Args:
            query: Поисковый запрос
            k: Количество результатов в финальном ранжировании
            alpha: Вес векторной части [0, 1], (1-alpha) для BM25
            bm25_k: Количество кандидатов из BM25 (по умолчанию 3*k)
            vec_k: Количество кандидатов из векторного поиска (по умолчанию 3*k)
        
        Returns:
            out: DataFrame с колонками:
                - rank: Позиция в ранжировании
                - score: Финальный скор
                - vec_score: Векторный скор (косинусное сходство)
                - bm25_score: BM25 скор (нормализованный)
                - id: ID документа
                - text: Текст документа
        """
        # Валидация входных параметров
        if not query or not query.strip():
            return self._empty_results()
        
        k = max(1, min(k, len(self.texts)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        
        # ИСПРАВЛЕНО: Более разумные значения по умолчанию
        bm25_k = bm25_k or min(3 * k, len(self.texts))
        vec_k = vec_k or min(3 * k, len(self.texts))
        
        # Получение кандидатов из обоих источников
        bm25_cands = self._search_bm25(query, bm25_k)
        vec_cands = self._search_vector(query, vec_k)
        
        # Проверка на пустой результат
        if not bm25_cands and not vec_cands:
            return self._empty_results()
        
        # Объединение и финальное ранжирование
        return self._merge_and_rank(bm25_cands, vec_cands, k, alpha)
    
    def _search_bm25(self, query: str, k: int) -> dict[int, float]:
        """
        Поиск по BM25
        
        Args:
            query: Поисковый запрос
            k: Количество топ результатов
        
        Returns:
            out: Словарь {doc_id: normalized_score}
        """
        if not self.use_bm25 or self.bm25 is None:
            return {}
        
        q_tok = tokenize_bm25(query)
        if not q_tok:
            return {}
        
        scores = self.bm25.get_scores(q_tok).astype("float32")
        if scores.size == 0:
            return {}
        
        # ИСПРАВЛЕНО: Правильный отбор и сортировка top-k
        if scores.size > k:
            # argpartition для эффективного отбора top-k
            top_idx = np.argpartition(scores, -k)[-k:]
            # Сортируем top-k по убыванию скора
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        else:
            # Если документов меньше k, просто сортируем все
            top_idx = np.argsort(scores)[::-1]
        
        top_scores = scores[top_idx]
        
        # ИСПРАВЛЕНО: Используем ранговую или softmax нормализацию
        if self.bm25_normalization == "rank":
            normalized = rank_normalize(top_scores)
        else:  # softmax
            normalized = softmax_normalize(top_scores)
        
        return {int(self.ids[idx]): float(norm_score) 
                for idx, norm_score in zip(top_idx, normalized)}
    
    def _search_vector(self, query: str, k: int) -> dict[int, float]:
        """
        Векторный поиск через FAISS
        
        Args:
            query: Поисковый запрос
            k: Количество топ результатов
        
        Returns:
            out: Словарь {doc_id: cosine_score}
        """
        if not query.strip():
            return {}
        
        # Создание query эмбеддинга
        q_emb = self.embedder.embed([query], treat_as="query")
        faiss.normalize_L2(q_emb)
        
        # FAISS поиск
        scores, doc_ids = self.faiss.search(q_emb, k) # type: ignore
        scores = scores[0]
        doc_ids = doc_ids[0]
        
        # Фильтрация валидных результатов (FAISS возвращает -1 для пустых слотов)
        valid = doc_ids != -1
        
        # Нормализация векторных скоров (косинусное сходство уже в [-1, 1])
        # Переводим в [0, 1] для единообразия
        valid_scores = (scores[valid] + 1.0) / 2.0
        
        return {int(doc_id): float(score) 
                for doc_id, score in zip(doc_ids[valid], valid_scores)}
    
    def _merge_and_rank(
        self,
        bm25_cands: dict[int, float],
        vec_cands: dict[int, float],
        k: int,
        alpha: float,
    ) -> pd.DataFrame:
        """
        Объединение кандидатов и финальное ранжирование
        
        Args:
            bm25_cands: Кандидаты из BM25 {doc_id: score}
            vec_cands: Кандидаты из векторного поиска {doc_id: score}
            k: Количество финальных результатов
            alpha: Вес векторного поиска
        
        Returns:
            out: Ранжированный DataFrame с результатами
        """
        # Если BM25 не работает, используем только векторный поиск
        eff_alpha = alpha if bm25_cands else 1.0
        
        # Объединение всех уникальных кандидатов
        all_ids = set(bm25_cands.keys()) | set(vec_cands.keys())
        
        # Вычисление финальных скоров
        results = []
        for doc_id in all_ids:
            vec_score = vec_cands.get(doc_id, 0.0)
            bm25_score = bm25_cands.get(doc_id, 0.0)
            final_score = eff_alpha * vec_score + (1.0 - eff_alpha) * bm25_score
            results.append((doc_id, final_score, vec_score, bm25_score))
        
        # Сортировка по финальному скору и отбор top-k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]
        
        df = pd.DataFrame(
            results,
            columns=["id", "score", "vec_score", "bm25_score"]
        )
        df["text"] = df["id"].map(lambda i: self.texts[i])
        df["rank"] = np.arange(1, len(df) + 1)
        
        return df[["rank", "score", "vec_score", "bm25_score", "id", "text"]]
    
    def _empty_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["rank", "score", "vec_score", "bm25_score", "id", "text"]
        )
    
    def save(self, path: str | Path) -> None:
        """ Сохранение индекса на диск """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.faiss, str(path / "faiss.index"))
        
        data = {
            "texts": self.texts,
            "ids": self.ids,
            "emb": self.emb,
            "embedding_config": self.embedding_config.to_dict(),
            "use_bm25": self.use_bm25,
            "tokenized": self.tokenized,
            "bm25_normalization": self.bm25_normalization,
        }
        with open(path / "index.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str | Path,
        embedder: Optional[BaseEmbedder] = None,
        **embedder_kwargs
    ) -> "HybridIndex":
        """ Загрузка индекса с диска и валидация """
        path = Path(path)
        
        with open(path / "index.pkl", "rb") as f:
            data = pickle.load(f)
        
        saved_config = EmbeddingConfig.from_dict(data["embedding_config"])
        
        # Если эмбеддер не передан, создаём из сохранённой конфигурации
        if embedder is None:
            embedder = create_embedder(saved_config, **embedder_kwargs)
            print(f"Created embedder from saved config: {saved_config.model_name}")
        else:
            # Проверяем совместимость
            validate_embedder_compatibility(embedder.config, saved_config)
            print(f"Using provided embedder: {embedder.config.model_name}")
        
        obj = cls.__new__(cls)
        obj.texts = data["texts"]
        obj.ids = data["ids"]
        obj.emb = data["emb"]
        obj.embedder = embedder
        obj.embedding_config = saved_config
        obj.use_bm25 = data["use_bm25"]
        obj.bm25_normalization = data.get("bm25_normalization", "rank")
        obj.bm25 = None
        obj.tokenized = data.get("tokenized")
        
        if obj.use_bm25 and obj.tokenized:
            print("Rebuilding BM25 index...")
            obj.bm25 = BM25Okapi(obj.tokenized)
        
        obj.faiss = faiss.read_index(str(path / "faiss.index"))
        print(f"Index loaded from {path}: {len(obj.texts)} documents")
        return obj
    
    def add_documents(self, new_texts: pd.Series) -> None:
        """ Добавление новых документов в индекс """
        clean_texts = new_texts.dropna().astype("string")
        if clean_texts.empty:
            return
        
        new_texts_list = clean_texts.tolist()
        n_old = len(self.texts)
        n_new = len(new_texts_list)
        
        # Обновление текстов и ID
        self.texts.extend(new_texts_list)
        new_ids = np.arange(n_old, n_old + n_new, dtype=np.int64)
        self.ids = np.concatenate([self.ids, new_ids])
        
        # Создание эмбеддингов для новых документов
        print(f"Creating embeddings for {n_new} new documents...")
        new_emb, _ = self.embedder.embed_series(
            clean_texts,
            treat_as="passage"
        )
        
        # ИСПРАВЛЕНО: Нормализуем эмбеддинги перед добавлением в FAISS
        new_emb_normalized = new_emb.astype(np.float32, copy=True)
        faiss.normalize_L2(new_emb_normalized)
        self.faiss.add_with_ids(new_emb_normalized, new_ids) # type: ignore
        
        # Сохраняем не нормализованные эмбеддинги
        self.emb = np.vstack([self.emb, new_emb])
        
        # Обновление BM25
        if self.use_bm25:
            print("Rebuilding BM25 index...")
            self.bm25, self.tokenized = build_bm25(self.texts)
        
        print(f"Added {n_new} documents. Total: {len(self.texts)}")
    
    def remove_documents(self, doc_ids: list[int]) -> None:
        """ Удаление документов из индекса """
        if not doc_ids:
            return
        
        doc_ids_set = set(doc_ids)
        keep_mask = np.array([i not in doc_ids_set for i in self.ids])
        
        if not keep_mask.any():
            raise ValueError("Cannot remove all documents from index")
        
        # Обновление данных
        self.texts = [t for i, t in enumerate(self.texts) if keep_mask[i]]
        self.ids = self.ids[keep_mask]
        self.emb = self.emb[keep_mask]
        
        # Пересоздание ID для непрерывности
        self.ids = np.arange(len(self.texts), dtype=np.int64)
        
        # Переиндексация FAISS
        print("Rebuilding FAISS index...")
        self.faiss = build_faiss_ip_index(self.emb, self.ids)
        
        # Переиндексация BM25
        if self.use_bm25:
            print("Rebuilding BM25 index...")
            self.bm25, self.tokenized = build_bm25(self.texts)
        
        print(f"Removed {len(doc_ids)} documents. Total: {len(self.texts)}")
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """ Получение текста документа по ID """
        if 0 <= doc_id < len(self.texts):
            return self.texts[doc_id]
        return None
    
    def get_embedding(self, doc_id: int) -> Optional[np.ndarray]:
        """ Получение эмбеддинга документа по ID """
        if 0 <= doc_id < len(self.emb):
            return self.emb[doc_id]
        return None
    
    def __len__(self) -> int:
        """ Количество документов в индексе """
        return len(self.texts)
    
    def __repr__(self) -> str:
        bm25_status = "enabled" if self.use_bm25 else "disabled"
        return (
            f"HybridIndex(\n"
            f"  documents={len(self.texts)},\n"
            f"  model='{self.embedding_config.model_name}',\n"
            f"  bm25={bm25_status},\n"
            f"  bm25_norm='{self.bm25_normalization}'\n"
            f")"
        )
    
    def __getitem__(self, doc_id: int) -> str:
        if 0 <= doc_id < len(self.texts):
            return self.texts[doc_id]
        raise IndexError(f"Document ID {doc_id} out of range [0, {len(self.texts)})")