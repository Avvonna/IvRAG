from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

# Абстрактные классы для совместимости

@dataclass
class EmbeddingConfig:
    model_type: str
    model_name: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingConfig":
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"EmbeddingConfig({self.model_type}:{self.model_name})"


# Базовый класс, чтобы было проще добавлять другие эмбеддеры

class BaseEmbedder(ABC):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    def embed(
        self, 
        texts: list[str], 
        treat_as: Literal["passage", "query"] = "passage"
    ) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass
    
    def embed_series(
        self,
        s: pd.Series,
        treat_as: Literal["passage", "query"] = "passage",
    ) -> tuple[np.ndarray, list[str]]:
        s = s.fillna("").astype("string")
        texts = s.tolist()
        
        if not texts:
            dim = self.get_dimension()
            return np.empty((0, dim), dtype=np.float32), []
        
        emb = self.embed(texts, treat_as=treat_as)
        return emb, texts
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config.model_name})"


# Классы реализованных эмбеддеров

class SentenceTransformerEmbedder(BaseEmbedder):   
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        config = EmbeddingConfig(
            model_type="sentence-transformer",
            model_name=model_name
        )
        super().__init__(config)
        
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("torch и sentence-transformers должны быть установлены"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(
        self,
        texts: list[str],
        treat_as: Literal["passage", "query"] = "passage"
    ) -> np.ndarray:
        texts_with_prefix = [self._add_prefix(t, treat_as) for t in texts]
        
        emb = self.model.encode(
            texts_with_prefix,
            batch_size=self.batch_size,
            normalize_embeddings=False, # эмбеддинги уже нормализуются при построении faiss
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return emb.astype("float32")
    
    def get_dimension(self) -> int | None:
        return self._dimension
    
    def _add_prefix(self, text: str, treat_as: str) -> str:
        name = self.config.model_name.lower()
        if "e5" in name or "bge" in name:
            prefix = "query" if treat_as == "query" else "passage"
            return f"{prefix}: {text}"
        return text


# Фабрика для создания эмбеддеров

def create_embedder(config: EmbeddingConfig, **kwargs) -> BaseEmbedder:
    """ Функция для создания эмбеддера из конфигурации """
    embedder_map = {
        "sentence-transformer": SentenceTransformerEmbedder
    }
    
    embedder_class = embedder_map.get(config.model_type)
    if embedder_class is None:
        raise ValueError(
            f"Unknown embedder type: {config.model_type}. "
            f"Available: {list(embedder_map.keys())}"
        )
    
    return embedder_class(model_name=config.model_name, **kwargs)

def validate_embedder_compatibility(
    current: EmbeddingConfig,
    saved: EmbeddingConfig
) -> None:
    """ Проверка совместимости двух конфигураций эмбеддеров """

    if current.model_name != saved.model_name:
        raise ValueError(
            f"Embedder model mismatch!\n"
            f"  Index was built with: {saved.model_name}\n"
            f"  You're trying to use: {current.model_name}\n"
            f"Please use the same model or rebuild the index."
        )
    
    if current.model_type != saved.model_type:
        raise ValueError(
            f"Embedder type mismatch!\n"
            f"  Index type: {saved.model_type}\n"
            f"  Current type: {current.model_type}"
        )
