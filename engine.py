"""
BCCS — Moteur RAG souverain
============================
Pipeline Retrieval-Augmented Generation entièrement local.

Composants :
    - Ingestion  : scan data/ (PDF, CSV) → chunks → embeddings
    - Stockage   : Qdrant (vector store persistant)
    - Inférence  : Ollama (LLM + embeddings, 100 % on-premise)
    - Sécurité   : détection d'insuffisance de contexte → refus explicite
                   (zéro hallucination tolérée)

Usage CLI :
    python src/engine.py ingest          # Indexe data/raw/
    python src/engine.py query "Quels sont les horaires de la mairie ?"
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# --- LlamaIndex core ---
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.base.response.schema import Response

# --- Intégrations souveraines ---
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

# --- Qdrant ---
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bccs.engine")


# ---------------------------------------------------------------------------
# Configuration (lecture depuis les variables d'environnement)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BCCSConfig:
    """Paramètres de l'engine, injectés via l'environnement Docker."""

    # Chemins
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("BCCS_DATA_DIR", "data/raw")))

    # Ollama
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral:7b-instruct"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # Qdrant
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "bccs-documents"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))

    # RAG
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "64")))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    score_threshold: float = field(default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.60")))
    min_context_nodes: int = field(default_factory=lambda: int(os.getenv("RAG_MIN_CONTEXT_NODES", "1")))


# ---------------------------------------------------------------------------
# Exceptions métier
# ---------------------------------------------------------------------------
class BCCSError(Exception):
    """Erreur de base du moteur BCCS."""


class InsufficientContextError(BCCSError):
    """
    Levée quand le contexte récupéré est insuffisant pour répondre
    de manière fiable. Prévient toute hallucination.
    """

    def __init__(self, query: str, retrieved: int, threshold: float, score: Optional[float] = None) -> None:
        self.query = query
        self.retrieved = retrieved
        self.threshold = threshold
        self.score = score
        detail = f"score={score:.3f}" if score is not None else "aucun nœud récupéré"
        super().__init__(
            f"Contexte insuffisant pour répondre à « {query} » "
            f"({retrieved} nœud(s) récupéré(s), {detail}, "
            f"seuil de confiance={threshold}). "
            "Réponse refusée pour prévenir toute hallucination."
        )


class IngestionError(BCCSError):
    """Erreur lors de l'indexation des documents."""


class ConnectionError(BCCSError):  # noqa: A001
    """Impossible de joindre Ollama ou Qdrant."""


# ---------------------------------------------------------------------------
# Enum statuts de réponse
# ---------------------------------------------------------------------------
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Résultat structuré
# ---------------------------------------------------------------------------
@dataclass
class BCCSResponse:
    """Réponse retournée par le moteur — toujours typée, jamais ambiguë."""

    status: ResponseStatus
    answer: str
    sources: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    query: str = ""
    error_message: str = ""

    def is_reliable(self) -> bool:
        return self.status == ResponseStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "answer": self.answer,
            "sources": self.sources,
            "scores": self.scores,
            "query": self.query,
            "error_message": self.error_message,
            "reliable": self.is_reliable(),
        }


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------
class BCCSEngine:
    """
    Moteur RAG souverain de la BCCS.

    Cycle de vie :
        engine = BCCSEngine()
        engine.ingest()          # une fois, ou à chaque mise à jour des docs
        response = engine.query("Ma question citoyenne")
    """

    def __init__(self, config: Optional[BCCSConfig] = None) -> None:
        self.config = config or BCCSConfig()
        self._index: Optional[VectorStoreIndex] = None
        self._qdrant_client: Optional[QdrantClient] = None
        self._setup_llama_index()

    # ------------------------------------------------------------------
    # Configuration LlamaIndex globale
    # ------------------------------------------------------------------
    def _setup_llama_index(self) -> None:
        """Configure LlamaIndex pour utiliser Ollama exclusivement."""
        logger.info("Configuration du moteur LlamaIndex (Ollama souverain)…")
        try:
            Settings.llm = Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                request_timeout=120.0,
                context_window=8192,
                temperature=0.1,       # Réponses déterministes, moins d'inventions
            )
            Settings.embed_model = OllamaEmbedding(
                model_name=self.config.ollama_embed_model,
                base_url=self.config.ollama_base_url,
                request_timeout=60.0,
            )
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                paragraph_separator="\n\n",
            )
            Settings.num_output = 1024
            logger.info("LlamaIndex configuré — LLM: %s | Embed: %s",
                        self.config.ollama_model, self.config.ollama_embed_model)
        except Exception as exc:
            raise ConnectionError(f"Impossible de configurer Ollama : {exc}") from exc

    # ------------------------------------------------------------------
    # Connexion Qdrant
    # ------------------------------------------------------------------
    def _get_qdrant_client(self) -> QdrantClient:
        """Retourne (et met en cache) le client Qdrant."""
        if self._qdrant_client is None:
            logger.info("Connexion à Qdrant — %s:%s", self.config.qdrant_host, self.config.qdrant_port)
            try:
                kwargs: dict = {
                    "host": self.config.qdrant_host,
                    "port": self.config.qdrant_port,
                    "timeout": 10,
                }
                if self.config.qdrant_api_key:
                    kwargs["api_key"] = self.config.qdrant_api_key
                self._qdrant_client = QdrantClient(**kwargs)
                # Ping pour valider la connexion
                self._qdrant_client.get_collections()
                logger.info("Connexion Qdrant établie ✓")
            except Exception as exc:
                raise ConnectionError(f"Impossible de joindre Qdrant : {exc}") from exc
        return self._qdrant_client

    # ------------------------------------------------------------------
    # Chargement des documents (PDF + CSV)
    # ------------------------------------------------------------------
    def _load_documents(self) -> list[Document]:
        """
        Charge les documents depuis data_dir.
        Supporte : PDF, CSV, TXT, DOCX, HTML, MD.
        """
        data_path = self.config.data_dir
        if not data_path.exists():
            raise IngestionError(f"Le dossier de données est introuvable : {data_path}")

        docs: list[Document] = []

        # --- PDF, TXT, DOCX, HTML, MD via SimpleDirectoryReader ---
        supported_non_csv = [".pdf", ".txt", ".docx", ".html", ".md"]
        non_csv_files = [f for f in data_path.rglob("*") if f.suffix.lower() in supported_non_csv]

        if non_csv_files:
            logger.info("Chargement de %d fichier(s) (PDF/TXT/DOCX/HTML/MD)…", len(non_csv_files))
            try:
                reader = SimpleDirectoryReader(
                    input_dir=str(data_path),
                    required_exts=supported_non_csv,
                    recursive=True,
                    exclude_hidden=True,
                )
                docs.extend(reader.load_data())
            except Exception as exc:
                raise IngestionError(f"Erreur lors du chargement des fichiers binaires : {exc}") from exc

        # --- CSV : traitement ligne par ligne pour enrichir les métadonnées ---
        csv_files = list(data_path.rglob("*.csv"))
        for csv_path in csv_files:
            logger.info("Ingestion CSV : %s", csv_path.name)
            try:
                docs.extend(self._load_csv(csv_path))
            except Exception as exc:
                logger.warning("Échec CSV %s : %s (fichier ignoré)", csv_path.name, exc)

        if not docs:
            raise IngestionError(
                f"Aucun document trouvé dans « {data_path} ». "
                "Déposez des fichiers PDF ou CSV et relancez l'ingestion."
            )

        logger.info("%d document(s) chargé(s) au total.", len(docs))
        return docs

    @staticmethod
    def _load_csv(csv_path: Path) -> list[Document]:
        """Convertit chaque ligne d'un CSV en Document LlamaIndex."""
        documents: list[Document] = []
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                text = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
                if text.strip():
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source": csv_path.name,
                                "row": i,
                                "type": "csv",
                            },
                        )
                    )
        return documents

    # ------------------------------------------------------------------
    # Ingestion : chargement → chunking → embedding → stockage Qdrant
    # ------------------------------------------------------------------
    def ingest(self, force: bool = False) -> int:
        """
        Lance la pipeline d'ingestion complète.

        Args:
            force: Si True, recrée la collection même si elle existe.

        Returns:
            Nombre de nœuds indexés.

        Raises:
            IngestionError: En cas d'échec irrémédiable.
        """
        logger.info("=== Démarrage de l'ingestion BCCS ===")
        client = self._get_qdrant_client()

        # Vérification / création de la collection
        existing = [c.name for c in client.get_collections().collections]
        if self.config.qdrant_collection in existing and force:
            logger.warning("Collection existante supprimée (mode force).")
            client.delete_collection(self.config.qdrant_collection)

        # Chargement des documents
        documents = self._load_documents()

        # Construction du vector store et de l'index
        try:
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.qdrant_collection,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info("Génération des embeddings et indexation Qdrant…")
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
        except Exception as exc:
            raise IngestionError(f"Échec de l'indexation : {exc}") from exc

        # Compter les nœuds réellement insérés
        try:
            count = client.count(self.config.qdrant_collection).count
        except Exception:
            count = len(documents)

        logger.info("=== Ingestion terminée : %d nœud(s) dans Qdrant ===", count)
        return count

    # ------------------------------------------------------------------
    # Chargement de l'index existant (si ingestion déjà faite)
    # ------------------------------------------------------------------
    def _load_index(self) -> VectorStoreIndex:
        """Charge l'index depuis Qdrant (sans re-ingérer)."""
        if self._index is not None:
            return self._index

        client = self._get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        if self.config.qdrant_collection not in existing:
            raise IngestionError(
                f"Collection « {self.config.qdrant_collection} » absente de Qdrant. "
                "Lancez d'abord : python src/engine.py ingest"
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.config.qdrant_collection,
        )
        self._index = VectorStoreIndex.from_vector_store(vector_store)
        return self._index

    # ------------------------------------------------------------------
    # Vérification anti-hallucination du contexte récupéré
    # ------------------------------------------------------------------
    def _validate_context(self, nodes: list[NodeWithScore], query: str) -> None:
        """
        Analyse la pertinence des nœuds récupérés.
        Lève InsufficientContextError si le contexte est trop faible.

        Stratégie double :
          1. Nombre minimum de nœuds (min_context_nodes).
          2. Score de similarité minimum (score_threshold).
        """
        if len(nodes) < self.config.min_context_nodes:
            raise InsufficientContextError(
                query=query,
                retrieved=len(nodes),
                threshold=self.config.score_threshold,
                score=None,
            )

        best_score = max((n.score or 0.0) for n in nodes)
        if best_score < self.config.score_threshold:
            raise InsufficientContextError(
                query=query,
                retrieved=len(nodes),
                threshold=self.config.score_threshold,
                score=best_score,
            )

        logger.info(
            "Contexte validé : %d nœud(s), meilleur score=%.3f (seuil=%.2f)",
            len(nodes), best_score, self.config.score_threshold,
        )

    # ------------------------------------------------------------------
    # Requête RAG
    # ------------------------------------------------------------------
    def query(self, question: str) -> BCCSResponse:
        """
        Pose une question au moteur RAG.

        Returns:
            BCCSResponse structuré — vérifier .is_reliable() avant d'afficher.

        Raises:
            InsufficientContextError: Si le contexte est insuffisant.
            BCCSError: Pour toute autre erreur interne.
        """
        question = question.strip()
        if not question:
            return BCCSResponse(
                status=ResponseStatus.ERROR,
                answer="",
                query=question,
                error_message="La question ne peut pas être vide.",
            )

        logger.info("Requête RAG : « %s »", question)

        try:
            index = self._load_index()
        except IngestionError as exc:
            return BCCSResponse(
                status=ResponseStatus.ERROR,
                answer="",
                query=question,
                error_message=str(exc),
            )

        # Retrieval : récupérer les nœuds sans générer de réponse
        retriever = index.as_retriever(similarity_top_k=self.config.top_k)
        try:
            nodes: list[NodeWithScore] = retriever.retrieve(QueryBundle(question))
        except Exception as exc:
            raise BCCSError(f"Échec du retrieval : {exc}") from exc

        # ⚠️  Vérification anti-hallucination — lève une exception si insuffisant
        self._validate_context(nodes, question)

        # Génération de la réponse via le LLM
        query_engine = index.as_query_engine(
            similarity_top_k=self.config.top_k,
            streaming=False,
            response_mode="compact",
        )
        try:
            llm_response: Response = query_engine.query(question)
        except Exception as exc:
            raise BCCSError(f"Échec de la génération LLM : {exc}") from exc

        # Extraction des sources et scores
        sources = []
        scores = []
        for node in nodes:
            src = node.metadata.get("source") or node.metadata.get("file_name", "inconnu")
            if src not in sources:
                sources.append(src)
            scores.append(round(node.score or 0.0, 4))

        return BCCSResponse(
            status=ResponseStatus.SUCCESS,
            answer=str(llm_response).strip(),
            sources=sources,
            scores=scores,
            query=question,
        )


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    """Interface en ligne de commande minimale."""
    if len(sys.argv) < 2:
        print("Usage :")
        print("  python src/engine.py ingest")
        print('  python src/engine.py query "Votre question"')
        sys.exit(1)

    cmd = sys.argv[1].lower()
    engine = BCCSEngine()

    if cmd == "ingest":
        force = "--force" in sys.argv
        count = engine.ingest(force=force)
        print(f"✅ Ingestion réussie : {count} nœud(s) indexé(s).")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("❌ Fournissez une question : python src/engine.py query \"Ma question\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        try:
            response = engine.query(question)
            print(f"\n📋 Statut  : {response.status.value}")
            print(f"❓ Question : {response.query}")
            print(f"💬 Réponse  : {response.answer}")
            print(f"📚 Sources  : {', '.join(response.sources) or 'N/A'}")
            print(f"📊 Scores   : {response.scores}")
        except InsufficientContextError as exc:
            print(f"\n⚠️  REFUS ANTI-HALLUCINATION : {exc}")
            sys.exit(2)
        except BCCSError as exc:
            print(f"\n❌ Erreur moteur : {exc}")
            sys.exit(1)

    else:
        print(f"Commande inconnue : {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
