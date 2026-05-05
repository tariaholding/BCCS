"""
BCCS — Tests unitaires du moteur RAG
======================================

Stratégie de test :
    - Tests UNITAIRES  : mocks complets, aucun service externe requis.
    - Tests INTÉGRATION: marqués @pytest.mark.integration, nécessitent
                         Qdrant et Ollama actifs (skip automatique sinon).

Lancement :
    pytest src/test_engine.py -v                        # unitaires seuls
    pytest src/test_engine.py -v -m integration         # avec services
    pytest src/test_engine.py -v --cov=src/engine       # avec couverture
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ajout du répertoire racine au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    BCCSConfig,
    BCCSEngine,
    BCCSError,
    BCCSResponse,
    ConnectionError,
    IngestionError,
    InsufficientContextError,
    ResponseStatus,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def config() -> BCCSConfig:
    """Configuration de test (pointe vers des services locaux)."""
    return BCCSConfig()


@pytest.fixture()
def mock_qdrant_client() -> MagicMock:
    """Client Qdrant entièrement simulé."""
    client = MagicMock()
    # get_collections() retourne une liste vide par défaut
    collections_mock = MagicMock()
    collections_mock.collections = []
    client.get_collections.return_value = collections_mock
    # count() retourne 42 nœuds fictifs
    count_mock = MagicMock()
    count_mock.count = 42
    client.count.return_value = count_mock
    return client


@pytest.fixture()
def node_above_threshold() -> MagicMock:
    """Nœud RAG avec score suffisant (au-dessus du seuil 0.60)."""
    node = MagicMock()
    node.score = 0.85
    node.metadata = {"source": "reglement_urbanisme.pdf", "file_name": "reglement_urbanisme.pdf"}
    node.text = "La mairie est ouverte du lundi au vendredi de 9h à 17h."
    return node


@pytest.fixture()
def node_below_threshold() -> MagicMock:
    """Nœud RAG avec score insuffisant (sous le seuil 0.60)."""
    node = MagicMock()
    node.score = 0.32
    node.metadata = {"source": "unknown.txt"}
    node.text = "Contenu non pertinent."
    return node


# ===========================================================================
# 1. Tests de BCCSConfig
# ===========================================================================

class TestBCCSConfig:
    def test_default_values(self) -> None:
        cfg = BCCSConfig()
        assert cfg.qdrant_host == "localhost"
        assert cfg.qdrant_port == 6333
        assert cfg.score_threshold == 0.60
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.top_k == 5

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_HOST", "qdrant-prod")
        monkeypatch.setenv("QDRANT_PORT", "6334")
        monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.75")
        monkeypatch.setenv("OLLAMA_MODEL", "mixtral:8x7b-instruct")
        cfg = BCCSConfig()
        assert cfg.qdrant_host == "qdrant-prod"
        assert cfg.qdrant_port == 6334
        assert cfg.score_threshold == 0.75
        assert cfg.ollama_model == "mixtral:8x7b-instruct"

    def test_data_dir_is_path(self) -> None:
        cfg = BCCSConfig()
        assert isinstance(cfg.data_dir, Path)


# ===========================================================================
# 2. Tests des exceptions
# ===========================================================================

class TestExceptions:
    def test_insufficient_context_error_message(self) -> None:
        exc = InsufficientContextError(
            query="Horaires mairie",
            retrieved=1,
            threshold=0.60,
            score=0.25,
        )
        assert "Horaires mairie" in str(exc)
        assert "0.250" in str(exc)
        assert exc.retrieved == 1
        assert exc.threshold == 0.60

    def test_insufficient_context_no_score(self) -> None:
        exc = InsufficientContextError(
            query="Test", retrieved=0, threshold=0.60, score=None
        )
        assert "aucun nœud récupéré" in str(exc)

    def test_ingestion_error_is_bccs_error(self) -> None:
        exc = IngestionError("Dossier introuvable")
        assert isinstance(exc, BCCSError)

    def test_connection_error_is_bccs_error(self) -> None:
        exc = ConnectionError("Ollama injoignable")
        assert isinstance(exc, BCCSError)


# ===========================================================================
# 3. Tests de BCCSResponse
# ===========================================================================

class TestBCCSResponse:
    def test_success_is_reliable(self) -> None:
        resp = BCCSResponse(
            status=ResponseStatus.SUCCESS,
            answer="Les horaires sont 9h-17h.",
            sources=["mairie.pdf"],
            scores=[0.87],
            query="Horaires mairie",
        )
        assert resp.is_reliable() is True

    def test_insufficient_context_not_reliable(self) -> None:
        resp = BCCSResponse(
            status=ResponseStatus.INSUFFICIENT_CONTEXT,
            answer="",
            query="Question sans contexte",
        )
        assert resp.is_reliable() is False

    def test_error_not_reliable(self) -> None:
        resp = BCCSResponse(
            status=ResponseStatus.ERROR,
            answer="",
            error_message="Erreur Qdrant",
        )
        assert resp.is_reliable() is False

    def test_to_dict_structure(self) -> None:
        resp = BCCSResponse(
            status=ResponseStatus.SUCCESS,
            answer="Réponse",
            sources=["a.pdf"],
            scores=[0.9],
            query="Q",
        )
        d = resp.to_dict()
        assert d["status"] == "success"
        assert d["reliable"] is True
        assert "answer" in d
        assert "sources" in d
        assert "scores" in d


# ===========================================================================
# 4. Tests de la validation de contexte (cœur anti-hallucination)
# ===========================================================================

class TestContextValidation:
    """Tests du garde-fou central contre les hallucinations."""

    def _make_engine(self) -> BCCSEngine:
        """Instancie un engine sans appeler Ollama ni Qdrant."""
        with patch.object(BCCSEngine, "_setup_llama_index"):
            return BCCSEngine()

    def test_raises_when_no_nodes(self) -> None:
        engine = self._make_engine()
        with pytest.raises(InsufficientContextError) as exc_info:
            engine._validate_context([], "Question sans réponse")
        assert exc_info.value.retrieved == 0

    def test_raises_when_score_below_threshold(self, node_below_threshold: MagicMock) -> None:
        engine = self._make_engine()
        with pytest.raises(InsufficientContextError) as exc_info:
            engine._validate_context([node_below_threshold], "Question peu pertinente")
        assert exc_info.value.score == pytest.approx(0.32, abs=1e-3)

    def test_passes_when_score_above_threshold(self, node_above_threshold: MagicMock) -> None:
        engine = self._make_engine()
        # Ne doit pas lever d'exception
        engine._validate_context([node_above_threshold], "Horaires mairie")

    def test_raises_when_insufficient_nodes_count(self, node_above_threshold: MagicMock) -> None:
        """Vérifie que min_context_nodes est bien respecté."""
        cfg = BCCSConfig()
        # On force min_context_nodes à 3
        object.__setattr__(cfg, "min_context_nodes", 3)
        with patch.object(BCCSEngine, "_setup_llama_index"):
            engine = BCCSEngine(config=cfg)
        with pytest.raises(InsufficientContextError):
            engine._validate_context([node_above_threshold], "Question")

    def test_best_score_used_for_validation(self) -> None:
        """Le score le plus élevé parmi les nœuds doit être utilisé."""
        engine = self._make_engine()
        low_node = MagicMock()
        low_node.score = 0.40
        low_node.metadata = {"source": "a.pdf"}
        high_node = MagicMock()
        high_node.score = 0.90
        high_node.metadata = {"source": "b.pdf"}
        # Ne doit pas lever d'exception (best_score=0.90 > 0.60)
        engine._validate_context([low_node, high_node], "Question")


# ===========================================================================
# 5. Tests de la connexion Qdrant (unitaires avec mock)
# ===========================================================================

class TestQdrantConnection:
    """Vérifie le comportement lors de la connexion à Qdrant."""

    def _make_engine(self) -> BCCSEngine:
        with patch.object(BCCSEngine, "_setup_llama_index"):
            return BCCSEngine()

    @patch("src.engine.QdrantClient")
    def test_qdrant_client_is_created_with_correct_params(
        self, mock_client_cls: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        mock_client_cls.return_value = mock_qdrant_client
        engine = self._make_engine()
        client = engine._get_qdrant_client()
        mock_client_cls.assert_called_once_with(
            host="localhost",
            port=6333,
            timeout=10,
        )
        assert client is mock_qdrant_client

    @patch("src.engine.QdrantClient")
    def test_qdrant_client_is_cached(
        self, mock_client_cls: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """_get_qdrant_client() ne crée le client qu'une seule fois."""
        mock_client_cls.return_value = mock_qdrant_client
        engine = self._make_engine()
        engine._get_qdrant_client()
        engine._get_qdrant_client()
        assert mock_client_cls.call_count == 1

    @patch("src.engine.QdrantClient")
    def test_raises_connection_error_on_qdrant_failure(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.side_effect = Exception("Connection refused")
        engine = self._make_engine()
        with pytest.raises(ConnectionError, match="Impossible de joindre Qdrant"):
            engine._get_qdrant_client()

    @patch("src.engine.QdrantClient")
    def test_api_key_passed_when_set(
        self, mock_client_cls: MagicMock, mock_qdrant_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QDRANT_API_KEY", "super-secret-key")
        mock_client_cls.return_value = mock_qdrant_client
        engine = self._make_engine()
        engine._get_qdrant_client()
        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs.get("api_key") == "super-secret-key"


# ===========================================================================
# 6. Tests du chargement CSV
# ===========================================================================

class TestCSVLoader:
    def test_csv_loaded_as_documents(self, tmp_path: Path) -> None:
        csv_content = "titre,description\nHoraires,Lundi 9h-17h\nContact,mairie@ville.fr"
        csv_file = tmp_path / "horaires.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        docs = BCCSEngine._load_csv(csv_file)
        assert len(docs) == 2
        assert "Horaires" in docs[0].text
        assert docs[0].metadata["source"] == "horaires.csv"
        assert docs[0].metadata["type"] == "csv"

    def test_empty_rows_ignored(self, tmp_path: Path) -> None:
        csv_content = "titre,description\n,\nHoraires,Lundi 9h-17h\n,"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        docs = BCCSEngine._load_csv(csv_file)
        assert len(docs) == 1  # Seule la ligne non vide est indexée

    def test_csv_row_index_in_metadata(self, tmp_path: Path) -> None:
        csv_content = "a,b\nval1,val2\nval3,val4"
        csv_file = tmp_path / "index_test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        docs = BCCSEngine._load_csv(csv_file)
        assert docs[0].metadata["row"] == 0
        assert docs[1].metadata["row"] == 1


# ===========================================================================
# 7. Tests de l'ingestion (mock complet)
# ===========================================================================

class TestIngestion:
    def _make_engine(self, config: Optional[BCCSConfig] = None) -> BCCSEngine:
        with patch.object(BCCSEngine, "_setup_llama_index"):
            return BCCSEngine(config=config)

    def test_raises_ingestion_error_when_data_dir_missing(self) -> None:
        cfg = BCCSConfig()
        object.__setattr__(cfg, "data_dir", Path("/nonexistent/path"))
        engine = self._make_engine(cfg)
        with pytest.raises(IngestionError, match="introuvable"):
            engine._load_documents()

    def test_raises_ingestion_error_when_no_documents(self, tmp_path: Path) -> None:
        cfg = BCCSConfig()
        object.__setattr__(cfg, "data_dir", tmp_path)
        engine = self._make_engine(cfg)
        with pytest.raises(IngestionError, match="Aucun document"):
            engine._load_documents()

    @patch("src.engine.QdrantClient")
    @patch("src.engine.VectorStoreIndex")
    @patch("src.engine.SimpleDirectoryReader")
    def test_ingest_returns_node_count(
        self,
        mock_reader_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_client_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Préparer un vrai fichier CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("titre,contenu\nHoraires,9h-17h", encoding="utf-8")

        mock_client_cls.return_value = mock_qdrant_client
        mock_reader_cls.return_value.load_data.return_value = []
        mock_index_cls.from_documents.return_value = MagicMock()

        cfg = BCCSConfig()
        object.__setattr__(cfg, "data_dir", tmp_path)
        engine = self._make_engine(cfg)

        count = engine.ingest()
        assert count == 42  # Valeur retournée par mock_qdrant_client.count


# ===========================================================================
# 8. Tests de la requête (mock complet)
# ===========================================================================

class TestQuery:
    def _make_engine(self) -> BCCSEngine:
        with patch.object(BCCSEngine, "_setup_llama_index"):
            return BCCSEngine()

    def test_empty_question_returns_error(self) -> None:
        engine = self._make_engine()
        response = engine.query("   ")
        assert response.status == ResponseStatus.ERROR
        assert "vide" in response.error_message

    def test_query_returns_reliable_response(
        self, node_above_threshold: MagicMock
    ) -> None:
        engine = self._make_engine()
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node_above_threshold]
        mock_index.as_retriever.return_value = mock_retriever
        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = MagicMock(__str__=lambda s: "La mairie ouvre à 9h.")
        mock_index.as_query_engine.return_value = mock_query_engine

        engine._index = mock_index
        response = engine.query("Horaires mairie")

        assert response.status == ResponseStatus.SUCCESS
        assert response.is_reliable() is True
        assert "9h" in response.answer
        assert "reglement_urbanisme.pdf" in response.sources

    def test_query_raises_on_low_context(
        self, node_below_threshold: MagicMock
    ) -> None:
        engine = self._make_engine()
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node_below_threshold]
        mock_index.as_retriever.return_value = mock_retriever
        engine._index = mock_index

        with pytest.raises(InsufficientContextError):
            engine.query("Question hors sujet totalement")


# ===========================================================================
# 9. Tests d'intégration (nécessitent Qdrant actif)
# ===========================================================================

@pytest.mark.integration
class TestQdrantIntegration:
    """
    Tests d'intégration réels — nécessitent Qdrant sur localhost:6333.
    Lancement : pytest src/test_engine.py -v -m integration
    """

    def test_qdrant_ping(self) -> None:
        """Vérifie que Qdrant répond et retourne la liste des collections."""
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333, timeout=5)
        result = client.get_collections()
        assert hasattr(result, "collections"), "Qdrant n'a pas retourné de collections"

    def test_qdrant_collection_creation(self) -> None:
        """Crée et supprime une collection de test."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(host="localhost", port=6333, timeout=5)
        test_collection = "bccs-integration-test"

        client.recreate_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        collections = [c.name for c in client.get_collections().collections]
        assert test_collection in collections

        client.delete_collection(test_collection)
        collections_after = [c.name for c in client.get_collections().collections]
        assert test_collection not in collections_after
