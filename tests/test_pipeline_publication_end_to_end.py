"""Build, adopt, append and publish a disposable native serving revision."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest
from tests.test_pipeline_entity_baseline import Model as NerModel
from tests.test_pipeline_semantic_baseline import TinyEmbedding

from semora.corpus import indexer
from semora.ner.indexing import build_ner_index
from semora.ner.names import build_entity_names
from semora.ner.spellfix import install_spellfix
from semora.pipeline.baseline import adopt_baseline
from semora.pipeline.batches import run_batch
from semora.pipeline.entities import index_batch_entities
from semora.pipeline.entity_adoption import adopt_entity_baseline
from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.ner import index_batch_ner
from semora.pipeline.ner_adoption import adopt_ner_baseline
from semora.pipeline.publication_audit import audit_publication
from semora.pipeline.publication_publish import publish_batch
from semora.pipeline.semantic import index_batch_vectors
from semora.pipeline.semantic_adoption import adopt_semantic_baseline
from semora.pipeline.semantic_faiss import build_semantic_generation
from semora.retrieval import SearchEngine
from semora.retrieval.indexing import build_bm25_index, build_lemma_index
from semora.retrieval.semantic_faiss import FaissBuildConfig
from semora.retrieval.semantic_store import read_semantic_build_spec
from semora.retrieval.semantic_vectors import build_semantic_vectors

np = pytest.importorskip("numpy")


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class Lemmatizer:
    def lemmatize_many(self, texts):
        return [text.casefold() for text in texts]


def test_native_publication_from_legacy_to_new_batch(tmp_path: Path, monkeypatch) -> None:
    library = os.environ.get("SEMORA_TEST_SPELLFIX")
    if not library:
        pytest.skip("Set SEMORA_TEST_SPELLFIX to a trusted spellfix library")
    try:
        import faiss
    except ImportError:
        pytest.skip("Native FAISS cannot load in this Windows process")
    faiss.omp_set_num_threads(1)
    install_spellfix(tmp_path, library)
    old = tmp_path / "legacy" / "Paper"
    old.mkdir(parents=True)
    (old / "old.md").write_text("# Old\nAna visits Ljubljana today.\n", encoding="utf-8")
    database = tmp_path / "indexes" / "semora.sqlite"
    semantic = database.parent / "semantic"
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: Tokenizer())
    indexer.ingest_corpus(old.parent, database, token_count=3, token_overlap=1)
    build_bm25_index(database)
    build_lemma_index(database, lemmatizer=Lemmatizer())
    build_semantic_vectors(database, semantic, dimensions=4, shard_size=2, model=TinyEmbedding())
    spec = read_semantic_build_spec(semantic)
    assert spec is not None
    (semantic / "manifest.json").write_text(json.dumps({
        **spec.as_dict(), "normalized": True, "metric": "cosine_via_inner_product",
        "chunks": spec.valid_chunks, "vectors_complete": True, "index_complete": False,
    }), encoding="utf-8")
    build_ner_index(database, model=NerModel(), device="cpu", pipeline_depth=1)
    entity = database.parent / "entities"
    build_entity_names(tmp_path, batch_size=1)
    adopt_baseline(database)
    adopt_semantic_baseline(database, semantic)
    adopt_ner_baseline(database)
    adopt_entity_baseline(database, entity)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    markdown = b"# New\nAna travels by train.\n"
    metadata = json.dumps({"Record": {"source": "Paper", "date": "1925 02 18"}}).encode()
    (incoming / "new.md").write_bytes(markdown)
    (incoming / "new_metadata.json").write_bytes(metadata)
    manifest = BatchManifest("new-batch", (
        BatchIssue("Paper/new.md", "new.md", "new_metadata.json",
                   hashlib.sha256(markdown).hexdigest(), hashlib.sha256(metadata).hexdigest()),
    ))
    run_batch(manifest, incoming, database, token_count=3, token_overlap=1, tokenizer=Tokenizer())
    index_batch_lexical(database, "new-batch")
    index_batch_lexical(database, "new-batch", stage="lemma", lemmatizer=Lemmatizer())
    index_batch_vectors(database, "new-batch", semantic, dimensions=4, shard_size=2, model=TinyEmbedding())
    index_batch_ner(database, "new-batch", model=NerModel())
    index_batch_entities(database, "new-batch", entity)
    generation = build_semantic_generation(
        database, semantic, config=FaissBuildConfig(index_type="flat"), faiss_module=faiss
    )["generation"]
    checked = audit_publication(database, "new-batch", semantic, generation, entity, faiss_module=faiss)
    assert checked["eligible"], checked["errors"]
    published = publish_batch(database, "new-batch", semantic, generation, entity, faiss_module=faiss)
    assert published["published"] and published["publication_ready"]

    class QueryModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode_query(self, _texts, **_kwargs):
            return np.asarray([[1, 1, 2, 3]], dtype="float32")

        encode = encode_query

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=QueryModel))
    engine = SearchEngine(database, semantic)
    try:
        assert engine.index_status()["bm25"]["indexed_articles"] == 2
        assert engine.index_status()["lemma"]["indexed_articles"] == 2
        hits = engine.search("semantic", "Ana", limit=10)
        assert hits
        found = engine.search_entities("Ana")
        assert found["entities"]
        assert found["generation"] == published["publication_id"]
        articles = engine.search_entity_articles("Ana", generation=found["generation"])
        assert articles["hits"]
        assert articles["generation"] == published["publication_id"]
    finally:
        engine.close()

    # A later batch overlaps the first issue. The old serving pointer must
    # become stale before its new issue is visible, then receive a new token.
    later = "# Later\nZürich welcomes Ana.\n".encode()
    later_metadata = json.dumps({"Record": {"source": "Paper", "date": "1925 02 19"}}).encode()
    (incoming / "later.md").write_bytes(later)
    (incoming / "later_metadata.json").write_bytes(later_metadata)
    second = BatchManifest("later-batch", (
        manifest.issues[0],
        BatchIssue("Paper/later.md", "later.md", "later_metadata.json",
                   hashlib.sha256(later).hexdigest(), hashlib.sha256(later_metadata).hexdigest()),
    ))
    run_batch(second, incoming, database, token_count=3, token_overlap=1, tokenizer=Tokenizer())
    with pytest.raises(ValueError, match="stale"):
        SearchEngine(database, semantic)
    assert index_batch_lexical(database, "later-batch")["added_articles"] == 1
    assert index_batch_lexical(
        database, "later-batch", stage="lemma", lemmatizer=Lemmatizer()
    )["added_articles"] == 1
    assert index_batch_vectors(
        database, "later-batch", semantic, dimensions=4, shard_size=2, model=TinyEmbedding()
    )["added_chunks"] > 0
    assert index_batch_ner(database, "later-batch", model=NerModel())["added_articles"] == 1
    assert index_batch_entities(database, "later-batch", entity)["added_articles"] == 1
    later_generation = build_semantic_generation(
        database, semantic, config=FaissBuildConfig(index_type="flat"), faiss_module=faiss
    )["generation"]
    assert later_generation != generation
    republished = publish_batch(
        database, "later-batch", semantic, later_generation, entity, faiss_module=faiss
    )
    assert republished["published"] and republished["publication_id"] != published["publication_id"]
    reopened = SearchEngine(database, semantic)
    try:
        assert reopened.index_status()["bm25"]["indexed_articles"] == 3
        assert reopened.index_status()["lemma"]["indexed_articles"] == 3
        assert reopened.search_entities("Ana")["generation"] == republished["publication_id"]
        with pytest.raises(ValueError, match="changed"):
            reopened.search_entity_articles("Ana", generation=published["publication_id"])
    finally:
        reopened.close()
    reranked = SearchEngine(database, semantic, semantic_rerank=True)
    try:
        assert reranked.search("semantic", "Ana", limit=3)
    finally:
        reranked.close()
