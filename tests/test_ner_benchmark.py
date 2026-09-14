import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from types import SimpleNamespace

import pytest

from semora.diagnostics.ner.adapters import align_classla_tokens
from semora.diagnostics.ner.base import Entity, Prediction, validate
from semora.diagnostics.ner.report import highlight, report
from semora.diagnostics.ner.runner import run
from semora.diagnostics.ner.sample import create_sample, readonly
from semora.diagnostics.ner.summary import summarize


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as db, db:
        db.executescript("""
            CREATE TABLE newspapers (newspaper_id TEXT PRIMARY KEY, title TEXT, date TEXT, urn TEXT);
            INSERT INTO newspapers VALUES ('n', 'Jutro', '1934-01-01', 'URN1');
            CREATE TABLE articles (article_id TEXT, newspaper_id TEXT, title TEXT, content TEXT, is_valid INTEGER);
        """)
        for i in range(20):
            db.execute(
                "INSERT INTO articles(rowid, article_id, newspaper_id, content, is_valid) VALUES(?,?,?,?,?)",
                (i * 3 + 1, str(i), "n", f"Janez <script>alert('x')</script> {i}", int(i % 2 == 0)),
            )
    output = tmp_path / "sample.sqlite"
    before = source.read_bytes()
    create_sample(source, output, count=6, seed=4)
    assert source.read_bytes() == before
    return source, output


def test_reproducible_sampling_with_gaps(sample, tmp_path):
    source, first = sample
    second = tmp_path / "second.sqlite"
    create_sample(source, second, count=6, seed=4)
    with closing(readonly(first)) as a, closing(readonly(second)) as b:
        assert (
            a.execute("SELECT value FROM metadata").fetchone()[0]
            == b.execute("SELECT value FROM metadata").fetchone()[0]
        )
        assert all(int(row[0]) % 2 == 0 for row in a.execute("SELECT article_id FROM articles"))
    with pytest.raises(FileExistsError):
        create_sample(source, first, count=6)


def test_span_alignment_and_html_escaping():
    text = "Žiga <script>"
    validate(text, [Entity(0, 4, "PER", "Žiga")])
    with pytest.raises(ValueError):
        validate(text, [Entity(1, 4, "PER", "Žiga")])
    rendered = highlight(text, [{"start": 0, "end": 4, "label": 'PER"<'}], 0, len(text))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&quot;&lt;" in rendered


class FakeAdapter:
    metadata = {"fake": True}

    def __init__(self, empty=False, fail=False):
        self.empty, self.fail = empty, fail
        self.calls = 0

    def predict(self, texts, batch_size):
        self.calls += 1
        if self.fail and self.calls == 3:
            raise RuntimeError("simulated inference failure")
        return Prediction([[] if self.empty else [Entity(0, 5, "PER", text[:5])] for text in texts])


def test_classla_paragraph_offset_reconstruction():
    class Token:
        def __init__(self, text):
            self.text = text

        @property
        def end_char(self):
            return self._end_char

    tokens = [Token("Žiga"), Token("Žiga")]
    document = SimpleNamespace(iter_tokens=lambda: iter(tokens))
    align_classla_tokens(document, "Žiga\n\nŽiga")
    assert [(t._start_char, t._end_char) for t in tokens] == [(0, 4), (6, 10)]
    with pytest.raises(ValueError, match="Cannot align"):
        align_classla_tokens(document, "Žiga ! Žiga")


def test_worker_cleanup_reaps_descendant():
    psutil = pytest.importorskip("psutil")
    from semora.diagnostics.ner.__main__ import stop_worker

    code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    try:
        child_pid = int(process.stdout.readline())
    finally:
        stop_worker(process)
        process.stdout.close()
    assert process.poll() is not None
    assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE


def test_run_report_and_failure(sample, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("psutil")
    _, source = sample
    monkeypatch.setattr("semora.diagnostics.ner.runner.load_adapter", lambda *a, **k: FakeAdapter())
    a = tmp_path / "a"
    run(source, a, "fake", device="cpu", batch_articles=2)
    metrics = json.loads((a / "metrics.json").read_text())
    assert metrics["completed_articles"] == 6 and metrics["status"] == "complete"
    monkeypatch.setattr("semora.diagnostics.ner.runner.load_adapter", lambda *a, **k: FakeAdapter(empty=True))
    b = tmp_path / "b"
    run(source, b, "fake", device="cpu", batch_articles=2)
    report(source, [a, b], tmp_path / "report", limit=3)
    page = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert "6 of 6" in page and "<script>" not in page and "&lt;script&gt;" in page
    monkeypatch.setattr("semora.diagnostics.ner.runner.load_adapter", lambda *a, **k: FakeAdapter(fail=True))
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        run(source, failed, "fake", device="cpu", batch_articles=2)
    assert json.loads((failed / "metrics.json").read_text())["completed_articles"] == 2
    with pytest.raises(ValueError, match="completed runs"):
        report(source, [a, failed], tmp_path / "invalid")
    summarize([a, b, failed], tmp_path / "summary")
    summary = json.loads((tmp_path / "summary" / "summary.json").read_text())
    assert len(summary) == 1 and summary[0]["repeats"] == 2
    assert summary[0]["articles_per_second_std"] is not None
    assert "failed" in (tmp_path / "summary" / "statuses.csv").read_text()


def test_report_rejects_different_sample(sample, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("psutil")
    _, source = sample
    monkeypatch.setattr("semora.diagnostics.ner.runner.load_adapter", lambda *a, **k: FakeAdapter())
    target = tmp_path / "run"
    run(source, target, "fake", device="cpu")
    path = target / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["sample"]["sample_sha256"] = "different"
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="identical sample"):
        report(source, [target, target], tmp_path / "report", names=["a", "b"])


@pytest.mark.skipif(os.environ.get("SEMORA_TEST_NER_MODELS") != "1", reason="Requires cached NER models and CUDA")
@pytest.mark.parametrize("adapter", ["classla", "transformers"])
def test_real_models_preserve_boundaries_unicode_and_long_article_tail(adapter):
    from semora.diagnostics.ner.adapters import load_adapter

    model = load_adapter(adapter, device="cuda")
    sentence = "Janez Novak živi v Ljubljani."
    texts = ["Že danes: " + sentence, "To je preprost stavek.\n\n" * 400 + sentence]
    prediction = model.predict(texts, batch_size=16)
    assert len(prediction.entities) == 2
    for text, entities in zip(texts, prediction.entities, strict=True):
        validate(text, entities)
        expected = text.index("Janez Novak")
        assert any(e.start == expected and e.text == "Janez Novak" and e.label == "PER" for e in entities)
