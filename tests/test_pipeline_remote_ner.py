"""Exercise remote NER through real HTTP and a disposable pipeline database."""

import hashlib
import json
import socket
import sqlite3
import threading
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semora.execution import ExecutionConfig
from semora.execution.contracts import semantic_fingerprint
from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.ner import index_batch_ner

IDENTITY = {'format': 'semora.classla-ner.identity.v1', 'test': 'fixed'}


def ingest(root, entries):
    source = root / 'inputs'
    records = []
    for name, content in entries.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Article\n' + content + '\n', encoding='utf-8')
        metadata = path.with_name(path.stem + '_metadata.json')
        metadata.write_text(json.dumps({'Record': {'source': 'Paper'}}), encoding='utf-8')
        records.append(BatchIssue(relative_path=name, markdown_path=name,
            metadata_path=metadata.relative_to(source).as_posix(),
            markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest()))
    database = root / 'indexes' / 'semora.sqlite'
    run_batch(BatchManifest('batch', tuple(records)), source, database, stages=('newspapers', 'articles'))
    return database


@contextmanager
def worker(identity=IDENTITY, *, drop=0, fail=False, barrier=None, before_response=None, runtime=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == '/v1/capabilities'
            assert self.headers['Authorization'] == 'Bearer ' + 's' * 32
            self.reply({'api_version': 'v1', 'ready': True, 'runtime': runtime or {}, 'tasks': {'ner': {
                'enabled': True, 'ready': True, 'identity': identity,
                'fingerprint': semantic_fingerprint(identity)}}, 'limits': {
                'max_documents': 32, 'max_input_bytes': 1048576,
                'max_request_bytes': 2097152, 'max_response_bytes': 4194304}})

        def do_POST(self):
            assert self.path == '/v1/ner'
            assert self.headers['Authorization'] == 'Bearer ' + 's' * 32
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(request)
            if barrier:
                barrier.wait(timeout=3)
            results = []
            for document in request['documents']:
                text = document['text']
                entities = [{'start': i, 'end': i + 3, 'label': 'PER', 'text': 'Ana', 'score': None}
                            for i in range(len(text)) if text.startswith('Ana', i)]
                results.append({'document_id': document['document_id'],
                                'content_sha256': document['content_sha256'], 'entities': entities})
            if before_response:
                before_response()
            if fail or len(calls) <= drop:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.reply({key: request[key] for key in ('api_version', 'request_id', 'task', 'fingerprint')} |
                       {'results': results})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def config(tmp_path, *urls, attempts=2):
    token = tmp_path / 'token'
    token.write_text('s' * 32, encoding='utf-8')
    return ExecutionConfig('remote', urls, token_file=token, read_timeout=1,
                           max_attempts=attempts, retry_delay=0.01)


def counts(database):
    with closing(sqlite3.connect(database)) as db:
        return tuple(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                     for table in ('article_ner_state', 'article_entities'))


def test_two_workers_mask_tables_and_preserve_offsets(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana <table><td>Ana</td></table> Ana',
                                 'Paper/b.md': 'No names.'})
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("UPDATE pipeline_batch_table_policies SET policy='exclude_html_tables_v1',state='active'")
    barrier = threading.Barrier(2)
    with worker(barrier=barrier) as (url1, calls1), worker(barrier=barrier) as (url2, calls2):
        result = index_batch_ner(database, 'batch', batch_articles=1, execution=config(tmp_path, url1, url2))
    assert result['added_articles'] == 2 and result['added_entities'] == 2
    assert len(calls1) == len(calls2) == 1
    with closing(sqlite3.connect(database)) as db:
        content = db.execute("SELECT content FROM articles WHERE content LIKE 'Ana%'").fetchone()[0]
        spans = db.execute('SELECT char_start,char_end FROM article_entities ORDER BY char_start').fetchall()
    assert [content[a:b] for a, b in spans] == ['Ana', 'Ana']
    assert spans[-1][0] == content.rindex('Ana')


def test_dropped_result_retries_without_duplicate_rows(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.'})
    with worker(drop=1) as (url, calls):
        result = index_batch_ner(database, 'batch', execution=config(tmp_path, url))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result['added_articles'] == result['added_entities'] == 1
    assert counts(database) == (1, 1)


def test_failed_worker_and_interrupted_coordinator_resume(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.', 'Paper/b.md': 'Ana again.'})
    with worker() as (url, _):
        def stop(event):
            if event['event'] == 'ner_checkpoint':
                raise RuntimeError('stop after checkpoint')
        with pytest.raises(RuntimeError, match='stop after checkpoint'):
            index_batch_ner(database, 'batch', batch_articles=1, progress=stop, execution=config(tmp_path, url))
    assert counts(database) == (1, 1)
    with worker(fail=True) as (url, calls):
        with pytest.raises(RuntimeError, match='retries'):
            index_batch_ner(database, 'batch', batch_articles=1, execution=config(tmp_path, url))
        assert len(calls) == 2
    assert counts(database) == (1, 1)
    with worker() as (url, _):
        result = index_batch_ner(database, 'batch', execution=config(tmp_path, url))
    assert result['added_articles'] == result['skipped_articles'] == 1
    assert counts(database) == (2, 2)


def test_incompatible_workers_do_not_register_run(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.'})
    with closing(sqlite3.connect(database)) as db:
        epoch = db.execute('SELECT epoch FROM pipeline_publication_state').fetchone()[0]
    with worker() as (url1, _), worker({**IDENTITY, 'test': 'other'}) as (url2, _):
        with pytest.raises(RuntimeError, match='incompatible'):
            index_batch_ner(database, 'batch', execution=config(tmp_path, url1, url2))
    with closing(sqlite3.connect(database)) as db:
        assert db.execute('SELECT COUNT(*) FROM ner_runs').fetchone()[0] == 0
        assert db.execute('SELECT epoch FROM pipeline_publication_state').fetchone()[0] == epoch
    assert counts(database) == (0, 0)


def test_existing_local_identity_requires_new_run(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.'})
    with closing(sqlite3.connect(database)) as db, db:
        db.execute('INSERT INTO ner_runs(ner_run_id,config_json) VALUES (?,?)',
                   ('classla-sl', json.dumps({'model': {'adapter': 'classla'}})))
        epoch = db.execute('SELECT epoch FROM pipeline_publication_state').fetchone()[0]
    with worker() as (url, calls):
        with pytest.raises(ValueError, match='different run ID'):
            index_batch_ner(database, 'batch', execution=config(tmp_path, url))
        assert not calls
        report = index_batch_ner(database, 'batch', run_id='remote', execution=config(tmp_path, url))
    assert report['added_articles'] == 1
    with closing(sqlite3.connect(database)) as db:
        assert db.execute('SELECT epoch FROM pipeline_publication_state').fetchone()[0] == epoch + 1


def test_source_changed_during_remote_inference_rolls_back(tmp_path):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.'})
    def mutate():
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("UPDATE articles SET content='Changed'")
    with worker(before_response=mutate) as (url, _):
        with pytest.raises(RuntimeError, match='Article changed'):
            index_batch_ner(database, 'batch', execution=config(tmp_path, url))
    assert counts(database) == (0, 0)


@pytest.mark.parametrize('change, error', [
    ("DELETE FROM pipeline_batch_issues WHERE batch_id='batch'", 'membership changed'),
    ("UPDATE pipeline_batches SET state='failed' WHERE batch_id='batch'", 'membership changed'),
    ("UPDATE pipeline_batch_table_policies SET state='repairing' WHERE batch_id='batch'", 'policy changed'),
])
def test_batch_mutation_during_remote_inference_rolls_back(tmp_path, change, error):
    database = ingest(tmp_path, {'Paper/a.md': 'Ana.'})

    def mutate():
        with closing(sqlite3.connect(database)) as db, db:
            db.execute(change)

    with worker(before_response=mutate) as (url, _):
        with pytest.raises(RuntimeError, match=error):
            index_batch_ner(database, 'batch', execution=config(tmp_path, url))
    assert counts(database) == (0, 0)


def test_grouped_remote_writes_interrupt_and_resume_atomically(tmp_path):
    database = ingest(tmp_path, {f"{i}.md": "Ana" for i in range(5)})
    seen = []
    def stop(event):
        if event["event"] == "ner_checkpoint":
            seen.append(event["completed_articles"])
            raise RuntimeError("operator checkpoint")
    with worker() as (url, calls):
        execution = config(tmp_path, url)
        with pytest.raises(RuntimeError, match="operator checkpoint"):
            index_batch_ner(database, 'batch', batch_articles=1, write_batch_articles=3,
                            execution=execution, progress=stop)
        assert seen == [3]
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM article_ner_state").fetchone()[0] == 3
        report = index_batch_ner(database, 'batch', batch_articles=1, write_batch_articles=3,
                                 execution=execution)
        assert report["complete"] and report["completed_articles"] == 5
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM article_entities").fetchone()[0] == 5


def test_grouped_remote_source_change_rolls_back_whole_write(tmp_path):
    database = ingest(tmp_path, {"1.md": "Ana", "2.md": "Ana"})
    def change():
        if len(calls) == 2:
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE articles SET content='Changed' "
                                   "WHERE article_id = (SELECT MAX(article_id) FROM articles)")
    with worker(before_response=change) as (url, calls):
        with pytest.raises(RuntimeError, match="Article changed"):
            index_batch_ner(database, 'batch', batch_articles=1, write_batch_articles=2,
                            execution=config(tmp_path, url))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM article_ner_state").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM article_entities").fetchone()[0] == 0
