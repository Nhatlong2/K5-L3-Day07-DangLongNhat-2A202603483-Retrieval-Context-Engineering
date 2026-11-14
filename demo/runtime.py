"""Connect the demo to local indexes, benchmark artifacts and Day 07 retrieval."""

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient, models

from demo.agent import DemoProvider
from src.common import (CHUNK_SNAPSHOT, DEFAULT_STORE, ROOT, WORK, LabError,
                        read_json, safe_error, write_json)
from src.day06 import DIMENSIONS, EMBEDDING_MODEL, ActiveStore, load_snapshot, rebuild
from src.experiment import trace_query
from src.provider import usage_report
from src.retrieval import Retriever

DEMO_DIR = ROOT / 'demo'
DEMO_WORK = WORK / 'demo'
DEMO_SNAPSHOT = DEMO_WORK / 'day06-chunks.json'
RESULTS = ROOT / 'results' / 'agent-dev'
METHODS = {'dense': 'Dense', 'lexical': 'Lexical', 'hybrid': 'Hybrid'}
PLAN = read_json(ROOT / 'benchmark' / 'experiments.json')
PRICES = read_json(ROOT / 'benchmark' / 'prices.json')
COLLECTION = 'chunks'
BATCH_SIZE = 32


def demo_config(preset, top_k, budget, model):
    config = {key: PLAN[key] for key in ('max_searches', 'ranking_cutoff', 'rrf_k', 'max_completion_tokens')}
    config.update(PLAN['variants'][preset], id=preset, top_k=top_k, context_budget_chars=budget,
                  agent_model=model, rrf_candidates=max(PLAN['rrf_candidates'], top_k))
    return config


class UploadStore:
    def __init__(self, path, chunks):
        self.path = Path(path)
        self.chunks = {chunk['id']: chunk for chunk in chunks}

    def manifest(self):
        return {'model': EMBEDDING_MODEL, 'dimensions': DIMENSIONS, 'distance': 'cosine',
                'ids': sorted(self.chunks)}

    def ready(self):
        try:
            return read_json(self.path / 'ready.json') == self.manifest()
        except (OSError, ValueError):
            return False

    @contextmanager
    def connect(self):
        try:
            client = QdrantClient(path=str(self.path))
        except Exception as exc:
            raise LabError(safe_error(exc, 'Mở Qdrant demo') +
                           ' Kiểm tra xem một tác vụ khác đang dùng cùng chỉ mục.') from None
        try:
            yield client
        finally:
            client.close()

    def verify(self, client):
        if client.collection_exists(COLLECTION):
            vector = client.get_collection(COLLECTION).config.params.vectors
            if (isinstance(vector, models.VectorParams) and vector.size == DIMENSIONS
                    and vector.distance == models.Distance.COSINE
                    and client.count(COLLECTION, exact=True).count == len(self.chunks)):
                return
        raise LabError('Chỉ mục vector demo chưa hoàn tất hoặc bị lỗi. Hãy tạo lại chỉ mục.')

    def build(self, provider, progress):
        with self.connect() as client:
            if self.ready():
                try:
                    self.verify(client)
                except LabError:
                    pass
                else:
                    return
            ready = self.path / 'ready.json'
            ready.unlink(missing_ok=True)
            if client.collection_exists(COLLECTION):
                client.delete_collection(COLLECTION)
            client.create_collection(COLLECTION, vectors_config=models.VectorParams(
                size=DIMENSIONS, distance=models.Distance.COSINE,
            ))
            rows = list(self.chunks.values())
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                vectors = provider.embed_documents([chunk['text'] for chunk in batch])
                client.upsert(COLLECTION, points=[
                    models.PointStruct(id=chunk['id'], vector=vector)
                    for chunk, vector in zip(batch, vectors, strict=True)
                ], wait=True)
                progress(start + len(batch), len(rows))
            self.verify(client)
            write_json(ready, self.manifest())

    def search(self, vector, filters, limit):
        with self.connect() as client:
            if not self.ready():
                raise LabError('Hãy tạo chỉ mục vector cho cấu hình cắt đang xem.')
            self.verify(client)
            points = client.query_points(
                COLLECTION, query=vector, limit=len(self.chunks), with_payload=False,
                search_params=models.SearchParams(exact=True),
            ).points
        rows = [{**self.chunks[str(point.id)], 'score': float(point.score)} for point in points]
        return sorted(rows, key=lambda row: (-row['score'], row['id']))[:limit]


def official_snapshot():
    path = DEMO_SNAPSHOT if DEMO_SNAPSHOT.is_file() else CHUNK_SNAPSHOT
    return load_snapshot(path)


def prepare_official(day06):
    provenance, chunks = rebuild(day06)
    write_json(DEMO_SNAPSHOT, {'provenance': provenance, 'chunks': chunks})


def official_store_error(day06, store):
    path = Path(store).expanduser() if store.strip() else Path(day06).expanduser() / DEFAULT_STORE
    if not all((path / name).is_file() for name in ('active.json', 'meta.json')):
        return 'Chọn thư mục Qdrant Day 06 có active.json và meta.json. Xem mục Day 06 trong demo/README.md.'
    return None


def run_query(question, filters, config, mode, corpus, chunks, *, lexical=None,
              vector_store=None, official=None):
    """Run one action with scoped resources and release Qdrant even on error."""
    start = perf_counter()
    needs_vector = config['method'] != 'lexical'
    provider = store = None
    result = {'question': question, 'filter': filters, 'configuration': config['id'],
              'method': config['method'], 'mode': mode, 'settings': config, 'corpus': corpus}
    try:
        if mode == 'agent' or needs_vector:
            provider = DemoProvider(corpus)
        if needs_vector and official:
            day06, store_path, provenance = official
            problem = official_store_error(day06, store_path)
            if problem:
                raise LabError(problem)
            store = ActiveStore(Path(day06).expanduser(),
                                Path(store_path).expanduser() if store_path.strip() else None,
                                provenance, chunks)
        elif needs_vector:
            if not vector_store or not vector_store.ready():
                raise LabError('Hãy tạo chỉ mục vector cho cấu hình cắt đang xem.')
            store = vector_store
        retriever = Retriever(chunks if lexical is None else (), store=store, provider=provider)
        if lexical is not None:
            retriever.lexical = lexical
        result.update(trace_query(question, filters, config, retriever, provider, PRICES, mode))
    except Exception as exc:
        result.update(status='error', error=safe_error(exc, 'Chuẩn bị truy vấn'), events=[],
                      usage=usage_report(provider.calls if provider else [], PRICES))
    finally:
        if isinstance(store, ActiveStore):
            store.close()
        if provider:
            provider.close()
    result['total_ms'] = (perf_counter() - start) * 1000
    return result


def build_upload_vectors(store, progress):
    start = perf_counter()
    provider = None
    try:
        provider = DemoProvider('the uploaded document')
        store.build(provider, progress)
        outcome = {'status': 'ready'}
    except Exception as exc:
        outcome = {'status': 'error', 'error': safe_error(exc, 'Tạo chỉ mục vector')}
    finally:
        if provider:
            provider.close()
    calls = provider.calls if provider else []
    usage = usage_report([{**call, 'kind': 'query_embedding'} for call in calls], PRICES)
    usage['calls'] = calls
    usage['document_embedding'] = usage.pop('query_embedding')
    return {'mode': 'indexing', 'method': 'dense', 'model': EMBEDDING_MODEL,
            'usage': usage, 'chunk_count': len(store.chunks), 'elapsed_ms': (perf_counter() - start) * 1000,
            **outcome}


def ratio(stat):
    return f"{stat['n']}/{stat['N']}" if stat['N'] else '—'


def benchmark_results():
    selected = read_json(RESULTS / 'selection.json')['chosen']
    rows = []
    for split in ('dev', 'test'):
        for path in sorted((RESULTS / split).glob('*/reviewed_metrics.json')):
            grades = read_json(path)
            ranking = read_json(path.with_name('summary.json'))['retrieval']
            mean = grades['grounded_answer']['mean']
            rows.append({'Split': split, 'Cấu hình': path.parent.name,
                         'Hit@3': ratio(ranking['evidence_hit_at_3']),
                         'Context có evidence': ratio(grades['context_evidence_rate']),
                         'Grounded / 2': '—' if mean is None else f'{mean:.2f}',
                         'Citation hợp lệ': ratio(grades['citation_validity'])})
    traces = {t['query_id']: t for t in read_json(RESULTS / 'dev' / selected['id'] / 'traces.json')
              if t['query_id'] in ('D01', 'DN01')}
    return selected, rows, traces
