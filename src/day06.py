"""Rebuild and verify the exact Day 6 B payloads, then open the active local store."""

import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

from starter.convert_day6 import convert_chunk

from .common import (DEFAULT_STORE, FAMILIES, ROOT, LabError, SetupUnavailable,
                     read_json, safe_error, sha256_file, sha256_json)

EMBEDDING_MODEL = 'text-embedding-3-large'
DIMENSIONS = 3072
CHUNK_COUNT = 264
SOURCE_UNITS = 45
MAX_CHARS = 500
DATA_FILES = ('corpus.jsonl', 'manifest.json', 'sources.json', 'PYTHON_DOC_LICENSE.txt')
REFERENCE = ROOT / 'benchmark' / 'day06_reference.json'

# A separate interpreter avoids importing the wrong package, since both repos use `src`.
REBUILD_SCRIPT = '''
import json
import sys
from dataclasses import asdict
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'starter'))
from src.chunking import by_paragraph
from src.ingestion import load_corpus
pages = load_corpus(root / 'data/manifest.json', root / 'data/corpus.jsonl')
setting = json.loads((root / 'benchmark/strategies.json').read_text(encoding='utf-8'))['strategies']['B']
chunks = [asdict(chunk) for page in pages.values() for chunk in by_paragraph(page, setting['max_chars'])]
print(json.dumps({'source_units': len(pages), 'chunks': chunks}))
'''


def git_day06(repo):
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(repo), 'status', '--short', '--branch'],
                                         text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        raise LabError('Pass the Day 6 Git checkout with --day06.') from None
    return commit, status[0]


def chunk_map(chunks):
    by_id = {}
    for chunk in chunks:
        try:
            chunk_id = chunk['id']
            metadata = chunk['metadata']
            start, end = metadata['start_char'], metadata['end_char']
            valid = (str(UUID(chunk_id)) == chunk_id and chunk_id not in by_id
                     and all(isinstance(chunk[key], str) and chunk[key].strip()
                             for key in ('source', 'source_id', 'page', 'text'))
                     and type(start) is int and type(end) is int and start >= 0
                     and 0 < end - start == len(chunk['text']) <= MAX_CHARS
                     and metadata['unit_url'] == f"{chunk['source']}#{chunk['page']}"
                     and metadata['api_family'] in FAMILIES)
        except (AttributeError, KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise LabError('Invalid Day 6 B payload (ID, source, anchor, API family or offsets).')
        by_id[chunk_id] = chunk
    if len(by_id) != CHUNK_COUNT:
        raise LabError(f'Expected {CHUNK_COUNT} unique B chunks, found {len(by_id)}.')
    return by_id


def rebuild(repo):
    repo = Path(repo).resolve()
    if not all((repo / 'data' / name).is_file() for name in DATA_FILES):
        raise LabError('Pass the Day 6 checkout containing the frozen corpus and manifest with --day06.')
    commit, branch = git_day06(repo)
    reference = read_json(REFERENCE)
    for name, digest in reference['data_sha256'].items():
        upstream = repo / 'data' / name
        # Day 6 does not specify a checkout line ending for the license text.
        upstream_digest = (hashlib.sha256(upstream.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                           if name.endswith('.txt') else sha256_file(upstream))
        if upstream_digest != digest or sha256_file(ROOT / 'data' / name) != digest:
            raise LabError(f'Day 6 data hash mismatch for {name}. Use the corpus specified in the lab.')
    signatures = reference['signatures']
    signed_files = {
        'benchmark/strategies.json': signatures['configuration'],
        'benchmark/openai_large_model.json': signatures['model_configuration'],
        **{f'starter/src/{name}': digest for name, digest in signatures['implementation'].items()},
        **reference['loader_sha256'],
    }
    for name, digest in signed_files.items():
        if sha256_file(repo / name) != digest:
            raise LabError(f'Day 6 implementation/configuration hash mismatch for {name}.')
    result = subprocess.run([sys.executable, '-I', '-c', REBUILD_SCRIPT, str(repo)],
                            capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise LabError('Day 6 could not rebuild B. Check the checkout and install Day 7 requirements.')
    rebuilt = json.loads(result.stdout)
    chunks = sorted(chunk_map([convert_chunk(chunk) for chunk in rebuilt['chunks']]).values(),
                    key=lambda chunk: chunk['id'])
    if rebuilt['source_units'] != SOURCE_UNITS:
        raise LabError(f'Expected {SOURCE_UNITS} Day 6 HTML source units.')
    digest = sha256_json(chunks)
    if digest != reference['chunks_sha256']:
        raise LabError('Rebuilt B payload hash differs from the Day 6 reference.')
    return {
        'status': 'corpus_rebuilt',
        'day06_commit': commit,
        'day06_branch': branch,
        'corpus_sha256': signatures['corpus'],
        'source_units': rebuilt['source_units'],
        'chunk_count': len(chunks),
        'chunks_sha256': digest,
        'signatures': signatures,
        'store': None,
    }, chunks


def load_snapshot(path):
    if not path.is_file():
        raise LabError('Chunk snapshot missing. Run prepare first.')
    snapshot = read_json(path)
    chunks = sorted(chunk_map(snapshot['chunks']).values(), key=lambda chunk: chunk['id'])
    provenance = snapshot['provenance']
    reference = read_json(REFERENCE)
    if (provenance['chunks_sha256'] != sha256_json(chunks)
            or provenance['chunks_sha256'] != reference['chunks_sha256']
            or provenance['signatures'] != reference['signatures']
            or provenance['corpus_sha256'] != reference['data_sha256']['corpus.jsonl']):
        raise LabError('Chunk snapshot differs from the frozen Day 6 B payloads. Run prepare again.')
    return provenance, chunks


def from_point(point):
    metadata = dict(point.payload or {})
    fields = {key: metadata.pop(key, None) for key in ('document_id', 'source', 'page', 'text')}
    return convert_chunk({'chunk_id': str(point.id), **fields, 'metadata': metadata})


class ActiveStore:
    def __init__(self, repo, path, expected, chunks):
        from qdrant_client import QdrantClient, models

        self.models = models
        path = (Path(path) if path else Path(repo) / DEFAULT_STORE).resolve()
        state_file = path / 'active.json'
        if not state_file.is_file() or not (path / 'meta.json').is_file():
            raise SetupUnavailable('Day 6 Qdrant store missing. Set --store to the directory with active.json and meta.json.')
        state = read_json(state_file)
        self.collection = state.get('indexes', {}).get('B')
        ingest_id = state.get('ingest_id')
        stats = state.get('statistics', {}).get('B', {})
        if (not isinstance(self.collection, str) or not self.collection
                or not isinstance(ingest_id, str) or not ingest_id
                or state.get('embedding_model') != EMBEDDING_MODEL
                or state.get('signatures') != expected['signatures']
                or stats.get('chunk_count') != CHUNK_COUNT):
            raise LabError('active.json does not describe the frozen OpenAI large collection B.')
        try:
            self.client = QdrantClient(path=str(path))
        except Exception as exc:
            raise LabError(safe_error(exc, 'Qdrant open') + ' Close Day 6 and other processes using this store.') from None
        try:
            if not self.client.collection_exists(self.collection):
                raise LabError('Collection B named in active.json is missing from the local Qdrant store.')
            schema = self.client.get_collection(self.collection).config.params.vectors
            count = self.client.count(self.collection, exact=True).count
            if (not isinstance(schema, models.VectorParams) or schema.size != DIMENSIONS
                    or schema.distance != models.Distance.COSINE or count != CHUNK_COUNT):
                raise LabError('Collection B must have 264 points and one cosine vector of size 3072 per point.')
            self.chunks = self._scroll()
            if chunk_map(self.chunks) != chunk_map(chunks):
                raise LabError('Collection B payloads or point IDs differ from the rebuilt corpus.')
            self.provenance = {
                'engine': 'qdrant-local', 'qdrant_client': version('qdrant-client'),
                'embedding_provider': 'openai', 'embedding_model': EMBEDDING_MODEL,
                'distance': 'cosine', 'dimensions': DIMENSIONS, 'point_count': count,
                'ingest_id': ingest_id, 'collection': self.collection,
                'ingest_started_at_utc': state.get('started_at_utc'),
                'ingest_finished_at_utc': state.get('finished_at_utc'),
                'B_ingest_seconds': stats.get('total_seconds'),
                'B_ingest_input_tokens': stats.get('api_input_tokens'),
                'B_ingest_usd': stats.get('paid_cost_usd'),
            }
        except BaseException:
            self.client.close()
            raise

    def close(self):
        self.client.close()

    def _scroll(self):
        rows = []
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, offset=offset, limit=64,
                                                with_payload=True, with_vectors=False)
            rows.extend(from_point(point) for point in points)
            if offset is None:
                return sorted(rows, key=lambda chunk: chunk['id'])

    def search(self, vector, filters, limit):
        models = self.models
        scope = models.Filter(must=[models.FieldCondition(
            key='api_family', match=models.MatchValue(value=filters['api_family']))]) if filters else None
        # Fetch all scoped cosine scores for deterministic ID tie-breaking at the cutoff.
        response = self.client.query_points(self.collection, query=vector, query_filter=scope,
                                            search_params=models.SearchParams(exact=True),
                                            limit=CHUNK_COUNT, with_payload=True, with_vectors=False)
        candidates = [{**from_point(point), 'score': float(point.score)} for point in response.points]
        return sorted(candidates, key=lambda chunk: (-chunk['score'], chunk['id']))[:limit]
