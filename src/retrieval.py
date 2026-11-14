"""Dense, TF-IDF and reciprocal rank fusion over one set of Day 6 B chunks."""

from starter.baseline import TfidfIndex

from .common import LabError, safe_error, validate_scope


def ranked(rows):
    return [{**row, 'rank': rank} for rank, row in enumerate(rows, start=1)]


def fuse(dense, lexical, k, limit):
    by_id = {}
    for method, rows in (('dense', dense), ('lexical', lexical)):
        for rank, row in enumerate(rows, start=1):
            chunk = by_id.setdefault(row['id'], {**row, 'branch_ranks': {'dense': None, 'lexical': None},
                                                'branch_scores': {'dense': None, 'lexical': None}})
            chunk['branch_ranks'][method] = rank
            chunk['branch_scores'][method] = row['score']
    for chunk in by_id.values():
        chunk['score'] = sum(1 / (k + rank) for rank in chunk['branch_ranks'].values() if rank is not None)
        chunk['strategy'] = 'hybrid'
    rows = sorted(by_id.values(), key=lambda chunk: (-chunk['score'], chunk['id']))
    return ranked(rows[:limit])


class Retriever:
    def __init__(self, chunks, store=None, provider=None):
        self.lexical = TfidfIndex(chunks)
        self.store = store
        self.provider = provider

    def search(self, query, filters, config, limit):
        filters = validate_scope(filters)
        method = config['method']
        if method == 'lexical':
            return ranked(self.lexical.search(query, limit, filters=filters)), {}
        if not self.store or not self.provider:
            raise LabError('Dense and hybrid require the verified Day 6 store and OpenAI embeddings.')
        vector = self.provider.embed(query)
        pool = config['rrf_candidates'] if method == 'hybrid' else limit
        try:
            dense = self.store.search(vector, filters, pool)
        except Exception as exc:
            raise LabError(safe_error(exc, 'Qdrant exact cosine query')) from None
        if method == 'dense':
            return ranked([{**row, 'strategy': 'dense'} for row in dense]), {}
        lexical = self.lexical.search(query, pool, filters=filters)
        fused = fuse(dense, lexical, config['rrf_k'], len(dense) + len(lexical))
        # Include omitted candidates without duplicating their text in the trace.
        ranks = [{key: row[key] for key in ('id', 'rank', 'score', 'branch_ranks', 'branch_scores')} for row in fused]
        return fused[:limit], {'rrf_ranking': ranks}
