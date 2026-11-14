"""Regression tests for Day 7. Synthetic API responses are test fixtures only."""

import json
import tempfile
import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import NAMESPACE_URL, uuid5

from src.agent import context_for, finish, run_agent
from src.common import LabError, sha256_json, write_json
from src.day06 import (CHUNK_COUNT, DIMENSIONS, EMBEDDING_MODEL, REFERENCE,
                       ActiveStore, from_point)
from src.experiment import choose, trace_query
from src.grading import evidence_hit, grade_agent, review_template
from src.provider import usage_report
from src.retrieval import Retriever, fuse
from starter.convert_day6 import convert_question


def chunk(cid, text, family='asyncio', rank=1):
    source = 'https://example.org/python.html'
    return {'id': cid, 'source_id': 'python', 'page': 'section', 'source': source, 'text': text,
            'metadata': {'api_family': family, 'unit_url': f'{source}#section',
                         'start_char': 0, 'end_char': len(text)}, 'score': 1.0, 'rank': rank}


def tool_response(query, call_id):
    values = {'type': 'function', 'id': call_id,
              'function': {'name': 'search_corpus', 'arguments': json.dumps({'query': query})}}
    call = SimpleNamespace(type='function', id=call_id, function=SimpleNamespace(**values['function']),
                           model_dump=lambda **_: values)
    return SimpleNamespace(finish_reason='tool_calls', message=SimpleNamespace(tool_calls=[call], refusal=None))


def final_response(answer, citations, status='answered'):
    content = json.dumps({'status': status, 'answer': answer, 'citations': citations})
    return SimpleNamespace(finish_reason='stop', message=SimpleNamespace(tool_calls=None, refusal=None, content=content))


CONFIG = {'id': 'lexical', 'method': 'lexical', 'agent_model': 'test-model', 'top_k': 3,
          'ranking_cutoff': 3, 'max_searches': 2, 'context_budget_chars': 120, 'max_completion_tokens': 600}
PRICES = {'embedding': {}, 'agent': {}}


class RetrievalTests(unittest.TestCase):
    def test_point_id_and_metadata(self):
        payload = {'document_id': 'docs', 'source': 'https://example.org', 'page': 'x', 'text': 'example',
                   'api_family': 'asyncio', 'start_char': 9, 'end_char': 16, 'unit_url': 'https://example.org#x'}
        point = SimpleNamespace(id='point-id', payload=payload)
        converted = from_point(point)
        self.assertEqual(converted['id'], 'point-id')
        self.assertEqual(converted['source_id'], 'docs')
        self.assertEqual(converted['page'], 'x')
        self.assertEqual(converted['metadata']['start_char'], 9)
        self.assertEqual(point.payload, payload)

    def test_convert_alternative_gold_and_normalization(self):
        raw = {'gold_source_id': 'python', 'gold_page': 'section', 'evidence_quote': 'café  new line',
               'alternative_gold': [{'gold_source_id': 'python', 'gold_page': 'other', 'evidence_quote': 'alternate'}]}
        gold = convert_question(raw)['gold']
        self.assertEqual(len(gold), 2)
        self.assertTrue(evidence_hit(chunk('a', 'cafe\u0301 new\nline'), gold))
        self.assertFalse(evidence_hit({**chunk('a', 'café new line'), 'page': 'wrong'}, gold))

    def test_rrf_dedup_and_id_tie_break(self):
        a, b, c = [chunk(cid, 'same keyword') for cid in 'abc']
        rows = fuse([b, c], [a, c], 60, 3)
        self.assertEqual([row['id'] for row in rows], ['c', 'a', 'b'])
        self.assertEqual(rows[0]['branch_ranks'], {'dense': 2, 'lexical': 2})
        self.assertAlmostEqual(rows[0]['score'], 2 / 62)
        self.assertEqual(rows[1]['rank'], 2)

    def test_filter_before_lexical_candidate_limit(self):
        index = Retriever([chunk('a', 'keyword', 'threading'), chunk('b', 'keyword test')])
        rows, _ = index.search('keyword', {'api_family': 'asyncio'}, CONFIG, 1)
        self.assertEqual([row['id'] for row in rows], ['b'])
        with self.assertRaises(LabError):
            index.search('keyword', {'document_id': 'python'}, CONFIG, 1)

    @unittest.skipUnless(find_spec('qdrant_client'), 'Install requirements to test local Qdrant')
    def test_active_store_pages_filters_and_checks_payload(self):
        from qdrant_client import QdrantClient, models

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            chunks = []
            for i in range(CHUNK_COUNT):
                cid = str(uuid5(NAMESPACE_URL, f'day07-test-{i}'))
                family = 'threading' if i % 2 else 'asyncio'
                row = chunk(cid, f'Fixture {i}', family)
                row.pop('score')
                row.pop('rank')
                chunks.append(row)
            ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
            write_json(path / 'active.json', {'indexes': {'B': 'fixture'}, 'ingest_id': 'fixture',
                                              'embedding_model': EMBEDDING_MODEL, 'signatures': ref['signatures'],
                                              'statistics': {'B': {'chunk_count': CHUNK_COUNT}}})
            vector = [1.0] + [0.0] * (DIMENSIONS - 1)
            client = QdrantClient(path=temp)
            try:
                client.create_collection('fixture', vectors_config=models.VectorParams(size=DIMENSIONS, distance='Cosine'))
                client.upsert('fixture', points=[models.PointStruct(
                    id=row['id'], vector=vector, payload={**row['metadata'], 'document_id': row['source_id'],
                                                          'page': row['page'], 'source': row['source'], 'text': row['text']})
                    for row in chunks])
            finally:
                client.close()
            store = ActiveStore(path, path, ref, chunks)
            try:
                self.assertEqual(len(store.chunks), CHUNK_COUNT)
                ids = sorted(row['id'] for row in chunks if row['metadata']['api_family'] == 'asyncio')
                self.assertEqual([r['id'] for r in store.search(vector, {'api_family': 'asyncio'}, 3)], ids[:3])
            finally:
                store.close()
            with self.assertRaisesRegex(LabError, 'payloads or point IDs differ'):
                ActiveStore(path, path, ref, [{**chunks[0], 'text': 'Fixture !'}, *chunks[1:]])


class AgentTests(unittest.TestCase):
    def test_second_search_can_replace_context_and_respects_tool_protocol(self):
        provider, retriever = Mock(), Mock()
        scope = {'api_family': 'asyncio'}
        calls = []
        responses = iter([tool_response('model rewrite before search', 'first'), tool_response('refinement', 'second'),
                          final_response('The answer is supported [b].', ['b'])])

        def complete(model, messages, **kwargs):
            calls.append((deepcopy(messages), kwargs['tool_choice']))
            return next(responses)

        provider.complete.side_effect = complete
        old, new = chunk('a', 'x' * 50), chunk('b', 'y' * 50)
        retriever.search.return_value = ([new, new], {})
        outcome = run_agent('original question', scope, CONFIG, retriever, provider, [old], {})
        self.assertEqual(outcome['status'], 'answered')
        self.assertEqual(outcome['searches'], 2)
        self.assertEqual(outcome['context']['ids'], ['b'])
        self.assertEqual(outcome['citation_sources'][0]['unit_url'], new['metadata']['unit_url'])
        self.assertEqual(outcome['events'][1]['query'], 'original question')
        self.assertEqual([choice if isinstance(choice, str) else choice['type'] for _, choice in calls],
                         ['function', 'auto', 'none'])
        self.assertEqual(calls[-1][0][-1]['content'], outcome['context']['text'])
        self.assertNotIn(old['text'], json.dumps(calls[-1][0]))
        self.assertLessEqual(outcome['context']['chars'], CONFIG['context_budget_chars'])
        retriever.search.assert_called_once_with('refinement', scope, CONFIG, 3)

    def test_invalid_or_out_of_context_inline_citation_abstains(self):
        packed = context_for([[chunk('a', 'evidence')]], 120)
        for answer, citations in [('Claim [a] [outside]', ['a']), ('Claim [outside]', ['outside']),
                                  ('Claim', []), ('Claim [a]', ['a', 'a'])]:
            with self.subTest(answer=answer, citations=citations):
                outcome = finish(json.dumps(dict(status='answered', answer=answer, citations=citations)), packed)
                self.assertEqual(outcome['status'], 'abstain')
                self.assertEqual(outcome['citations'], [])
        self.assertEqual(context_for([[chunk('a', 'evidence')]], 1)['ids'], [])

    def test_agent_request_contains_only_runtime_fields_and_keeps_error_sanitized(self):
        provider, retriever = Mock(), Mock()
        provider.calls = []
        provider.complete.side_effect = [tool_response('question', 'first'),
                                         RuntimeError('provider body with secret-example')]
        retriever.search.return_value = ([chunk('a', 'evidence')], {})
        trace = trace_query('question', {'api_family': 'asyncio'}, CONFIG, retriever, provider, PRICES, 'agent')
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['ranking_status'], 'ok')
        self.assertNotIn('secret-example', json.dumps(trace))
        request = provider.complete.call_args_list[0].args[1]
        self.assertEqual(json.loads(request[1]['content']), {'question': 'question', 'scope': {'api_family': 'asyncio'}})


class GradingTests(unittest.TestCase):
    def test_reviews_are_bound_to_trace_and_citation_support(self):
        cases = [{'id': 'q', 'answerable': True}]
        trace = {'query_id': 'q', 'status': 'answered', 'citations': ['a'], 'searches': 1,
                 'context': {'ids': ['a']}, 'events': [{'event': 'search'}]}
        reviews = review_template(cases, [trace])
        review = reviews[0]
        self.assertEqual(review['trace_sha256'], sha256_json(trace))
        with self.assertRaises(LabError):
            grade_agent(cases, [trace], reviews)
        review.update(reviewer='fixture reviewer', review_state='ai_assisted_unverified', reason='partial answer',
                      context_grade=2, answer_grade=1,
                      citation_reviews=[{'id': 'a', 'supports_claim': False, 'reason': 'different claim'}])
        metrics = grade_agent(cases, [trace], reviews)
        self.assertEqual(metrics['citation_validity'], {'n': 0, 'N': 1, 'ratio': 0})
        self.assertEqual(metrics['grounded_answer']['mean'], 1)
        self.assertEqual(metrics['context_evidence_rate']['ratio'], 1)
        with self.assertRaises(LabError):
            grade_agent(cases, [{**trace, 'searches': 2}], reviews)

    def test_unknown_price_does_not_win_a_cost_tie(self):
        self.assertIsNone(usage_report([{'kind': 'agent', 'model': 'unpriced', 'input_tokens': 1,
                                        'output_tokens': 2}], PRICES)['total_usd'])
        review = {'tool_use': {'N': 10}, 'grounded_answer': {'mean': 1.5}, 'citation_validity': {'ratio': 1},
                  'context_evidence_rate': {'ratio': .9}}
        summary = {'retrieval': {'evidence_hit_at_3': {'ratio': .8}},
                   'successful_api_cost_usd': .02, 'median_success_ms': 100}
        rows = {'a': (summary, review, {}), 'b': ({**summary, 'successful_api_cost_usd': None,
                                                  'median_success_ms': 500}, review, {})}
        self.assertEqual(choose(rows), 'a')


if __name__ == '__main__':
    unittest.main()
