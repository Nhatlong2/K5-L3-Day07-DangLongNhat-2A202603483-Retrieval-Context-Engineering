"""OpenAI API calls and provider-reported token usage."""

import math
import os
from time import perf_counter

from .common import ROOT, LabError, SetupUnavailable, safe_error
from .day06 import DIMENSIONS, EMBEDDING_MODEL


class OpenAIProvider:
    def __init__(self):
        from openai import OpenAI

        key = os.getenv('OPENAI_API_KEY', '').strip()
        if not key:
            raise SetupUnavailable('OPENAI_API_KEY is missing. Set the environment variable or the local .env.')
        self.client = OpenAI(api_key=key, base_url='https://api.openai.com/v1', timeout=45, max_retries=1)
        self.calls = []
        self.expected_agent_model = None

    def close(self):
        self.client.close()

    def embed(self, query):
        import numpy as np

        start = perf_counter()
        try:
            response = self.client.embeddings.create(input=[query], model=EMBEDDING_MODEL,
                                                     dimensions=DIMENSIONS, encoding_format='float')
        except Exception as exc:
            raise LabError(safe_error(exc, 'OpenAI embeddings')) from None
        self.calls.append({'kind': 'query_embedding', 'model': response.model,
                           'input_tokens': response.usage.prompt_tokens,
                           'elapsed_ms': (perf_counter() - start) * 1000})
        if response.model != EMBEDDING_MODEL or len(response.data) != 1 or response.data[0].index != 0:
            raise LabError('OpenAI returned an unexpected query embedding model or index.')
        vector = response.data[0].embedding
        if len(vector) != DIMENSIONS or not all(map(math.isfinite, vector)):
            raise LabError('OpenAI returned an invalid query embedding.')
        array = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(array).all() or not np.any(array):
            raise LabError('OpenAI returned an invalid query embedding.')
        return array.tolist()

    def complete(self, model, messages, **options):
        start = perf_counter()
        try:
            response = self.client.chat.completions.create(model=model, messages=messages, store=False, **options)
        except Exception as exc:
            raise LabError(safe_error(exc, 'OpenAI agent')) from None
        usage = response.usage
        details = usage.prompt_tokens_details if usage else None
        output_details = usage.completion_tokens_details if usage else None
        self.calls.append({
            'kind': 'agent', 'model': response.model,
            'input_tokens': usage.prompt_tokens if usage else None,
            'cached_input_tokens': (details.cached_tokens or 0) if details else 0,
            'output_tokens': usage.completion_tokens if usage else None,
            'reasoning_tokens': (output_details.reasoning_tokens or 0) if output_details else 0,
            'elapsed_ms': (perf_counter() - start) * 1000,
        })
        if self.expected_agent_model and response.model != self.expected_agent_model:
            raise LabError('The API returned a different agent model from the frozen dev run.')
        if not response.choices:
            raise LabError('OpenAI returned no agent choice.')
        return response.choices[0]


def load_local_env():
    from dotenv import load_dotenv

    load_dotenv(ROOT / '.env', override=False, interpolate=False, encoding='utf-8-sig')


def usage_report(calls, prices):
    totals = {}
    for kind, price_key in (('query_embedding', 'embedding'), ('agent', 'agent')):
        rows = [row for row in calls if row['kind'] == kind]
        total = {'requests_with_response': len(rows), 'models': sorted({row['model'] for row in rows}),
                 'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0,
                 'estimated_usd': 0.0}
        for row in rows:
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens'):
                value = row.get(key, 0)
                total[key] = total[key] + value if total[key] is not None and value is not None else None
            rate = prices[price_key].get(row['model'])
            input_tokens, output_tokens = row['input_tokens'], row.get('output_tokens', 0)
            if not rate or input_tokens is None or output_tokens is None:
                total['estimated_usd'] = None
            elif total['estimated_usd'] is not None:
                cached = row.get('cached_input_tokens', 0)
                total['estimated_usd'] += ((input_tokens - cached) * rate['input']
                                           + cached * rate.get('cached_input', 0)
                                           + output_tokens * rate.get('output', 0)) / 1_000_000
        totals[kind] = total
    costs = [total['estimated_usd'] for total in totals.values()]
    return {**totals, 'total_usd': sum(costs) if None not in costs else None, 'calls': calls}
