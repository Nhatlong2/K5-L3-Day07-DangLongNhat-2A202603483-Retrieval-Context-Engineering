"""Run comparable configurations and freeze the dev selection before test."""

import os
import platform
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .agent import context_for, run_agent
from .common import (ROOT, LabError, SetupUnavailable, now, read_json, safe_error, sha256_file,
                     sha256_json, validate_scope, write_json)
from .day06 import ActiveStore, load_snapshot
from .grading import grade_agent, save_run, validate_cases
from .provider import OpenAIProvider, usage_report
from .retrieval import Retriever

BENCHMARK = ROOT / 'benchmark'
EXPERIMENTS = BENCHMARK / 'experiments.json'
PRICES = BENCHMARK / 'prices.json'


def inputs():
    files = [*sorted((ROOT / 'src').glob('*.py')), ROOT / 'starter' / 'baseline.py',
             ROOT / 'starter' / 'agent_metrics.py', ROOT / 'starter' / 'convert_day6.py',
             ROOT / 'requirements.txt', EXPERIMENTS, PRICES, BENCHMARK / 'day06_reference.json']
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}


def package_versions():
    return {'python': platform.python_version(), 'platform': platform.system(),
            **{name: version(name) for name in ('openai', 'qdrant-client', 'numpy', 'pypdf', 'python-dotenv', 'httpx')}}


def settings():
    plan = read_json(EXPERIMENTS)
    model = os.getenv('OPENAI_AGENT_MODEL', '').strip() or plan['agent_model']
    shared = {key: plan[key] for key in ('max_searches', 'ranking_cutoff', 'rrf_candidates', 'rrf_k', 'max_completion_tokens')}
    configs = {name: {**shared, **variant, 'agent_model': model, 'id': name}
               for name, variant in plan['variants'].items()}
    for cfg in configs.values():
        if (cfg['method'] not in ('dense', 'lexical', 'hybrid')
                or any(type(cfg[key]) is not int or cfg[key] < 1 for key in (*shared, 'top_k', 'context_budget_chars'))
                or cfg['ranking_cutoff'] != 3
                or (cfg['method'] == 'hybrid'
                    and cfg['rrf_candidates'] < max(cfg['top_k'], cfg['ranking_cutoff']))):
            raise LabError('Invalid benchmark experiment settings.')
    if plan['baseline'] not in configs or configs[plan['baseline']]['method'] != 'dense':
        raise LabError('The baseline configuration must use dense retrieval.')
    return plan, configs


def trace_query(question, filters, config, retriever, provider, prices, mode):
    # Runtime receives only the actual query, allowed scope and settings.
    filters = validate_scope(filters)
    start = perf_counter()
    prior_calls = len(provider.calls) if provider else 0
    limit = max(config['top_k'], config['ranking_cutoff'])
    trace = {'question': question, 'filter': filters, 'configuration': config['id'],
             'method': config['method'], 'mode': mode, 'original_top3': [], 'original_candidates': [],
             'ranking_status': 'error', 'ranking_ms': None, 'events': [], 'context': None,
             'citation_sources': [], 'citations': [], 'answer': None, 'searches': 0}
    try:
        candidates, ranking_details = retriever.search(question, filters, config, limit)
        trace.update({'ranking_status': 'ok', 'original_candidates': candidates,
                      'original_top3': candidates[:config['ranking_cutoff']],
                      'ranking_ms': (perf_counter() - start) * 1000, **ranking_details})
        if mode == 'agent':
            trace.update(run_agent(question, filters, config, retriever, provider, candidates, ranking_details))
        else:
            trace.update({'status': 'retrieved', 'packed_offline': context_for(
                [candidates[:config['top_k']]], config['context_budget_chars'])})
    except Exception as exc:
        trace.update({'status': 'error', 'error': safe_error(exc, 'Query')})
    trace['usage'] = usage_report(provider.calls[prior_calls:] if provider else [], prices)
    trace['elapsed_ms'] = (perf_counter() - start) * 1000
    return trace


def open_runtime(stack, args, config, mode):
    provenance, chunks = load_snapshot(args.chunks)
    store = provider = None
    needs_api = mode == 'agent' or config['method'] != 'lexical'
    if needs_api:
        provider = OpenAIProvider()
        stack.callback(provider.close)
        store = ActiveStore(args.day06, args.store, provenance, chunks)
        stack.callback(store.close)
        provenance = {**provenance, 'status': 'store_verified', 'store': store.provenance}
        chunks = store.chunks
    else:
        provenance = {**provenance, 'status': 'corpus_rebuilt', 'store': None}
    return provenance, chunks, store, provider


def benchmark_runtime(stack, args, config):
    try:
        return open_runtime(stack, args, config, args.mode)
    except LabError as exc:
        label = 'not_run' if isinstance(exc, SetupUnavailable) else 'error'
        write_json(args.output / args.split / 'setup_error.json', {
            'split': args.split, 'mode': args.mode, 'status': label, 'time_utc': now(),
            'configuration': config, 'error': str(exc),
        })
        raise


def read_selection(path, plan, configs, provenance):
    if not path.is_file():
        raise LabError('No frozen dev selection. Complete dev reviews and run select before test.')
    record = read_json(path)
    saved = record.pop('selection_sha256', None)
    if (saved != sha256_json(record) or record['inputs_sha256'] != inputs()
            or record['chosen_config_sha256'] != sha256_json(record['chosen'])
            or record['benchmark_sha256'] != {split: sha256_file(BENCHMARK / f'{split}.json') for split in ('dev', 'test')}
            or record['store'] != provenance['store'] or record['chunks_sha256'] != provenance['chunks_sha256']
            or record['baseline'] != configs[plan['baseline']]
            or record['chosen'] != configs.get(record['chosen']['id'])):
        raise LabError('The frozen selection differs from the code, data, settings or store. Repeat dev selection.')
    return saved, list(dict.fromkeys([record['baseline']['id'], record['chosen']['id']])), record['agent_model_observed']


def run_benchmark(args):
    plan, configs = settings()
    if args.split == 'test' and (args.mode != 'agent' or args.configs):
        raise LabError('Test uses the baseline and frozen configuration with --mode agent.')
    names = args.configs or (list(configs) if args.mode == 'agent'
                            else [name for name, config in configs.items() if config['method'] == 'lexical'])
    if len(set(names)) != len(names) or any(name not in configs for name in names):
        raise LabError('Pass unique configuration names from benchmark/experiments.json.')
    setup = configs[plan['baseline']] if args.mode == 'agent' else configs[next(
        (name for name in names if configs[name]['method'] != 'lexical'), names[0])]
    prices = read_json(PRICES)
    start = perf_counter()
    with ExitStack() as stack:
        provenance, chunks, store, provider = benchmark_runtime(stack, args, setup)
        selection_sha = None
        if args.split == 'test':
            selection_sha, names, provider.expected_agent_model = read_selection(args.selection, plan, configs, provenance)
        elif args.mode == 'agent' and args.selection.exists():
            raise LabError('This experiment is already frozen. Use a new output and selection path for more dev runs.')
        out = args.output / args.split
        for name in names:
            if (out / name).exists():
                raise LabError(f'Results already exist for {args.split}/{name}. Use a fresh output directory.')
        case_path = BENCHMARK / f'{args.split}.json'
        cases = read_json(case_path)
        validate_cases(cases, chunks, args.split)
        retriever = Retriever(chunks, store, provider)
        setup_ms = (perf_counter() - start) * 1000
        shared = {'split': args.split, 'mode': args.mode, 'benchmark_sha256': sha256_file(case_path),
                  'inputs_sha256': inputs(), 'provenance': provenance, 'versions': package_versions(),
                  'price_table': prices, 'shared_setup_ms': setup_ms, 'selection_sha256': selection_sha}
        total_errors = 0
        for name in names:
            config = configs[name]
            started_at = now()
            traces = []
            for case in cases:
                trace = trace_query(case['question'], case.get('filter'), config, retriever, provider, prices, args.mode)
                traces.append({'query_id': case['id'], 'split': args.split, **trace})
                # Completed-case usage belongs to its trace and can be released.
                if provider:
                    provider.calls.clear()
            summary = save_run(out / name, cases, traces,
                               {**shared, 'configuration': config, 'configuration_sha256': sha256_json(config),
                                'started_at_utc': started_at, 'finished_at_utc': now()}, prices)
            total_errors += summary['status_counts'].get('error', 0)
            print(f"{args.split}/{name}: {summary['status_counts']}", flush=True)
    return 2 if total_errors else 0


def review_run(path):
    summary = read_json(path / 'summary.json')
    if summary['mode'] != 'agent':
        raise LabError('Only a real agent run takes reviewed context and answer labels.')
    traces = read_json(path / 'traces.json')
    case_path = BENCHMARK / f"{summary['split']}.json"
    cases = read_json(case_path)
    if (summary['benchmark_sha256'] != sha256_file(case_path)
            or summary['traces_sha256'] != sha256_json(traces)
            or [trace['query_id'] for trace in traces] != [case['id'] for case in cases]):
        raise LabError('Run trace does not match the benchmark split.')
    if not (path / 'reviews.json').is_file():
        raise LabError(f"Fill {path.name}/reviews.json from reviews.template.json before grading.")
    reviews = read_json(path / 'reviews.json')
    metrics = grade_agent(cases, traces, reviews)
    write_json(path / 'reviewed_metrics.json', {'split': summary['split'], 'configuration': summary['configuration'],
                                               'traces_sha256': sha256_json(traces), **metrics})
    return summary, metrics, {'traces_sha256': sha256_json(traces), 'reviews_sha256': sha256_json(reviews)}


def choose(rows):
    """Lexicographic choice with absent metrics sorted below measured values."""
    def metric(value):
        return float('-inf') if value is None else value

    compare_cost = all(summary['successful_api_cost_usd'] is not None for summary, *_ in rows.values())

    def key(name):
        summary, review, _ = rows[name]
        return (-review['tool_use']['N'], -metric(review['grounded_answer']['mean']),
                -metric(review['citation_validity']['ratio']), -metric(review['context_evidence_rate']['ratio']),
                -metric(summary['retrieval']['evidence_hit_at_3']['ratio']),
                summary['successful_api_cost_usd'] if compare_cost else 0,
                float('inf') if summary['median_success_ms'] is None else summary['median_success_ms'], name)

    return min(rows, key=key)


def freeze_selection(args):
    if args.selection.exists():
        raise LabError('Selection already exists. Preserve the existing experiment or use a fresh selection path.')
    plan, configs = settings()
    expected_inputs = inputs()
    dev_sha = sha256_file(BENCHMARK / 'dev.json')
    rows = {}
    for name, config in configs.items():
        path = args.output / 'dev' / name
        if not (path / 'summary.json').is_file():
            raise LabError(f'Run the real dev benchmark for {name} before selecting.')
        summary, metrics, digests = review_run(path)
        if (summary['split'] != 'dev' or summary['inputs_sha256'] != expected_inputs
                or summary['configuration'] != config or summary['benchmark_sha256'] != dev_sha):
            raise LabError(f'{name}: rerun dev with the current code and benchmark settings.')
        rows[name] = (summary, metrics, digests)
    provenance = rows[plan['baseline']][0]['provenance']
    if (not provenance['store']
            or any(row[0]['provenance']['store'] != provenance['store']
                   or row[0]['provenance']['chunks_sha256'] != provenance['chunks_sha256'] for row in rows.values())):
        raise LabError('All dev runs must share the same verified Day 6 collection.')
    if not any(metrics['tool_use']['N'] for _, metrics, _ in rows.values()):
        raise LabError('No successful dev agent queries to compare.')
    agent_models = {model for summary, *_ in rows.values() for model in summary['usage']['agent']['models']}
    if len(agent_models) != 1:
        raise LabError('Dev returned multiple agent model IDs. Use one fixed model snapshot for comparison.')
    winner = choose(rows)
    record = {'selected_at_utc': now(), 'selection_order': plan['selection_order'], 'inputs_sha256': expected_inputs,
              'benchmark_sha256': {'dev': dev_sha, 'test': sha256_file(BENCHMARK / 'test.json')},
              'baseline': configs[plan['baseline']], 'chosen': configs[winner],
              'agent_model_observed': agent_models.pop(),
              'chosen_config_sha256': sha256_json(configs[winner]), 'store': provenance['store'],
              'chunks_sha256': provenance['chunks_sha256'],
              'dev': {name: {**digests, 'reviewers': metrics['reviewers'], 'successes': metrics['tool_use']['N'],
                             'status_counts': summary['status_counts'], 'grounded_mean': metrics['grounded_answer']['mean'],
                             'citation_validity': metrics['citation_validity'], 'context_evidence_rate': metrics['context_evidence_rate'],
                             'evidence_hit_at_3': summary['retrieval']['evidence_hit_at_3'],
                             'successful_api_cost_usd': summary['successful_api_cost_usd'],
                             'median_success_ms': summary['median_success_ms']}
                      for name, (summary, metrics, digests) in rows.items()}}
    record['selection_sha256'] = sha256_json(record)
    write_json(args.selection, record)
    return record
