"""Day 7 command line. Run from the repository root with python -m src.run."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

from .common import (CHUNK_SNAPSHOT, DEFAULT_DAY06, WORK, LabError, now, read_json,
                     safe_error, sha256_file, write_json)
from .day06 import ActiveStore, rebuild, load_snapshot
from .experiment import (BENCHMARK, PRICES, freeze_selection, open_runtime,
                         review_run, run_benchmark, settings, trace_query)
from .grading import validate_cases, validate_disjoint
from .provider import load_local_env
from .retrieval import Retriever


def add_paths(parser):
    parser.add_argument('--day06', type=Path, default=DEFAULT_DAY06, help='Day 6 Git checkout')
    parser.add_argument('--store', type=Path, help='Day 6 Qdrant directory (defaults to the OpenAI large store)')
    parser.add_argument('--chunks', type=Path, default=CHUNK_SNAPSHOT, help='Local JSON payload snapshot')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Rebuild and optionally verify the Day 6 B snapshot')
    add_paths(prepare)
    prepare.add_argument('--offline', action='store_true', help='Rebuild payloads without opening Qdrant')
    validate = commands.add_parser('validate', help='Verify splits, scope and evidence against B')
    validate.add_argument('--chunks', type=Path, default=CHUNK_SNAPSHOT)
    validate.add_argument('--output', type=Path, default=WORK / 'benchmark_validation.json')
    bench = commands.add_parser('benchmark', help='Run a split and save per-query traces')
    add_paths(bench)
    bench.add_argument('--split', choices=('dev', 'test'), required=True)
    bench.add_argument('--mode', choices=('retrieval', 'agent'), default='agent')
    bench.add_argument('--configs', nargs='+', help='Configuration IDs from benchmark/experiments.json')
    bench.add_argument('--output', type=Path, default=WORK / 'runs')
    bench.add_argument('--selection', type=Path, default=WORK / 'selection.json')
    grade = commands.add_parser('grade', help='Aggregate manually reviewed answers and context')
    grade.add_argument('--run', type=Path, required=True, help='A configuration result directory with reviews.json')
    select = commands.add_parser('select', help='Freeze the winner using graded dev results')
    select.add_argument('--output', type=Path, default=WORK / 'runs')
    select.add_argument('--selection', type=Path, default=WORK / 'selection.json')
    ask = commands.add_parser('ask', help='Ask the real agent a question')
    add_paths(ask)
    ask.add_argument('--config', default='dense')
    ask.add_argument('--api-family')
    ask.add_argument('--output', type=Path)
    ask.add_argument('question')
    return parser.parse_args()


def prepare(args):
    if args.offline and args.store:
        raise LabError('Use --offline for reconstruction or --store to verify a local store.')
    provenance, chunks = rebuild(args.day06)
    if not args.offline:
        with ExitStack() as stack:
            store = ActiveStore(args.day06, args.store, provenance, chunks)
            stack.callback(store.close)
            provenance = {**provenance, 'status': 'store_verified', 'store': store.provenance}
            chunks = store.chunks
    write_json(args.chunks, {'provenance': provenance, 'chunks': chunks})
    print(f"{provenance['status']}: {len(chunks)} unique B chunks, SHA-256 {provenance['chunks_sha256']}")


def validate(args):
    provenance, chunks = load_snapshot(args.chunks)
    cases = {split: read_json(BENCHMARK / f'{split}.json') for split in ('dev', 'test')}
    checks = {split: validate_cases(rows, chunks, split) for split, rows in cases.items()}
    validate_disjoint(cases['dev'], cases['test'])
    write_json(args.output, {'checked_at_utc': now(), 'provenance': provenance,
                             'benchmark_sha256': {split: sha256_file(BENCHMARK / f'{split}.json') for split in cases},
                             'checks': checks})
    print('Validated dev and test source anchors, scopes, counts and evidence.')


def ask(args):
    _, configs = settings()
    if args.config not in configs or not args.question.strip():
        raise LabError('Pass a known configuration and a nonempty question.')
    config = configs[args.config]
    filters = {'api_family': args.api_family} if args.api_family else None
    with ExitStack() as stack:
        provenance, chunks, store, provider = open_runtime(stack, args, config, 'agent')
        retriever = Retriever(chunks, store, provider)
        trace = trace_query(args.question, filters, config, retriever, provider, read_json(PRICES), 'agent')
    if args.output:
        write_json(args.output, {**trace, 'provenance': provenance, 'settings': config})
    if trace['status'] == 'error':
        raise LabError(trace['error'])
    print(trace['answer'])
    for row in trace['citation_sources']:
        print(f"[{row['id']}] {row['unit_url']}")


def main():
    args = parse_args()
    try:
        load_local_env()
        if args.command == 'prepare':
            prepare(args)
        elif args.command == 'validate':
            validate(args)
        elif args.command == 'benchmark':
            return run_benchmark(args)
        elif args.command == 'grade':
            _, metrics, _ = review_run(args.run)
            print(f"Reviewed {metrics['tool_use']['N']} successful queries.")
        elif args.command == 'select':
            record = freeze_selection(args)
            print(f"Selected {record['chosen']['id']}, SHA-256 {record['selection_sha256']}")
        else:
            ask(args)
    except Exception as exc:
        print(safe_error(exc, args.command), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
