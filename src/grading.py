"""Benchmark validation, source-aware evidence metrics and reviewed answer grades."""

from collections import Counter, defaultdict
from statistics import mean, median

from starter.agent_metrics import aggregate

from .common import LabError, normalize, sha256_json, validate_scope, write_json
from .provider import usage_report


def evidence_hit(chunk, gold):
    return any(chunk['source_id'] == row['source_id'] and chunk['page'] == row['page']
               and normalize(row['evidence']) in normalize(chunk['text']) for row in gold)


def validate_cases(cases, chunks, split):
    if not isinstance(cases, list):
        raise LabError('Benchmark must be a JSON array.')
    ids, questions = set(), set()
    counts = Counter()
    matches = []
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk['source_id'], chunk['page']].append(chunk)
    for case in cases:
        cid, question = case.get('id'), case.get('question')
        if (case.get('split') != split or not isinstance(cid, str) or not cid.strip() or cid in ids
                or not isinstance(question, str) or not question.strip() or type(case.get('answerable')) is not bool
                or normalize(question).casefold() in questions):
            raise LabError(f'{split}: invalid or duplicate case/question.')
        ids.add(cid)
        questions.add(normalize(question).casefold())
        scope = validate_scope(case.get('filter'))
        if scope and case.get('filter_context') != f"Requested API family: {scope['api_family']}":
            raise LabError(f'{cid}: missing pre-query API family context.')
        counts[case['answerable']] += 1
        if not case['answerable']:
            if (case.get('expected_behavior') != 'search_and_abstain' or not case.get('absence_reason', '').strip()
                    or case.get('gold') or case.get('expected_answer')):
                raise LabError(f'{cid}: invalid abstention case.')
            continue
        gold = case.get('gold')
        if not isinstance(gold, list) or not gold or not case.get('expected_answer', '').strip():
            raise LabError(f'{cid}: answerable case is missing evidence or expected answer.')
        for item in gold:
            if (not isinstance(item, dict) or set(item) != {'source_id', 'page', 'evidence'}
                    or not all(isinstance(value, str) and value.strip() for value in item.values())):
                raise LabError(f'{cid}: invalid gold evidence schema.')
            hits = [chunk['id'] for chunk in by_source[item['source_id'], item['page']]
                    if evidence_hit(chunk, [item]) and (not scope or chunk['metadata']['api_family'] == scope['api_family'])]
            if not hits:
                raise LabError(f'{cid}: evidence missing from a single chunk with matching source, anchor and scope.')
            matches.append({'query_id': cid, 'source_id': item['source_id'], 'page': item['page'], 'chunk_ids': hits})
    if counts[True] < (15 if split == 'dev' else 5) or counts[False] < 2:
        raise LabError(f'{split}: need at least 15/5 answerable and 2 unanswerable cases.')
    return {'split': split, 'answerable': counts[True], 'unanswerable': counts[False], 'gold_matches': matches}


def validate_disjoint(dev, test):
    for key in ('id', 'question'):
        if {normalize(case[key]).casefold() for case in dev} & {normalize(case[key]).casefold() for case in test}:
            raise LabError(f'Dev and test share a case {key}.')

    def evidence_set(cases):
        return {(item['source_id'], item['page'], normalize(item['evidence']))
                for case in cases for item in case.get('gold', [])}
    if evidence_set(dev) & evidence_set(test):
        raise LabError('Dev and test reuse the same gold excerpt.')


def metric(n, N):
    return {'n': n, 'N': N, 'ratio': n / N if N else None}


def score_retrieval(cases, traces):
    by_id = {trace['query_id']: trace for trace in traces}
    rows = []
    for case in cases:
        if not case['answerable']:
            continue
        trace = by_id[case['id']]
        if trace['ranking_status'] == 'error':
            continue
        ranking = trace['original_top3']
        hits = [row['id'] for row in ranking if evidence_hit(row, case['gold'])]
        context = trace.get('context') or trace.get('packed_offline') or {}
        context_hits = [row['id'] for row in context.get('chunks', []) if evidence_hit(row, case['gold'])]
        tool_hits = None
        if trace['status'] in ('answered', 'abstain'):
            tool_hits = sorted({row['id'] for event in trace['events'] if event['event'] == 'search'
                                for row in event['candidates'] if evidence_hit(row, case['gold'])})
        rows.append({'query_id': case['id'], 'hit1': bool(ranking and ranking[0]['id'] in hits),
                     'hit3': bool(hits), 'top3_hit_ids': hits, 'context_gold_ids': context_hits,
                     'packed_context_hit': bool(context_hits), 'context_measured': bool(context),
                     'agent_context_sent': trace.get('context') is not None, 'agent_tool_gold_ids': tool_hits})
    contexts = [row for row in rows if row['context_measured']]
    tool_rows = [row for row in rows if row['agent_tool_gold_ids'] is not None]
    return {
        'hit_at_1': metric(sum(row['hit1'] for row in rows), len(rows)),
        'evidence_hit_at_3': metric(sum(row['hit3'] for row in rows), len(rows)),
        'agent_tool_candidate_evidence_hit': metric(sum(bool(row['agent_tool_gold_ids']) for row in tool_rows),
                                                   len(tool_rows)),
        'automatic_context_gold_match': metric(sum(row['packed_context_hit'] for row in contexts), len(contexts)),
        'rows': rows,
    }


def review_template(cases, traces):
    by_id = {case['id']: case for case in cases}
    return [{
        'query_id': trace['query_id'], 'trace_sha256': sha256_json(trace), 'reviewer': '',
        'review_state': 'pending', 'reason': '',
        **({'context_grade': None, 'answer_grade': None} if by_id[trace['query_id']]['answerable']
           else {'abstention_correct': None}),
        'citation_reviews': [{'id': cid, 'supports_claim': None, 'reason': ''} for cid in trace['citations']],
    } for trace in traces if trace['status'] in ('answered', 'abstain')]


def grade_agent(cases, traces, reviews):
    by_id = {case['id']: case for case in cases}
    review_by_id = {review['query_id']: review for review in reviews}
    expected = {trace['query_id'] for trace in traces if trace['status'] != 'error'}
    if set(review_by_id) != expected or len(review_by_id) != len(reviews):
        raise LabError('Provide exactly one review for every successful query.')
    prepared = []
    context_grades = Counter()
    reviewers = set()
    for trace in traces:
        cid = trace['query_id']
        if trace['status'] == 'error':
            continue
        review = review_by_id[cid]
        if (not review.get('reviewer', '').strip() or not review.get('reason', '').strip()
                or review.get('trace_sha256') != sha256_json(trace)
                or review.get('review_state') not in ('human_reviewed', 'ai_assisted_unverified')):
            raise LabError(f'{cid}: fill reviewer, review_state and reason on the matching trace review.')
        reviewers.add((review['reviewer'], review['review_state']))
        if by_id[cid]['answerable']:
            grade = review.get('context_grade')
            if type(grade) is not int or grade not in (0, 1, 2):
                raise LabError(f'{cid}: provide a context grade between 0 and 2.')
            if grade and not (trace.get('context') or {}).get('ids'):
                raise LabError(f'{cid}: a context grade above zero requires evidence sent to the model.')
            context_grades[grade] += 1
        citation_reviews = review.get('citation_reviews')
        if (not isinstance(citation_reviews, list) or [row.get('id') for row in citation_reviews] != trace['citations']
                or any(type(row.get('supports_claim')) is not bool or not row.get('reason', '').strip()
                       for row in citation_reviews)):
            raise LabError(f'{cid}: provide support and reason for every citation in order.')
        context_ids = set((trace.get('context') or {}).get('ids', []))
        prepared.append({**review, 'citation_support': [row['supports_claim'] and row['id'] in context_ids
                                                       for row in citation_reviews]})
    try:
        result = aggregate(cases, traces, prepared)
    except ValueError as exc:
        raise LabError(str(exc)) from None
    for value in result.values():
        if isinstance(value, dict) and 'n' in value and 'N' in value:
            value.update(metric(value['n'], value['N']))
    return {**result,
            'reviewers': [{'name': name, 'state': state} for name, state in sorted(reviewers)],
            'context_grade_distribution': {str(i): context_grades[i] for i in range(3)},
            'context_evidence_rate': metric(context_grades[2], context_grades.total()),
            'search_distribution': dict(sorted(Counter(t['searches'] for t in traces if t['status'] != 'error').items()))}


def summarize(cases, traces, prices):
    statuses = Counter(trace['status'] for trace in traces)
    rank_errors = sum(trace['ranking_status'] == 'error' for trace in traces)
    total = usage_report([row for trace in traces for row in trace['usage']['calls']], prices)
    successful = [trace for trace in traces if trace['status'] != 'error']
    elapsed = [trace['elapsed_ms'] for trace in successful]
    contexts = [trace['context'] or trace.get('packed_offline') for trace in successful]
    sizes = [context['chars'] for context in contexts if context is not None]
    return {
        'query_count': len(cases), 'answerable_count': sum(case['answerable'] for case in cases),
        'status_counts': dict(sorted(statuses.items())), 'ranking_errors': rank_errors,
        'ranking_errors_answerable': sum(case['answerable'] and trace['ranking_status'] == 'error'
                                         for case, trace in zip(cases, traces, strict=True)),
        'retrieval': score_retrieval(cases, traces),
        'usage': total,
        'successful_api_cost_usd': usage_report([row for trace in successful for row in trace['usage']['calls']],
                                                prices)['total_usd'],
        'elapsed_ms_total': sum(trace['elapsed_ms'] for trace in traces),
        'median_success_ms': median(elapsed) if elapsed else None,
        'context_chars': {'N': len(sizes), 'mean': mean(sizes) if sizes else None,
                          'max': max(sizes, default=None)},
    }


def save_run(out, cases, traces, metadata, prices):
    summary = {**metadata, 'traces_sha256': sha256_json(traces), **summarize(cases, traces, prices)}
    write_json(out / 'traces.json', traces)
    if metadata['mode'] == 'agent':
        write_json(out / 'reviews.template.json', review_template(cases, traces))
    write_json(out / 'summary.json', summary)
    return summary
