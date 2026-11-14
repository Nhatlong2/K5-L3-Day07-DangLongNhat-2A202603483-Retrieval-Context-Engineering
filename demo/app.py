"""Run from the Day 07 checkout: python -m streamlit run demo/app.py."""

import json
import os
import sys
from html import escape
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script; import the Day 07 package from its checkout.
repo_path = str(Path(__file__).resolve().parents[1])
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from demo.documents import Chunking, cut_document, read_document
from demo.runtime import (DEMO_DIR, DEMO_WORK, METHODS, PLAN, UploadStore, benchmark_results,
                          build_upload_vectors, demo_config, official_snapshot, official_store_error,
                          prepare_official, run_query)
from src.agent import INLINE_ID
from src.common import DEFAULT_DAY06, FAMILIES, ROOT, safe_error
from src.day06 import DIMENSIONS, EMBEDDING_MODEL
from src.provider import load_local_env
from starter.baseline import TfidfIndex

UPLOAD = 'Tài liệu vừa nạp'
OFFICIAL = 'Corpus Day 06'
ABSTAIN = 'Thiếu bằng chứng trong tài liệu.'
STATUSES = {'answered': 'Đã trả lời', 'abstain': ABSTAIN, 'error': 'Lỗi',
            'retrieved': 'Đã truy xuất', 'ready': 'Đã lập chỉ mục', 'unavailable': 'Chưa sẵn sàng'}
STATE = st.session_state


def init():
    st.set_page_config(page_title='Day 07 · Retrieval Lab', layout='wide', page_icon='🔎')
    st.markdown("""
<style>
.block-container {max-width: 1220px; padding-top: 2.5rem;}
.source-page {white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid #8293aa60;
              border-radius: .65rem; padding: 1rem; margin: .4rem 0; max-height: 25rem;
              overflow: auto; font: inherit; line-height: 1.55;}
.source-page mark {color: #172b24; background: #a5ebc8; border-radius: .15rem;}
</style>
""", unsafe_allow_html=True)
    load_local_env()
    if 'history' not in STATE:
        STATE.history = []
    return bool(os.getenv('OPENAI_API_KEY', '').strip())


def remember(label, payload):
    STATE.history.append((label, payload))
    STATE.history = STATE.history[-20:]


@st.cache_data(show_spinner=False)
def extract(name, data):
    return read_document(name, data)


@st.cache_data(show_spinner=False)
def official_pages():
    with (ROOT / 'data' / 'corpus.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    return {(row['document_id'], row['page']): row['text'] for row in rows}


def source_page(page, span=None):
    if span:
        start, end = span
        page = (escape(page[:start]) + '<mark>' + escape(page[start:end])
                + '</mark>' + escape(page[end:]))
    else:
        page = escape(page)
    st.html(f'<div class="source-page">{page}</div>')


def source_view(chunk, page):
    meta = chunk['metadata']
    start, end = meta['start_char'], meta['end_char']
    st.text(f"{chunk['source']} · Trang/mục {chunk['page']}\n[{start}, {end}) · {chunk['id']}")
    if page is not None and page[start:end] == chunk['text']:
        before, after = max(0, start - 250), min(len(page), end + 250)
        source_page(page[before:after], (start - before, end - before))
    else:
        source_page(chunk['text'], (0, len(chunk['text'])))
    url = meta.get('unit_url')
    if url and url.startswith('https://docs.python.org/'):
        st.link_button('Mở trang tài liệu Python', url)


def documents_tab(has_key):
    st.subheader('Nạp tài liệu')
    upload = st.file_uploader('TXT, Markdown hoặc PDF có lớp văn bản', type=['txt', 'md', 'pdf'],
                              help='Tối đa 20 MB.')
    use_sample = st.checkbox('Dùng tệp mẫu Phòng Delta', value=True, disabled=upload is not None)
    if upload is not None:
        if upload.size > 20 * 1024 * 1024:
            st.error('Tệp vượt quá giới hạn 20 MB.')
            return None
        name, data = upload.name, upload.getvalue()
    elif use_sample:
        sample = DEMO_DIR / 'phong_delta.txt'
        name, data = sample.name, sample.read_bytes()
    else:
        return None
    try:
        document = extract(name, data)
    except Exception as exc:
        st.error(safe_error(exc, 'Trích xuất tài liệu'))
        return None
    for warning in document.warnings:
        st.warning(warning)

    page_col, strategy_col = st.columns([1.4, 1], gap='large')
    with strategy_col:
        st.subheader('Cắt tài liệu')
        method = st.radio('Cách cắt', ('paragraph', 'fixed'),
                          format_func={'paragraph': 'B · Theo đoạn', 'fixed': 'A · Cửa sổ ký tự'}.get,
                          horizontal=True, key='chunk-method')
        size = st.number_input('Số ký tự tối đa', min_value=1, max_value=4000, value=500, step=1)
        overlap = st.number_input('Overlap', min_value=0, max_value=3999, value=50, step=1,
                                  disabled=method == 'paragraph')
        try:
            config = Chunking(method, size, overlap if method == 'fixed' else 0)
        except Exception as exc:
            st.error(safe_error(exc, 'Cấu hình cắt'))
            config = None
        key = config.index_key(document) if config else None
        if st.button('Cắt và xem trước', type='primary', disabled=config is None):
            STATE.preview = (key, document.name, cut_document(document, config))
        preview = STATE.get('preview')
        chunks = preview[2] if preview and preview[:2] == (key, document.name) else []
        if chunks:
            st.metric('Số chunk', len(chunks))
        else:
            st.caption('Bấm cắt để xem chunk cho cấu hình hiện tại.')
    with page_col:
        st.subheader('Văn bản trích xuất')
        page = st.selectbox('Trang', list(document.pages), key=f'page-{document.digest}')
        text = document.pages[page]
        st.caption(f'{len(text):,} ký tự đã chuẩn hóa · Trang {page}/{len(document.pages)}')
        source_page(text) if text else st.info('Trang này không có văn bản trích xuất được.')

    if not chunks:
        return None
    chosen = st.selectbox('Chunk cần kiểm tra', range(len(chunks)),
                          format_func=lambda i: (f"#{i + 1} · Trang {chunks[i]['page']} · "
                                                 f"{len(chunks[i]['text'])} ký tự · {chunks[i]['id'][:8]}"),
                          key=f'chunk-{key}')
    chunk = chunks[chosen]
    source_view(chunk, document.pages[chunk['page']])

    st.subheader('Lập chỉ mục')
    lexical_col, dense_col = st.columns(2)
    lexical_entry = STATE.get('lexical')
    lexical = lexical_entry[2] if lexical_entry and lexical_entry[:2] == (key, document.name) else None
    vector_store = UploadStore(DEMO_WORK / 'vectors' / key, chunks)
    with lexical_col:
        if st.button('Lập chỉ mục TF-IDF', disabled=lexical is not None):
            lexical = TfidfIndex(chunks)
            STATE.lexical = (key, document.name, lexical)
        st.caption(f'TF-IDF · {len(chunks)} chunk' if lexical is not None else 'TF-IDF · Chưa lập chỉ mục')
    with dense_col:
        st.caption(f'{EMBEDDING_MODEL} · {DIMENSIONS:,} chiều · Qdrant cosine')
        st.caption('Khi tạo vector, các đoạn tài liệu sẽ được gửi tới OpenAI.')
        if st.button('Tạo chỉ mục vector', disabled=not has_key):
            progress = st.progress(0, 'Đang tạo chỉ mục vector…')
            trace = build_upload_vectors(vector_store, lambda n, total: progress.progress(n / total))
            progress.empty()
            remember('Tạo chỉ mục vector', trace)
            STATE.index_result = (key, trace)
        last = STATE.get('index_result')
        if last and last[0] == key and last[1]['status'] == 'error':
            st.error(last[1]['error'])
        if vector_store.ready():
            st.caption(f'Qdrant · {len(chunks)} chunk')
        else:
            st.caption('Qdrant · Chưa lập chỉ mục')
        if not has_key:
            st.caption('Cần OPENAI_API_KEY để tạo vector.')
    return {'key': key, 'document': document, 'chunks': chunks,
            'lexical': lexical, 'vector_store': vector_store}


def day06_controls():
    with st.expander('Kết nối Day 06'):
        day06 = st.text_input('Checkout Day 06', value=os.getenv('DAY06_PATH', str(DEFAULT_DAY06)))
        store = st.text_input('Thư mục Qdrant Day 06', value=os.getenv('DAY06_STORE', ''),
                              placeholder=r'starter\.day06\qdrant_openai_large trong checkout Day 06')
        if st.button('Chuẩn bị snapshot Lexical', disabled=not day06.strip()):
            try:
                with st.spinner('Đang dựng lại collection B từ checkout Day 06…'):
                    prepare_official(Path(day06).expanduser())
                st.success('Đã chuẩn bị snapshot Day 06.')
            except Exception as exc:
                st.error(safe_error(exc, 'Chuẩn bị snapshot Day 06'))
    try:
        provenance, chunks = official_snapshot()
    except Exception as exc:
        st.info(safe_error(exc, 'Đọc snapshot Day 06') + ' Mở Kết nối Day 06 để chuẩn bị.')
        provenance, chunks = None, []
    problem = official_store_error(day06, store)
    if problem:
        st.caption(problem)
    family = st.selectbox('Phạm vi API', [None, *FAMILIES], format_func=lambda item: item or 'Tất cả')
    return (day06, store, provenance), chunks, problem, {'api_family': family} if family else None


def query_tab(upload, has_key):
    st.subheader('Truy xuất và hỏi đáp')
    corpus = st.radio('Nguồn truy xuất', (UPLOAD, OFFICIAL), horizontal=True)
    official, lexical, vectors, filters = None, None, None, None
    if corpus == OFFICIAL:
        official, chunks, problem, filters = day06_controls()
        lexical_ready = bool(chunks)
        dense_ready = lexical_ready and problem is None
        data_key = (official[0], official[1], filters)
    else:
        chunks = upload['chunks'] if upload else []
        lexical = upload['lexical'] if upload else None
        vectors = upload['vector_store'] if upload else None
        lexical_ready = lexical is not None
        dense_ready = vectors is not None and vectors.ready()
        data_key = (upload['key'], upload['document'].name) if upload else None
        if not lexical_ready or not dense_ready:
            st.caption('Mở tab Tài liệu & chunk để cắt tài liệu và lập chỉ mục cần dùng.')

    preset = st.selectbox('Cấu hình', list(PLAN['variants']), index=2,
                          format_func=lambda key: METHODS.get(key, key))
    variant = PLAN['variants'][preset]
    cols = st.columns(2)
    with cols[0]:
        top_k = st.number_input('Top-k', min_value=1, max_value=25, value=variant['top_k'],
                                step=1, key=f'k-{preset}')
    with cols[1]:
        budget = st.number_input('Ngân sách context (ký tự)', min_value=1, max_value=16000,
                                 value=variant['context_budget_chars'], step=100, key=f'budget-{preset}')
    model = os.getenv('OPENAI_AGENT_MODEL', '').strip() or PLAN['agent_model']
    config = demo_config(preset, top_k, budget, model)
    question = st.text_area('Câu hỏi', key='question', height=90,
                            placeholder='Phòng Delta mở cửa lúc mấy giờ vào thứ Hai?').strip()
    signature = (corpus, data_key, preset, top_k, budget, question)
    blocked = {
        'lexical': None if lexical_ready else 'Cần chỉ mục TF-IDF hoặc snapshot Day 06.',
        'dense': ('Cần OPENAI_API_KEY.' if not has_key else
                  (None if dense_ready else 'Cần chỉ mục vector hoặc store Day 06.')),
    }
    blocked['hybrid'] = blocked['dense'] or blocked['lexical']
    reason = ('Cần OPENAI_API_KEY.' if not has_key else blocked[config['method']])
    if reason:
        st.caption(f'Hỏi agent · {reason}')
    elif question:
        st.caption(f'Agent · {model} · Tối đa {config["max_searches"]} lượt search')
    if has_key and (dense_ready or not reason):
        st.caption('Tác vụ dùng OpenAI gửi câu hỏi và context cần thiết tới API.')

    compare_col, ask_col = st.columns(2)
    with compare_col:
        compare = st.button('So sánh truy xuất', disabled=not question or all(blocked.values()))
    with ask_col:
        ask = st.button('Hỏi agent', type='primary', disabled=not question or bool(reason))

    if compare or ask:
        STATE.pop('citation-live', None)
        with st.spinner('Đang truy xuất…' if compare else 'Agent đang tìm bằng chứng…'):
            if compare:
                results = {}
                for method in METHODS:
                    if blocked[method]:
                        results[method] = {'method': method, 'status': 'unavailable', 'error': blocked[method],
                                           'question': question, 'corpus': corpus}
                        continue
                    comparison = config | {'method': method, 'id': method}
                    results[method] = run_query(question, filters, comparison, 'retrieval', corpus, chunks,
                                                lexical=lexical, vector_store=vectors, official=official)
                remember('So sánh truy xuất', results)
            else:
                result = run_query(question, filters, config, 'agent', corpus, chunks,
                                   lexical=lexical, vector_store=vectors, official=official)
                results = {'agent': result}
                remember('Hỏi agent', result)
        STATE.latest = (signature, results)

    latest = STATE.get('latest')
    if latest and latest[0] == signature:
        results = latest[1]
        if 'agent' in results:
            pages = (official_pages() if official else
                     {(upload['document'].digest, p): text for p, text in upload['document'].pages.items()})
            render_answer(results['agent'], pages, 'live')
        else:
            for column, method in zip(st.columns(3), METHODS, strict=True):
                with column:
                    st.subheader(METHODS[method])
                    render_retrieval(results[method])


def render_retrieval(result):
    if 'error' in result:
        st.info(result['error'])
        return
    for chunk in result['original_candidates'][:result['settings']['top_k']]:
        label = f"#{chunk['rank']} · {chunk['score']:.4f} · Trang/mục {chunk['page']}"
        with st.expander(label):
            st.text(chunk['source'])
            st.code(chunk['text'], language=None, wrap_lines=True)
            if 'branch_ranks' in chunk:
                ranks = chunk['branch_ranks']
                st.caption(f"Dense #{ranks['dense'] or '—'} · Lexical #{ranks['lexical'] or '—'}")
    if not result['original_candidates']:
        st.caption('Không tìm thấy chunk phù hợp.')
    st.caption(f"{result['ranking_ms']:.1f} ms · {result['method'].upper()}")
    render_context(result.get('packed_offline'))


def render_context(context):
    if context is None:
        return
    with st.expander(f"Context · {context['chars']:,}/{context['budget_chars']:,} ký tự"):
        if context['ids']:
            st.text('\n'.join(context['ids']))
        st.code(context['text'] or 'Context rỗng.', language=None, wrap_lines=True)


def render_answer(result, pages, key):
    status = result['status']
    citations = result.get('citations', [])
    st.subheader(STATUSES.get(status, status))
    if status == 'error':
        st.error(result['error'])
    elif status == 'answered':
        labels = {cid: f'[{i}]' for i, cid in enumerate(citations, 1)}
        st.text(INLINE_ID.sub(lambda match: labels.get(match[1], match[0]), result['answer']))
    elif result.get('stop_reason') == 'invalid_citations':
        st.caption('Citation do agent đề xuất thiếu hoặc nằm ngoài context.')
    context = result.get('context') or {}
    chunks = {chunk['id']: chunk for chunk in context.get('chunks', [])}
    selected_key = f'citation-{key}'
    for index, cid in enumerate(citations, 1):
        if cid in chunks and st.button(f'[{index}] Mở citation · {cid[:8]}', key=f'{key}-{cid}'):
            STATE[selected_key] = cid
    picked = chunks.get(STATE.get(selected_key))
    if picked:
        source_view(picked, pages.get((picked['source_id'], picked['page'])))
    render_context(result.get('context'))


def trace_details(trace, key):
    if 'status' not in trace:
        for name, result in trace.items():
            st.subheader(METHODS[name])
            trace_details(result, f'{key}-{name}')
    else:
        st.text(trace.get('question', 'Tạo chỉ mục vector'))
        st.caption(STATUSES.get(trace['status'], trace['status']))
        if 'error' in trace:
            st.error(trace['error'])
        if 'filter' in trace:
            top_k = trace.get('settings', {}).get('top_k')
            scope = trace['filter'] or 'Toàn bộ nguồn đã chọn'
            st.caption(f'Phạm vi: {scope} · Top-k: {top_k or "xem từng tool call"}')
        ranking_ms = trace.get('ranking_ms')
        if ranking_ms is not None:
            st.caption(f'Xếp hạng truy vấn gốc · {ranking_ms:.1f} ms')
        if trace.get('original_candidates'):
            with st.expander('Ứng viên truy vấn gốc'):
                st.json(trace['original_candidates'], expanded=False)
        for step, event in enumerate(trace.get('events', []), 1):
            name = event.get('tool', event['event'])
            ms = event.get('elapsed_ms')
            duration = f' · {ms:.1f} ms' if ms is not None else ''
            with st.expander(f'{step}. {name}{duration}'):
                if event['event'] == 'search':
                    st.text(event['query'])
                    st.caption(f"Top-k {event['top_k']} · {len(event.get('candidates', []))} chunk")
                st.json(event, expanded=False)
        usage = trace.get('usage')
        if usage is not None:
            cost = usage['total_usd']
            st.caption('Chưa có đơn giá' if cost is None else f'Chi phí ước tính · ${cost:.8f} USD')
            if trace['status'] == 'error':
                st.caption('Chi phí chỉ tính các phản hồi API đã nhận được.')
            if usage['calls']:
                st.dataframe(usage['calls'], hide_index=True, width='stretch')
        ms = trace.get('total_ms', trace.get('elapsed_ms'))
        if ms is not None:
            st.caption(f'Tổng thời gian · {ms:.1f} ms')
        if trace.get('answer') and trace['status'] == 'answered':
            st.text(trace['answer'])
        if trace.get('citations'):
            st.text('\n'.join(trace['citations']))
        render_context(trace.get('context') or trace.get('packed_offline'))
    st.download_button('Tải trace JSON', json.dumps(trace, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name=f'{key}.json', mime='application/json', key=f'export-{key}')


def trace_tab():
    st.subheader('Trace phiên demo')
    if STATE.history:
        labels = [f'{i}. {label}' for i, (label, _) in enumerate(STATE.history, 1)]
        choice = st.selectbox('Lượt chạy', range(len(labels)), index=len(labels) - 1,
                              format_func=labels.__getitem__,
                              key=f'history-{id(STATE.history[-1])}')
        trace_details(STATE.history[choice][1], 'day07-demo')
    else:
        st.caption('Trace xuất hiện sau khi truy xuất, hỏi agent hoặc tạo chỉ mục vector.')
    st.divider()
    st.subheader('Kết quả đã nộp')
    try:
        selected, metrics, traces = benchmark_results()
    except (OSError, ValueError, KeyError) as exc:
        st.error(safe_error(exc, 'Đọc kết quả đã lưu'))
        return
    st.caption(f"Kết quả đã lưu · Cấu hình được chọn: {METHODS[selected['method']]}")
    st.dataframe(metrics, hide_index=True, width='stretch')
    case = st.selectbox('Ví dụ từ dev', list(traces),
                        format_func={'D01': 'D01 · Có bằng chứng', 'DN01': 'DN01 · Thiếu bằng chứng'}.get)
    if case:
        trace = traces[case]
        st.text(trace['question'])
        render_answer(trace, official_pages(), f'saved-{case}')
        with st.expander('Các bước của lượt chạy đã lưu'):
            trace_details(trace, f'day07-{case}')


def main():
    has_key = init()
    st.title('Day 07 · Retrieval & Context Engineering')
    st.caption('Nạp và cắt tài liệu, so sánh truy xuất, rồi kiểm tra câu trả lời tại nguồn.')
    if not has_key:
        st.info('Chế độ offline · Nạp, cắt, TF-IDF và kết quả đã lưu sẵn sàng. Dense, Hybrid và agent cần API key.')
    documents, queries, traces = st.tabs(('Tài liệu & chunk', 'Truy xuất & hỏi đáp', 'Trace & kết quả'))
    with documents:
        upload = documents_tab(has_key)
    with queries:
        query_tab(upload, has_key)
    with traces:
        trace_tab()


main()
