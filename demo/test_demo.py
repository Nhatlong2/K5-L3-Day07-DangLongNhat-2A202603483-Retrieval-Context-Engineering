"""Offline regressions for the demo, including Streamlit and local Qdrant."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from openai.types.chat.chat_completion import Choice
from pypdf import PdfWriter
from pypdf.generic import (DecodedStreamObject, DictionaryObject, NameObject)
from streamlit.testing.v1 import AppTest

from demo.agent import normalize_answer
from demo.documents import Chunking, cut_document, read_document
from demo.runtime import DEMO_DIR, UploadStore, demo_config, run_query
from src.agent import context_for, finish
from src.common import LabError
from src.day06 import DIMENSIONS
from src.provider import OpenAIProvider

MONDAY = 'Phòng Delta mở cửa lúc mấy giờ vào thứ Hai?'
SATURDAY = 'Phòng Delta mở cửa lúc mấy giờ vào thứ Bảy?'
TEST_VECTOR = [1.0] + [0.0] * (DIMENSIONS - 1)


def sample_chunks():
    path = DEMO_DIR / 'phong_delta.txt'
    document = read_document(path.name, path.read_bytes())
    return document, cut_document(document, Chunking())


def tool_reply(query):
    return Choice.model_validate({'index': 0, 'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': 'call_test', 'function': {
            'name': 'search_corpus', 'arguments': json.dumps({'query': query}),
        }}],
    }})


def final_reply(claims):
    return Choice.model_validate({'index': 0, 'finish_reason': 'stop', 'message': {
        'role': 'assistant', 'content': json.dumps({
            'status': 'answered' if claims else 'abstain', 'claims': claims,
        }),
    }})


def pdf_bytes(text):
    pdf = PdfWriter()
    pdf.add_blank_page(width=200, height=200)
    if text:
        page = pdf.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 100 Td (Delta 08:30) Tj ET')
        page.replace_contents(stream)
    out = BytesIO()
    pdf.write(out)
    return out.getvalue()


class DocumentTests(unittest.TestCase):
    def test_utf8_bom_nfc_overlap_and_offsets(self):
        text = 'Phòng Delta mo\u031b\u0309 cửa.\r\n\r\n  Ca sáng 08:30.\r\n'
        doc = read_document('delta.md', text.encode('utf-8-sig'))
        self.assertEqual(doc.pages[1], 'Phòng Delta mở cửa.\n\nCa sáng 08:30.')
        for config in (Chunking('fixed', 20, 5), Chunking('paragraph', 20)):
            rows = cut_document(doc, config)
            self.assertEqual(rows, cut_document(doc, config))
            for row in rows:
                start, end = (row['metadata'][key] for key in ('start_char', 'end_char'))
                self.assertEqual(doc.pages[row['page']][start:end], row['text'])
                self.assertLessEqual(end - start, config.size)
        rows = cut_document(doc, Chunking('fixed', 20, 5))
        self.assertEqual(rows[0]['metadata']['end_char'] - rows[1]['metadata']['start_char'], 5)
        with self.assertRaises(LabError):
            Chunking('fixed', 20, 20)

    def test_paragraph_chunk_prefers_sentence(self):
        doc = read_document('delta.txt', b'This is sentence one. This is sentence two. Last sentence.')
        rows = cut_document(doc, Chunking('paragraph', 30))
        self.assertEqual(rows[0]['text'], 'This is sentence one.')

    def test_pdf_page_numbers_and_errors(self):
        doc = read_document('delta.pdf', pdf_bytes(True))
        self.assertEqual(doc.pages, {1: '', 2: 'Delta 08:30'})
        self.assertEqual(cut_document(doc, Chunking())[0]['page'], 2)
        self.assertEqual(len(doc.warnings), 1)
        for name, data, error in [('empty.txt', b'', 'rỗng'),
                                  ('legacy.md', b'\xff', 'UTF-8'),
                                  ('broken.pdf', b'not a pdf', 'PDF không đọc được'),
                                  ('blank.pdf', pdf_bytes(False), 'không trích xuất được chữ')]:
            with self.subTest(name=name), self.assertRaisesRegex(LabError, error):
                read_document(name, data)


class IndexAndCitationTests(unittest.TestCase):
    def test_vector_retry_and_changed_chunking(self):
        doc, chunks = sample_chunks()
        provider = SimpleNamespace(embed_documents=lambda texts: [TEST_VECTOR for _ in texts],
                                   embed=lambda _: TEST_VECTOR, calls=[], close=lambda: None)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            store = UploadStore(root / Chunking().index_key(doc), chunks)
            progress = []

            def fail(_):
                raise RuntimeError('batch failed')

            with self.assertRaisesRegex(RuntimeError, 'batch failed'):
                store.build(SimpleNamespace(embed_documents=fail), lambda *_: None)
            self.assertFalse(store.ready())
            store.build(provider, lambda n, total: progress.append((n, total)))
            self.assertEqual(progress[-1], (len(chunks), len(chunks)))
            self.assertTrue(store.ready())
            for method in ('dense', 'lexical', 'hybrid'):
                with patch('demo.runtime.DemoProvider', return_value=provider):
                    result = run_query(MONDAY, None, demo_config(method, 3, 1800, 'fixture'),
                                       'retrieval', 'test', chunks, vector_store=store)
                self.assertEqual(result['status'], 'retrieved')
                self.assertIn('08:30', result['packed_offline']['text'])
                if method == 'hybrid':
                    self.assertIn('branch_ranks', result['original_candidates'][0])
            changed = Chunking('fixed', 500, 50)
            self.assertFalse(UploadStore(root / changed.index_key(doc), cut_document(doc, changed)).ready())

    def test_every_claim_needs_citation_in_sent_context(self):
        _, chunks = sample_chunks()
        first, second = chunks[:2]
        context = context_for([[{**first, 'rank': 1}]], 1800)
        for cite, expected in ((first['id'], 'answered'), (second['id'], 'abstain')):
            proposed = json.dumps({'status': 'answered', 'claims': [
                {'text': 'Phòng mở cửa lúc 08:30.', 'chunk_ids': [cite]},
            ]})
            normalized = normalize_answer(proposed)
            result = finish(json.dumps(normalized), context)
            self.assertEqual(result['status'], expected)
        no_evidence = normalize_answer('{"status":"abstain","claims":[]}')
        self.assertEqual(finish(json.dumps(no_evidence), context)['citations'], [])

    def test_agent_tool_and_claim_flow(self):
        _, chunks = sample_chunks()
        valid = chunks[0]['id']
        for question, claims, expected in (
                (MONDAY, [{'text': 'Phòng mở cửa lúc 08:30.', 'chunk_ids': [valid]}], 'answered'),
                (SATURDAY, [], 'abstain'),
                (MONDAY, [{'text': 'Phòng mở cửa lúc 08:30.', 'chunk_ids': [str(uuid4())]}], 'abstain')):
            with self.subTest(question=question, claims=claims):
                with (patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test'}),
                      patch.object(OpenAIProvider, 'complete',
                                   side_effect=[tool_reply(question), final_reply(claims)]) as complete):
                    trace = run_query(question, None, demo_config('lexical', 3, 1800, 'fixture'),
                                      'agent', 'Tài liệu vừa nạp', chunks)
                self.assertEqual(trace['status'], expected)
                self.assertEqual(trace['searches'], 1)
                sent = complete.call_args.args[1]
                self.assertEqual(sent[-1]['content'], trace['context']['text'])
                self.assertIn('Tài liệu vừa nạp', sent[0]['content'])
                self.assertEqual(trace['citations'], [valid] if expected == 'answered' else [])
                if expected == 'answered':
                    self.assertIn('08:30', trace['answer'])


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'OPENAI_API_KEY': ' '})
        env.start()
        self.addCleanup(env.stop)
        self.app = AppTest.from_file(str(DEMO_DIR / 'app.py'), default_timeout=12).run()
        self.assertFalse(self.app.exception)

    def widget(self, kind, label):
        return next(w for w in getattr(self.app, kind) if w.label.startswith(label))

    def test_offline_compare_and_invalidate_old_index(self):
        self.widget('button', 'Cắt và xem trước').click().run()
        self.widget('button', 'Lập chỉ mục TF-IDF').click().run()
        self.widget('text_area', 'Câu hỏi').set_value(MONDAY).run()
        self.widget('button', 'So sánh truy xuất').click().run()
        result = self.app.session_state.history[-1][1]
        self.assertIn('08:30', result['lexical']['packed_offline']['text'])
        self.assertEqual(result['dense']['status'], 'unavailable')
        self.assertEqual(result['hybrid']['status'], 'unavailable')
        self.assertTrue(self.widget('button', 'Hỏi agent').disabled)
        self.app.radio(key='chunk-method').set_value('fixed').run()
        self.assertTrue(self.widget('button', 'So sánh truy xuất').disabled)
        self.widget('button', 'Cắt và xem trước').click().run()
        self.assertFalse(self.widget('button', 'Lập chỉ mục TF-IDF').disabled)
        self.assertFalse(self.app.exception)

    def test_replayed_citation_and_missing_evidence(self):
        self.widget('button', '[1] Mở citation').click().run()
        self.assertEqual(self.app.session_state['citation-saved-D01'],
                         '8f5a495e-57d2-5674-b60b-f411f7edea0a')
        self.widget('selectbox', 'Ví dụ từ dev').set_value('DN01').run()
        self.assertIn('Thiếu bằng chứng trong tài liệu.', [heading.value for heading in self.app.subheader])
        self.assertFalse(self.app.exception)

    def test_hybrid_ui_with_test_vectors_and_mocked_agent(self):
        _, chunks = sample_chunks()
        cid = chunks[0]['id']
        claims = [{'text': 'Phòng mở cửa lúc 08:30.', 'chunk_ids': [cid]}]
        with (TemporaryDirectory() as folder,
              patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test'}),
              patch('demo.runtime.DEMO_WORK', Path(folder)),
              patch('demo.agent.DemoProvider.embed_documents', side_effect=lambda texts: [TEST_VECTOR] * len(texts)),
              patch.object(OpenAIProvider, 'embed', return_value=TEST_VECTOR),
              patch.object(OpenAIProvider, 'complete', side_effect=[
                  tool_reply(MONDAY), final_reply(claims), tool_reply(SATURDAY), final_reply([]),
              ]) as complete):
            self.app.run()
            self.widget('button', 'Cắt và xem trước').click().run()
            self.widget('button', 'Lập chỉ mục TF-IDF').click().run()
            self.widget('button', 'Tạo chỉ mục vector').click().run()
            self.widget('text_area', 'Câu hỏi').set_value(MONDAY).run()
            self.widget('button', 'So sánh truy xuất').click().run()
            self.assertTrue(all(t['status'] == 'retrieved' for t in self.app.session_state.history[-1][1].values()))
            self.widget('button', 'Hỏi agent').click().run()
            answer = self.app.session_state.history[-1][1]
            self.assertIn('08:30', answer['answer'])
            self.widget('button', f'[1] Mở citation · {cid[:8]}').click().run()
            self.assertEqual(self.app.session_state['citation-live'], cid)
            self.assertEqual(complete.call_count, 2)
            self.widget('text_area', 'Câu hỏi').set_value(SATURDAY).run()
            self.widget('button', 'Hỏi agent').click().run()
            self.assertEqual(self.app.session_state.history[-1][1]['status'], 'abstain')
            self.assertFalse(self.app.exception)


if __name__ == '__main__':
    unittest.main()
