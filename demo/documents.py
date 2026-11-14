"""Extract and chunk uploaded documents, retaining normalized page offsets."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader

from src.common import LabError, sha256_json

PARAGRAPH_BREAK = re.compile(r'\n\n+')
SENTENCE_END = re.compile(r'''([.!?][”’"')]*)(?:\s+|$)''')


@dataclass(frozen=True)
class Document:
    name: str
    digest: str
    pages: dict[int, str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunking:
    method: str = 'paragraph'
    size: int = 500
    overlap: int = 0

    def __post_init__(self):
        if self.method not in ('fixed', 'paragraph') or self.size < 1:
            raise LabError('Chọn cách cắt hợp lệ và kích thước lớn hơn 0.')
        if not 0 <= self.overlap < self.size or (self.method == 'paragraph' and self.overlap):
            raise LabError('Overlap phải từ 0 đến dưới kích thước chunk. Cách B dùng overlap 0.')

    def index_key(self, document):
        return sha256_json([1, document.digest, self.method, self.size, self.overlap])


def normalize_text(text):
    # Adapted from Day 06 starter/src/ingestion.py. Keep paragraph and line breaks.
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def pdf_pages(data):
    pages, failed = {}, []
    try:
        pdf = PdfReader(BytesIO(data))
        if pdf.is_encrypted and not pdf.decrypt(''):
            raise LabError('PDF cần mật khẩu. Hãy mở khóa tệp trước khi nạp.')
        for number, page in enumerate(pdf.pages, 1):
            try:
                pages[number] = normalize_text(page.extract_text() or '')
            except Exception:
                pages[number] = ''
                failed.append(number)
    except LabError:
        raise
    except Exception:
        raise LabError('PDF không đọc được. Kiểm tra tệp hoặc xuất lại PDF.') from None
    if not any(pages.values()):
        raise LabError('PDF không trích xuất được chữ. Hãy dùng PDF có lớp văn bản.')
    warnings = tuple(f'Trang {n}: ' + ('lỗi trích xuất.' if n in failed else 'không có văn bản.')
                     for n, content in pages.items() if not content)
    return pages, warnings


def read_document(name, data):
    name = Path(name.replace('\\', '/')).name
    suffix = Path(name).suffix.lower()
    warnings = ()
    if not data:
        raise LabError('Tệp rỗng.')
    if suffix == '.pdf':
        pages, warnings = pdf_pages(data)
    elif suffix in ('.txt', '.md'):
        try:
            text = normalize_text(data.decode('utf-8-sig'))
        except UnicodeDecodeError:
            raise LabError('TXT hoặc Markdown phải được lưu ở UTF-8.') from None
        if not text or '\0' in text:
            raise LabError('Tệp rỗng hoặc chứa nội dung nhị phân.')
        pages = {1: text}
    else:
        raise LabError('Chọn tệp TXT, Markdown hoặc PDF.')
    return Document(name, hashlib.sha256(data).hexdigest(), pages, warnings)


def fixed_ranges(text, size, overlap):
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        yield start, end
        if end == len(text):
            break


def paragraph_ranges(text):
    start = 0
    for gap in PARAGRAPH_BREAK.finditer(text):
        yield start, gap.start()
        start = gap.end()
    if start < len(text):
        yield start, len(text)


def bounded_ranges(text, start, end, size):
    while end - start > size:
        window = text[start:start + size + 1]
        sentence_ends = [m.end(1) for m in SENTENCE_END.finditer(window)]
        ends = [offset for offset in sentence_ends if size // 2 <= offset <= size]
        cut = ends[-1] if ends else window.rfind('\n', size // 2, size + 1)
        if cut < 1:
            cut = window.rfind(' ', size // 2, size + 1)
        if cut < 1:
            cut = size
        yield start, start + cut
        start += cut
        while start < end and text[start].isspace():
            start += 1
    if start < end:
        yield start, end


def paragraph_chunks(text, size):
    # Adapted from Day 06 starter/src/chunking.py for uploaded pages.
    first = last = None
    for begin, finish in paragraph_ranges(text):
        for start, end in bounded_ranges(text, begin, finish, size):
            if first is None:
                first = start
            elif end - first > size:
                yield first, last
                first = start
            last = end
    if first is not None:
        yield first, last


def cut_document(document, config):
    key = config.index_key(document)
    chunks = []
    for page, text in document.pages.items():
        ranges = (fixed_ranges(text, config.size, config.overlap) if config.method == 'fixed'
                  else paragraph_chunks(text, config.size))
        for start, end in ranges:
            content = text[start:end]
            if not content.strip():
                continue
            chunks.append({
                'id': str(uuid5(NAMESPACE_URL, f'day07-demo:{key}:{page}:{start}:{end}')),
                'source_id': document.digest, 'source': document.name, 'page': page, 'text': content,
                'metadata': {'start_char': start, 'end_char': end, 'unit_url': None},
            })
    return chunks
