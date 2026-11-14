"""Shared file, scope and error contracts for the lab."""

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY06 = ROOT.parent / 'day06'
WORK = ROOT / '.day07'
CHUNK_SNAPSHOT = WORK / 'chunks.json'
DEFAULT_STORE = Path('starter/.day06/qdrant_openai_large')
FAMILIES = ('asyncio', 'concurrent.futures', 'threading')


class LabError(RuntimeError):
    """A message safe to show in a CLI result or benchmark trace."""


class SetupUnavailable(LabError):
    """A missing prerequisite that prevented API or store access."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                        encoding='utf-8', newline='\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sha256_file(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def sha256_json(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def validate_scope(filters):
    if filters is None or filters == {}:
        return None
    if (not isinstance(filters, dict) or set(filters) != {'api_family'}
            or filters['api_family'] not in FAMILIES):
        raise LabError('Invalid api_family scope. Use asyncio, concurrent.futures or threading.')
    return {'api_family': filters['api_family']}


def safe_error(exc, operation):
    if isinstance(exc, LabError):
        return str(exc)
    status = getattr(exc, 'status_code', None)
    http = f', HTTP {status}' if type(status) is int else ''
    quota = ' Check API billing/quota.' if getattr(exc, 'code', None) == 'insufficient_quota' else ''
    return f'{operation} failed ({type(exc).__name__}{http}).{quota}'
