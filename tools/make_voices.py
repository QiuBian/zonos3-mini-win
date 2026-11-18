#!/usr/bin/env python3
"""Write zonos3_addon/voices.json: ZONOS3-Mini's default voices as speaker embeddings.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Zyphra ships 27 default voices (gguf/voices/ of Zyphra/ZONOS3-Mini, Apache-2.0),
each a clip with its 192-value speaker embedding. The add-on needs only the
embedding, so it carries those (27 x 768 bytes) and none of the clips: a stock
voice then needs neither the speaker encoder nor ffmpeg.

Reads gguf/voices/README.md (name, style, gender per language) and each
<Language>/<Name>_*.embedding.npy at tools/pin_models.py's REVISION.

    python tools/make_voices.py
"""

from __future__ import annotations

import base64
import json
import re
import runpy
import struct
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'zonos3_addon' / 'voices.json'
PINS = runpy.run_path(str(ROOT / 'tools' / 'pin_models.py'), run_name='make_voices')
REPO, REVISION = PINS['REPO'], PINS['REVISION']


def fetch(path: str) -> bytes:
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(path)}'
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def listing() -> list[str]:
    url = f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}/gguf/voices?recursive=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        return [entry['path'] for entry in json.load(response) if entry['type'] == 'file']


def npy_float32(data: bytes) -> list[float]:
    """A little-endian float32 vector from a .npy file."""
    if data[:6] != b'\x93NUMPY':
        raise ValueError('not a .npy file')
    major = data[6]
    header_len = struct.unpack('<H' if major == 1 else '<I', data[8:10] if major == 1 else data[8:12])[0]
    start = (10 if major == 1 else 12) + header_len
    header = data[(10 if major == 1 else 12):start].decode('latin1')
    if "'descr': '<f4'" not in header or 'fortran_order\': False' not in header:
        raise ValueError(f'unexpected .npy header {header!r}')
    body = data[start:]
    return list(struct.unpack(f'<{len(body) // 4}f', body))


def main() -> int:
    readme = fetch('gguf/voices/README.md').decode('utf-8')
    files = listing()
    voices, language = [], None
    for line in readme.splitlines():
        heading = re.match(r'^## (.+)$', line)
        if heading:
            language = heading.group(1).strip()
            continue
        row = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if not language or len(row) != 4 or row[0] in ('Voice', '') or set(row[0]) <= set('-'):
            continue
        name, style, gender, ident = row[0], row[1], row[2], row[3].strip('`')
        matches = [p for p in files if p.startswith(f'gguf/voices/{language}/{name}_')
                   and p.endswith('.embedding.npy')]
        if len(matches) != 1:
            raise SystemExit(f'FATAL: {language}/{name}: expected one embedding, found {matches}')
        values = npy_float32(fetch(matches[0]))
        if len(values) != 192:
            raise SystemExit(f'FATAL: {matches[0]} holds {len(values)} values, not 192')
        voices.append({
            'id': 'zonos3-' + re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-'),
            'name': name,
            'language': language,
            'style': '' if style in ('—', '-') else style,
            'gender': gender,
            'upstream_id': ident,
            'source': matches[0],
            'embedding': base64.b64encode(struct.pack('<192f', *values)).decode('ascii'),
        })
    ids = [v['id'] for v in voices]
    if len(set(ids)) != len(ids):
        raise SystemExit(f'FATAL: voice ids collide: {ids}')
    OUT.write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'voices': voices}, indent=1,
                              ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {OUT}: {len(voices)} voices')
    return 0


if __name__ == '__main__':
    sys.exit(main())
