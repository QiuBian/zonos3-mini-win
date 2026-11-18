#!/usr/bin/env python3
"""Write zonos3_addon/pins.json: the model files the add-on downloads, at one revision, with size and sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Reads Hugging Face's API for Zyphra/ZONOS3-Mini at REVISION (the LFS sha256 of
each file) and writes the groups the add-on downloads:

- core: the backbone (Q4_0 experts, Q8_0 mixers) and the default waveform decoder;
- clone: the speaker encoder, for a voice cloned from a clip.

    python tools/pin_models.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'zonos3_addon' / 'pins.json'

REPO = 'Zyphra/ZONOS3-Mini'
REVISION = '2921b8821772b0bd10a04456863d21abf9d2f283'  # 2026-11-17
GROUPS = {
    'core': ['gguf/backbone-ptq.gguf', 'gguf/zac-keep2.gguf'],
    'clone': ['gguf/spk-encoder.gguf'],
}


def paths_info(paths: list[str]) -> dict[str, dict]:
    url = f'https://huggingface.co/api/models/{REPO}/paths-info/{REVISION}'
    body = urllib.parse.urlencode([('paths', p) for p in paths]).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=60) as response:
        return {entry['path']: entry for entry in json.load(response)}


def main() -> int:
    info = paths_info([p for paths in GROUPS.values() for p in paths])
    groups = {}
    for group, paths in GROUPS.items():
        files = []
        for path in paths:
            entry = info.get(path)
            if not entry or 'lfs' not in entry:
                raise SystemExit(f'FATAL: {path} is not an LFS file of {REPO} at {REVISION}')
            files.append({'path': path, 'size': entry['lfs']['size'], 'sha256': entry['lfs']['oid']})
        groups[group] = {'repo': REPO, 'revision': REVISION, 'files': files}
    OUT.write_text(json.dumps({'groups': groups}, indent=1) + '\n', encoding='utf-8')
    total = {g: sum(f['size'] for f in v['files']) for g, v in groups.items()}
    print(f'wrote {OUT}: ' + ', '.join(f'{g} {n / 1e6:,.0f} MB' for g, n in total.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
