"""manifest.json, pins.json, voices.json and the add-on agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import base64
import json
import re
import struct
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from zonos3_addon import ADDON_ID, __version__, voices  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'zonos3_addon' / 'pins.json').read_text(encoding='utf-8'))['groups']
VOICES = json.loads((REPO_ROOT / 'zonos3_addon' / 'voices.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_versions_agree():
    assert MANIFEST['version'] == __version__
    if tomllib is not None:
        pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        assert pyproject['project']['version'] == __version__
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True and MANIFEST['compute'] == 'cpu'
    assert {f['key'] for f in MANIFEST['config_schema']['fields']} == {'threads', 'voice_ref_audio'}


def test_manifest_models_are_the_pins():
    by_group = {'zonos3-mini': 'core', 'zonos3-speaker-encoder': 'clone'}
    assert [m['id'] for m in MANIFEST['models']] == list(by_group)
    for model in MANIFEST['models']:
        group = PINS[by_group[model['id']]]
        assert (model['repo'], model['revision']) == (group['repo'], group['revision'])
        assert model['size_mb'] == round(sum(f['size'] for f in group['files']) / 1e6)
        assert model['bundled'] is False


def test_pins_are_complete():
    assert set(PINS) == {'core', 'clone'}
    for name, group in PINS.items():
        assert group['repo'] == 'Zyphra/ZONOS3-Mini'
        assert re.fullmatch(r'[0-9a-f]{40}', group['revision']), name
        for pin in group['files']:
            assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
    assert [p['path'] for p in PINS['core']['files']] == ['gguf/backbone-ptq.gguf', 'gguf/zac-keep2.gguf']
    assert [p['path'] for p in PINS['clone']['files']] == ['gguf/spk-encoder.gguf']
    assert VOICES['revision'] == PINS['core']['revision'], 'the voices come from the pinned revision'


def test_stock_voices_are_embeddings():
    assert len(VOICES['voices']) == 27
    for entry in VOICES['voices']:
        values = struct.unpack('<192f', base64.b64decode(entry['embedding']))
        assert all(abs(v) < 1e3 for v in values) and any(values), entry['id']
    assert len({v.embedding for v in voices.stock()}) == 27
    assert voices.all_voices()[0].id == voices.CLONE
    assert voices.by_id('zonos3-savannah').display_name == 'Savannah (American, English, female)'
    assert voices.by_id('zonos3-gaspard').display_name == 'Gaspard (French, male)'


@pytest.mark.parametrize('tag,code', [
    ('en', 'en_us'), ('en-US', 'en_us'), ('en-GB', 'en_gb'), ('en_au', 'en_gb'), ('pt', 'pt_br'),
    ('pt-PT', 'pt_br'), ('fr-CA', 'fr_fr'), ('zh', 'cmn'), ('zh-Hans', 'cmn'), ('nb', 'no'), ('nn', 'no'),
    ('fil', 'tl'), ('iw', 'he'), ('ja', 'ja'), ('vi', 'vi'),
])
def test_language_tags(tag, code):
    assert voices.language_for(tag).code == code


@pytest.mark.parametrize('tag', ['', None, 'xx', 'yi', 'ca'])
def test_languages_it_does_not_speak(tag):
    assert voices.language_for(tag) is None


def test_every_manifest_language_resolves():
    resolved = {voices.language_for(tag).code for tag in voices.LANGUAGES}
    assert resolved == set(voices.LANGUAGE_BY_CODE) - {'en_gb'}
    assert len(voices.LANGUAGES) == 43
