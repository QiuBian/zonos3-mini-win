"""The native runtime tools/build_native.py built, and the upstream data beside it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Needs build/assets/ (or $ZONOS3_ASSETS). With $ZONOS3_TEST_MODELS_DIR (a
$SUBTITLD_MODELS_DIR that holds the downloaded model) it also speaks a few
lines in-process; tests/test_installed.py does the same through the frozen
binary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from zonos3_addon import engine, voices  # noqa: E402

NATIVE = json.loads((engine.assets_dir() / 'native.json').read_text(encoding='utf-8'))
MODELS = os.environ.get('ZONOS3_TEST_MODELS_DIR', '')


def test_the_built_library_loads_and_reports_its_commit():
    assert engine.library_path().is_file()
    interface, commit = engine.runtime_version().split()
    assert interface == engine.INTERFACE and commit == NATIVE['commit']


def test_the_language_table_is_upstreams():
    registry = json.loads((engine.assets_dir() / 'zonos3' / 'languages.json').read_text(encoding='utf-8'))
    upstream = {row['code']: (row['nemo'], row['speech']) for row in registry['server']}
    ours = {lang.code: (lang.nemo, lang.speech) for lang in voices.LANGUAGE_BY_CODE.values()}
    assert ours == upstream


def test_every_language_has_alignment_heads():
    for language in voices.LANGUAGE_BY_CODE.values():
        heads = engine.laci_heads(language)
        assert heads.is_file(), language.code
        if language.speech != 'en':
            assert heads.name == f'zonos3_mini_heads_{language.speech}.json', language.code
        assert json.loads(heads.read_text(encoding='utf-8'))['heads']


@pytest.fixture(scope='module')
def model():
    if not MODELS:
        pytest.skip('ZONOS3_TEST_MODELS_DIR is not set')
    os.environ['SUBTITLD_MODELS_DIR'] = MODELS
    from zonos3_addon import __main__ as addon
    loaded = engine.Zonos3(addon._file('backbone-ptq.gguf'), addon._file('zac-keep2.gguf'),
                           addon._file('spk-encoder.gguf'), engine.default_threads())
    yield loaded
    loaded.close()


def test_a_short_line_ends_with_the_text(model):
    english = voices.LANGUAGE_BY_CODE['en_us']
    samples, record = model.synthesize('Hello there.', english, voices.by_id('zonos3-savannah').embedding, seed=1)
    # Without LACI and the tail cut, seed 1 runs on to 2.9 s ("Hello there. I was a stud.").
    assert 0.5 < len(samples) / model.sample_rate < 2.0, record
    assert record['laci'] is True


def test_cloning_and_cancel(model):
    english = voices.LANGUAGE_BY_CODE['en_us']
    samples, _ = model.synthesize('Where were you last night?', english, voices.by_id('zonos3-wyatt').embedding)
    embedding = model.embed(samples, model.sample_rate)
    assert len(embedding) == 4 * engine.SPEAKER_DIM
    cancel = engine.Cancel()
    cancel.set()
    with pytest.raises(engine.Cancelled):
        model.synthesize('This will not be spoken.', english, embedding, cancel=cancel)
