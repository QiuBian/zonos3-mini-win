"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$ZONOS3_TEST_BINARY names the binary; without it these tests skip (and the
release workflow refuses skipped tests). The binary downloads the model into
$SUBTITLD_MODELS_DIR (1.2 GB) with nothing but what is in its own bundle,
then speaks lines in several languages and scripts, keeps a short line short,
clones a clip and cancels a line.

    ZONOS3_TEST_BINARY=dist/zonos3-addon/zonos3-addon pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('ZONOS3_TEST_BINARY', '')
pytestmark = pytest.mark.skipif(not BINARY, reason='ZONOS3_TEST_BINARY is not set')

LINES = [
    ('en-US', 'zonos3-wyatt', 'The train leaves at 10:45 from platform 3.'),
    ('de', 'zonos3-liesel', 'Guten Morgen, wie geht es Ihnen heute?'),
    ('pt-BR', 'zonos3-thiago', 'Você viu onde eu deixei as chaves?'),
    ('ja', 'zonos3-sakura', 'おはようございます、今日はいい天気ですね。'),
    ('hi', 'zonos3-ananya', 'नमस्ते, आज आप कैसे हैं?'),
    ('ru', 'zonos3-rupert', 'Привет, как у тебя дела сегодня?'),
]


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path))
        env.setdefault('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def say(self, req_id: str, out: Path, **params) -> dict:
        self.send({'id': req_id, 'type': 'tts.synthesize', 'params': {**params, 'output_path': str(out)}})
        return self.until_terminal(req_id)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('zonos3-addon ')


def test_self_test_loads_the_runtime():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['voices'] == 28 and report['runtime'].startswith('1 ')


def test_speaks_clones_and_cancels(tmp_path):
    addon = Addon(tmp_path, timeout=3600)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'zonos3'
        addon.send({'type': 'ready', 'host': 'test'})

        for n, (language, voice, text) in enumerate(LINES):
            out = tmp_path / f'{n}.wav'
            result = addon.say(str(n), out, text=text, voice=voice, language=language, seed=5)
            assert result['type'] == 'result', (language, result)
            assert result['data']['sample_rate'] == 44100 and 1.0 < result['data']['duration_sec'] < 8, language
            with wave.open(str(out)) as clip:
                assert clip.getframerate() == 44100 and clip.getnframes() > 44100, language

        # Seed 1 runs on to 2.9 s without the alignment check and the tail cut.
        short = addon.say('s', tmp_path / 's.wav', text='Hello there.', voice='zonos3-savannah', language='en', seed=1)
        assert short['type'] == 'result' and 0.5 < short['data']['duration_sec'] < 2.0, short

        clone = tmp_path / 'clone.wav'
        result = addon.say('b', clone, text='See you tomorrow.', voice='zonos3-clone', language='en', seed=5,
                           voice_ref_audio=str(tmp_path / '0.wav'))
        assert result['type'] == 'result', result
        assert clone.stat().st_size > 1000

        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Speaking...' and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'c'})

        addon.send({'id': 'c', 'type': 'tts.synthesize', 'params': {
            'text': 'This line is long enough to be interrupted before it ends. ' * 6,
            'voice': 'zonos3-imogen', 'language': 'en', 'output_path': str(tmp_path / 'c.wav')}})
        error = addon.until_terminal('c', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
        assert not (tmp_path / 'c.wav').exists()
    finally:
        assert addon.close() == 0
