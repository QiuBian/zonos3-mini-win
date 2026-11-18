"""Small pieces: audio files, the voice check, the host's rate, text normalization, TLS.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from zonos3_addon import __main__ as addon  # noqa: E402
from zonos3_addon import audio, engine, tls, voices  # noqa: E402


def _tone(seconds: float, rate: int = 44100, level: float = 0.5) -> array:
    return array('f', (level * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(seconds * rate))))


def test_wav_round_trip(tmp_path):
    samples = _tone(0.5)
    audio.write_wav(tmp_path / 'a.wav', samples, 44100)
    back, rate = audio.read_wav(tmp_path / 'a.wav')
    assert rate == 44100 and len(back) == len(samples)
    assert max(abs(a - b) for a, b in zip(back, samples)) < 1e-4
    assert not (tmp_path / 'a.wav.part').exists()


def test_stereo_wav_is_mixed_down(tmp_path):
    path = tmp_path / 's.wav'
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(array('h', [16384, 0] * 100).tobytes())
    samples, rate = audio.read_wav(path)
    assert rate == 16000 and len(samples) == 100 and abs(samples[0] - 0.25) < 1e-6


def test_not_a_wav(tmp_path):
    path = tmp_path / 'x.wav'
    path.write_bytes(b'not audio')
    with pytest.raises(audio.UnreadableAudio):
        audio.read_wav(path)


def test_voiced_seconds_counts_only_the_voice():
    take = _tone(0.3) + array('f', bytes(4 * 44100))
    assert abs(engine.voiced_seconds(take) - 0.3) < 0.03
    assert engine.voiced_seconds(array('f', bytes(4 * 44100))) == 0
    assert engine.voiced_seconds(_tone(1.0, level=0.005)) == 0


def test_the_hosts_rate_is_the_speaking_rate():
    assert engine.speaking_rate(0) == -1.0
    assert engine.speaking_rate(50) == pytest.approx(14.5 * 1.5)
    assert engine.speaking_rate(-100) == pytest.approx(14.5 * 0.5)
    assert engine.speaking_rate(300) == pytest.approx(14.5 * 2.0)
    assert addon._rate({'rate': 'fast'}) == 0.0 and addon._rate({'rate': -20}) == -20.0


def test_seed():
    assert addon._seed({}) == 1 and addon._seed({'seed': -1}) == 0xFFFFFFFF and addon._seed({'seed': 'x'}) == 1


@pytest.mark.parametrize('code,text,spoken', [
    ('en_us', 'It costs $5.32 at 10:45.', 'five dollars thirty two cents'),
    ('de', 'Am 3. Oktober.', 'dritten Oktober'),
    ('pt_br', 'Custa R$ 12,50.', 'doze reais'),
    ('en_us', '<i>Hello</i> there', 'Hello there'),
])
def test_text_is_normalized(code, text, spoken):
    assert spoken in engine.normalize(text, voices.LANGUAGE_BY_CODE[code])


def test_ca_bundle_only_for_frozen_linux_and_macos(monkeypatch):
    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.delenv('SSL_CERT_DIR', raising=False)
    assert tls.configure_ca_bundle(frozen=False, platform='linux') is None
    assert tls.configure_ca_bundle(frozen=True, platform='win32') is None
