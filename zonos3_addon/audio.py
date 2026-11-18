"""Read and write PCM WAV with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Subtitld's own reference clips are 16 kHz mono PCM WAV, which the speaker
encoder takes as they are (so does 44.1 kHz). Anything else is converted with
ffmpeg first (see __main__).
"""

from __future__ import annotations

import sys
import wave
from array import array
from pathlib import Path


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


def read_wav(path) -> tuple[array, int]:
    """(mono samples as float32 in [-1, 1], sample rate) of 16-bit PCM WAV."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width != 2:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    pcm = array('h')
    pcm.frombytes(data)
    if sys.byteorder != 'little':
        pcm.byteswap()
    if not pcm:
        raise UnreadableAudio(f'{path}: no audio')
    if channels == 1:
        return array('f', (x / 32768.0 for x in pcm)), rate
    frames = len(pcm) // channels
    return array('f', (sum(pcm[i * channels:(i + 1) * channels]) / (32768.0 * channels)
                       for i in range(frames))), rate


def write_wav(path: Path, samples: array, rate: int) -> None:
    """16-bit mono PCM, written beside `path` and renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    pcm = array('h', (int(round(max(-1.0, min(1.0, x)) * 32767.0)) for x in samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    with wave.open(str(part), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    part.replace(path)
