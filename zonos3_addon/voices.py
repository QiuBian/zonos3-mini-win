"""The voices the add-on offers, and the languages ZONOS3-Mini speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

- `zonos3-clone` clones the speaker from the video: the host hands over a clip
  of their speech (or Configure names one), and the speaker encoder turns it
  into a voice.
- The 27 stock voices are Zyphra's default voices, carried as their speaker
  embeddings (voices.json, written by tools/make_voices.py). Each was recorded
  in one language but speaks every language, with that language's accent.

The language is not an input of the model: it picks the text normalization
grammar and the alignment heads LACI reads, and short-text fillers would use
it. LANGUAGES mirrors upstream's registry (model/src/zonos3/registries/
languages.json, `server`), which tests/test_engine.py checks against the copy
build/assets carries.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

CLONE = 'zonos3-clone'
SAMPLE_RATE = 44100
_VOICES_FILE = Path(__file__).resolve().parent / 'voices.json'


@dataclass(frozen=True)
class Language:
    code: str    # upstream's server code: en_us, pt_br, cmn...
    name: str
    nemo: str    # zyphra_textnorm's grammar
    speech: str  # the alignment heads: laci_heads/zonos3_mini_heads_<speech>.json (en: zonos3_mini_heads.json)


_TABLE = (
    ('en_us', 'English', 'en', 'en'), ('en_gb', 'British English', 'en', 'en'),
    ('fr_fr', 'French', 'fr', 'fr'), ('de', 'German', 'de', 'de'), ('es', 'Spanish', 'es', 'es'),
    ('it', 'Italian', 'it', 'it'), ('pt_br', 'Portuguese', 'pt', 'pt'), ('ja', 'Japanese', 'ja', 'ja'),
    ('cmn', 'Chinese (Mandarin)', 'zh', 'zh'), ('ko', 'Korean', 'ko', 'ko'), ('ar', 'Arabic', 'ar', 'ar'),
    ('bn', 'Bengali', 'bn', 'bn'), ('bg', 'Bulgarian', 'bg', 'bg'), ('cs', 'Czech', 'cs', 'cs'),
    ('da', 'Danish', 'da', 'da'), ('nl', 'Dutch', 'nl', 'nl'), ('fi', 'Finnish', 'fi', 'fi'),
    ('ka', 'Georgian', 'ka', 'ka'), ('el', 'Greek', 'el', 'el'), ('gu', 'Gujarati', 'gu', 'gu'),
    ('he', 'Hebrew', 'he', 'he'), ('hi', 'Hindi', 'hi', 'hi'), ('hu', 'Hungarian', 'hu', 'hu'),
    ('id', 'Indonesian', 'id', 'id'), ('kn', 'Kannada', 'kn', 'kn'), ('ms', 'Malay', 'ms', 'ms'),
    ('ml', 'Malayalam', 'ml', 'ml'), ('mr', 'Marathi', 'mr', 'mr'), ('no', 'Norwegian', 'nb', 'no'),
    ('or', 'Odia', 'or', 'or'), ('pl', 'Polish', 'pl', 'pl'), ('pa', 'Punjabi', 'pa', 'pa'),
    ('ro', 'Romanian', 'ro', 'ro'), ('ru', 'Russian', 'ru', 'ru'), ('sk', 'Slovak', 'sk', 'sk'),
    ('sv', 'Swedish', 'sv', 'sv'), ('tl', 'Tagalog', 'tl', 'tl'), ('ta', 'Tamil', 'ta', 'ta'),
    ('te', 'Telugu', 'te', 'te'), ('th', 'Thai', 'th', 'th'), ('tr', 'Turkish', 'tr', 'tr'),
    ('uk', 'Ukrainian', 'uk', 'uk'), ('ur', 'Urdu', 'ur', 'ur'), ('vi', 'Vietnamese', 'vi', 'vi'),
)
LANGUAGE_BY_CODE = {row[0]: Language(*row) for row in _TABLE}

# The ISO 639-1 primary tag of a request -> upstream's code. Regions only matter for English.
_PRIMARY = {lang.code.split('_')[0]: lang.code for lang in LANGUAGE_BY_CODE.values() if lang.code != 'en_gb'}
_PRIMARY.update({'en': 'en_us', 'fr': 'fr_fr', 'pt': 'pt_br', 'zh': 'cmn', 'cmn': 'cmn',
                 'nb': 'no', 'nn': 'no', 'no': 'no', 'fil': 'tl',
                 'iw': 'he', 'in': 'id', 'zsm': 'ms'})
_BRITISH_REGIONS = {'gb', 'uk', 'ie', 'au', 'nz', 'za', 'in'}

# The manifest's languages: one ISO 639-1 tag per language upstream speaks.
LANGUAGES = ('en', 'fr', 'de', 'es', 'it', 'pt', 'ja', 'zh', 'ko', 'ar', 'bn', 'bg', 'cs', 'da', 'nl', 'fi',
             'ka', 'el', 'gu', 'he', 'hi', 'hu', 'id', 'kn', 'ms', 'ml', 'mr', 'nb', 'or', 'pl', 'pa', 'ro',
             'ru', 'sk', 'sv', 'tl', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi')


def split_tag(tag) -> tuple[str, str]:
    """`pt-BR` -> ('pt', 'br'). Lower-cased; ('', '') when there is none."""
    parts = str(tag or '').strip().lower().replace('_', '-').split('-')
    region = next((p for p in parts[1:] if len(p) == 2 or p.isdigit()), '')
    return parts[0], region


def language_for(tag) -> Language | None:
    """Upstream's language for a request's language tag, or None when ZONOS3-Mini doesn't speak it."""
    primary, region = split_tag(tag)
    code = _PRIMARY.get(primary)
    if code is None:
        return None
    if code == 'en_us' and region in _BRITISH_REGIONS:
        code = 'en_gb'
    return LANGUAGE_BY_CODE[code]


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    embedding: bytes = b''   # 192 little-endian float32; empty for the clone


_stock: tuple[Voice, ...] | None = None


def stock() -> tuple[Voice, ...]:
    global _stock
    if _stock is None:
        data = json.loads(_VOICES_FILE.read_text(encoding='utf-8'))
        voices = []
        for entry in data['voices']:
            described = ', '.join(x for x in (entry['style'], entry['language'], entry['gender']) if x)
            voices.append(Voice(entry['id'], f"{entry['name']} ({described})", base64.b64decode(entry['embedding'])))
        _stock = tuple(voices)
    return _stock


def all_voices() -> tuple[Voice, ...]:
    return (Voice(CLONE, "Speaker's own voice (cloned from the video)"), *stock())


def by_id(voice_id: str) -> Voice | None:
    return next((v for v in all_voices() if v.id == voice_id), None)


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in all_voices():
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name, 'sample_rate': SAMPLE_RATE}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out
