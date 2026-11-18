#!/usr/bin/env python3
"""Write the notices of the Rust crates zyphra-textnorm's extension links statically.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

cargo-about (https://github.com/EmbarkStudios/cargo-about) reads the Cargo.lock
of zyphra-textnorm at the tag of the version pyproject.toml pins and gathers
each crate's licence text for the extension's crate (python/), for the targets
the add-on ships (tools/cargo-about/about.toml). A crate under a licence not
accepted there fails the run. zyphra-textnorm's own LICENSE, NOTICE and
licenses/ come from the installed wheel at release time
(tools/assemble_licences.py).

The result is committed as THIRD_PARTY_LICENSES/zyphra-textnorm/CRATES.txt;
run this again whenever pyproject.toml moves to another version.

    cargo install --locked cargo-about --features cli
    git clone --branch v<version> https://github.com/Zyphra/zyphra-textnorm /tmp/zyphra-textnorm
    python tools/crate_licences.py /tmp/zyphra-textnorm
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABOUT = REPO_ROOT / 'tools' / 'cargo-about'
OUT = REPO_ROOT / 'THIRD_PARTY_LICENSES' / 'zyphra-textnorm'


def pinned_version() -> str:
    text = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    m = re.search(r'"zyphra-textnorm==([^"]+)"', text)
    if not m:
        raise SystemExit('FATAL: pyproject.toml pins no zyphra-textnorm version')
    return m.group(1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('upstream', type=Path, help='a checkout of Zyphra/zyphra-textnorm at the pinned tag')
    args = parser.parse_args(argv)

    version = pinned_version()
    tag = subprocess.run(['git', '-C', str(args.upstream), 'describe', '--tags', '--exact-match'],
                         capture_output=True, text=True).stdout.strip()
    if tag != f'v{version}':
        raise SystemExit(f'FATAL: {args.upstream} is at {tag or "no tag"}, expected v{version}')

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        template = Path(tmp) / 'about.hbs'
        template.write_text((ABOUT / 'about.hbs').read_text(encoding='utf-8').replace('@VERSION@', version),
                            encoding='utf-8')
        subprocess.run(['cargo', 'about', 'generate', '--locked', '--fail', '-c', str(ABOUT / 'about.toml'),
                        str(template), '-o', str(OUT / 'CRATES.txt')],
                       cwd=args.upstream / 'python', check=True)
    print(f"wrote {OUT / 'CRATES.txt'} ({(OUT / 'CRATES.txt').stat().st_size:,} bytes)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
