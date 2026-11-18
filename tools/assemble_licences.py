#!/usr/bin/env python3
"""Assemble THIRD_PARTY_LICENSES/ for a release archive, and refuse native code without a notice.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Besides the add-on's own code, the frozen bundle carries libzonos3_c: Zyphra's
ZONOS3-Mini runtime (MIT) with ggml (MIT), LMDB (OpenLDAP), pocketfft
(BSD-3-Clause) and nlohmann/json (MIT) compiled in, and on Linux parts of the
GCC runtime; Zyphra's zyphra-textnorm extension (MIT, Apache-2.0, Unicode-3.0)
with the Rust crates it links; CPython with the native libraries its standard
modules link against (OpenSSL among them, for the download), the PyInstaller
bootloader (GPL-2.0-or-later with the bootloader exception) and, on Windows,
the C runtime. The model is downloaded, not bundled; models/README.md
describes it.

The notices committed in THIRD_PARTY_LICENSES/ are copied as they are;
ZONOS3-Mini's own (its licence, NOTICE and the licences of what the library
compiles in) from build/licences/zonos3/, where tools/build_native.py puts
them; zyphra-textnorm's licence and notices from its installed wheel, beside
the committed CRATES.txt. CPython's licence comes from the interpreter running
this script (the one that froze the bundle), and PyInstaller's from its
installed distribution.

Then every native file in the bundle (ELF, Mach-O or PE, whatever its name)
must match a rule below, and the notice that rule names must be in the output.
Anything else fails the build: a library new to the bundle has to come with its
licence before it ships. The result is indexed in THIRD_PARTY_LICENSES/README.md.

    python tools/assemble_licences.py --bundle dist/zonos3-addon \
        --out release-staging/THIRD_PARTY_LICENSES
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ZONOS3_LICENCES = REPO_ROOT / 'build' / 'licences' / 'zonos3'

# (pattern on the lower-cased file name, notice inside THIRD_PARTY_LICENSES/, component)
RULES = [
    (r'^zonos3-addon(\.exe)?$', 'pyinstaller/COPYING.txt', 'PyInstaller bootloader'),
    (r'^(lib)?zonos3_c\.(so|dylib|dll)$', 'zonos3/LICENSE', 'ZONOS3-Mini runtime with ggml'),
    (r'^zyphra_textnorm(\.abi3)?\.(so|pyd)$', 'zyphra-textnorm/LICENSE', 'zyphra-textnorm'),
    # The GCC runtime that zyphra-textnorm's Rust extension links (unwinding), from the build system's GCC 8.
    (r'^libgcc_s\.so(\.\d+)*$', 'gcc-runtime/COPYING.RUNTIME', 'GCC runtime (libgcc_s)'),
    (r'^(libpython3[\d.]*\.(so|dylib)[\d.]*|python3\d*\.dll|python)$', 'python/LICENSE.txt', 'CPython'),
    (r'(\.cpython-[^/]+\.(so|dylib)|\.pyd)$', 'python/LICENSE.txt', 'CPython extension module'),
    (r'^libffi[-.\d]*\.(so|dylib|dll)', 'python-runtime-libs/libffi-LICENSE', 'libffi'),
    (r'^libexpat[-.\d]*\.(so|dylib|dll)', 'python-runtime-libs/expat-COPYING', 'Expat'),
    (r'^(libz\.(so|dylib)|zlib\d*\.dll)', 'python-runtime-libs/zlib-LICENSE', 'zlib'),
    # OpenSSL 1.1.1 (the Linux build's, from AlmaLinux 8) is under the OpenSSL
    # and SSLeay licences; OpenSSL 3 under Apache-2.0.
    (r'^lib(ssl|crypto)\.so\.1\.1', 'python-runtime-libs/openssl-1.1-LICENSE.txt', 'OpenSSL 1.1.1'),
    (r'^lib(ssl|crypto)[-.\d]', 'python-runtime-libs/openssl-LICENSE.txt', 'OpenSSL'),
    # Excluded from the bundle by pyinstaller.spec. Listed so that, should one
    # come back, the build names the notice to commit instead of a bare file.
    (r'^libbz2', 'python-runtime-libs/bzip2-LICENSE', 'bzip2'),
    (r'^liblzma', 'python-runtime-libs/xz-COPYING', 'XZ Utils'),
    (r'^libzstd', 'python-runtime-libs/zstd-LICENSE', 'zstd'),
    (r'^libmpdec', 'python-runtime-libs/mpdecimal-LICENSE', 'mpdecimal'),
    (r'^libuuid', 'python-runtime-libs/libuuid-COPYING', 'libuuid'),
    (r'^(vcruntime140(_1|_threads)?|msvcp140(_\w+)?|vcomp140|concrt140)\.dll$', 'msvc-runtime/NOTICE.txt',
     'Microsoft C runtime'),
    (r'^(ucrtbase|api-ms-win-[\w-]+)\.dll$', 'msvc-runtime/NOTICE.txt', 'Universal C runtime'),
]
_COMPILED = [(re.compile(pattern), notice, component) for pattern, notice, component in RULES]

_MACHO_MAGIC = {b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe',
                b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}


def is_native(path: Path) -> bool:
    with path.open('rb') as handle:
        head = handle.read(4)
    return head == b'\x7fELF' or head[:2] == b'MZ' or head in _MACHO_MAGIC


def rule_for(name: str) -> tuple[str, str] | None:
    lowered = name.lower()
    for pattern, notice, component in _COMPILED:
        if pattern.search(lowered):
            return notice, component
    return None


def cpython_licence() -> Path:
    stdlib = Path(sysconfig.get_paths()['stdlib'])
    for candidate in (Path(sys.base_prefix) / 'LICENSE.txt', stdlib / 'LICENSE.txt',
                      Path(sys.base_prefix) / 'LICENSE'):
        if candidate.is_file():
            return candidate
    raise SystemExit(f'FATAL: no CPython licence next to {sys.base_prefix} or in {stdlib}')


def pyinstaller_licence() -> tuple[Path, str]:
    try:
        dist = distribution('pyinstaller')
    except PackageNotFoundError:
        raise SystemExit('FATAL: PyInstaller is not installed in this interpreter') from None
    for entry in dist.files or []:
        if Path(str(entry)).name == 'COPYING.txt':
            path = Path(dist.locate_file(entry))
            if path.is_file():
                return path, dist.version
    raise SystemExit('FATAL: PyInstaller is installed but ships no COPYING.txt')


def textnorm_licences(dest: Path) -> None:
    """zyphra-textnorm's LICENSE, NOTICE and licenses/ from the installed wheel's dist-info."""
    try:
        dist = distribution('zyphra-textnorm')
    except PackageNotFoundError:
        raise SystemExit('FATAL: zyphra-textnorm is not installed in this interpreter') from None
    copied = 0
    for entry in dist.files or []:
        parts = Path(str(entry)).parts
        if len(parts) >= 3 and parts[0].endswith('.dist-info') and parts[1] == 'licenses':
            target = dest.joinpath(*parts[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist.locate_file(entry), target)
            copied += 1
    if not copied or not (dest / 'LICENSE').is_file():
        raise SystemExit(f'FATAL: zyphra-textnorm {dist.version} carries no licences in its dist-info')
    (dest / 'SOURCE.txt').write_text(f'zyphra-textnorm {dist.version}, from its wheel on PyPI; the Rust crates its\n'
                                     f'extension links are in CRATES.txt.\n', encoding='utf-8')


def assemble(bundle: Path, out: Path) -> list[tuple[str, str, str]]:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO_ROOT / 'THIRD_PARTY_LICENSES', out)
    if not (ZONOS3_LICENCES / 'LICENSE').is_file():
        raise SystemExit(f'FATAL: no {ZONOS3_LICENCES / "LICENSE"}; run tools/build_native.py')
    shutil.copytree(ZONOS3_LICENCES, out / 'zonos3')
    textnorm_licences(out / 'zyphra-textnorm')

    python_licence = cpython_licence()
    (out / 'python').mkdir()
    shutil.copyfile(python_licence, out / 'python' / 'LICENSE.txt')
    (out / 'python' / 'SOURCE.txt').write_text(
        f'CPython {platform.python_version()}, the interpreter that froze this add-on.\n'
        f'Licence copied from {python_licence}.\n', encoding='utf-8')
    licence, version = pyinstaller_licence()
    (out / 'pyinstaller').mkdir()
    shutil.copyfile(licence, out / 'pyinstaller' / 'COPYING.txt')
    (out / 'pyinstaller' / 'SOURCE.txt').write_text(
        f'PyInstaller {version}. Its bootloader is the add-on executable; the GPL-2.0\n'
        f'bootloader exception in COPYING.txt allows distributing it with this program.\n',
        encoding='utf-8')

    rows: list[tuple[str, str, str]] = []
    problems: list[str] = []
    for path in sorted(p for p in bundle.rglob('*') if p.is_file() and not p.is_symlink()):
        relative = 'bin/' + path.relative_to(bundle).as_posix()
        if not is_native(path):
            continue
        match = rule_for(path.name)
        if match is None:
            problems.append(f'{relative}: native file with no licence rule')
            continue
        notice, component = match
        if not (out / notice).is_file():
            problems.append(f'{relative}: {component} needs THIRD_PARTY_LICENSES/{notice}, which is not committed')
            continue
        rows.append((relative, component, notice))
    for component in ('ZONOS3-Mini runtime', 'zyphra-textnorm'):
        if not any(row[1].startswith(component) for row in rows):
            problems.append(f'the bundle has no {component}')
    required = ['zonos3/NOTICE', 'zonos3/third_party/ggml/LICENSE', 'zonos3/third_party/lmdb/LICENSE',
                'zonos3/third_party/pocketfft/LICENSE', 'zonos3/third_party/nlohmann-json/LICENSE',
                'zyphra-textnorm/NOTICE', 'zyphra-textnorm/CRATES.txt', 'zyphra-textnorm/licenses/nemo-Apache-2.0.txt',
                'models/README.md', 'models/Apache-2.0.txt']
    if sys.platform.startswith('linux'):
        # libstdc++ parts and libgcc are linked into libzonos3_c.
        required += ['gcc-runtime/COPYING.RUNTIME', 'gcc-runtime/COPYING3']
    for notice in required:
        if not (out / notice).is_file():
            problems.append(f'THIRD_PARTY_LICENSES/{notice} is not committed')
    if problems:
        raise SystemExit('FATAL: licences are incomplete:\n  ' + '\n  '.join(problems))

    lines = ['# Third-party licences', '',
             'Everything frozen into this add-on, and the notice in this folder that covers it.', '',
             '| File | Component | Notice |', '|------|-----------|--------|']
    lines += [f'| `{relative}` | {component} | [{notice}]({notice}) |' for relative, component, notice in rows]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True, help='the PyInstaller onedir output')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = assemble(args.bundle, args.out)
    print(f'{len(rows)} bundled file(s) covered:')
    for relative, component, notice in rows:
        print(f'  {relative}  ->  {notice}  ({component})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
