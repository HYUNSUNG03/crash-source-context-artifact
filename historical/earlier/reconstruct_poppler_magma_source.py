"""Reconstruct the static Poppler source used by Magma; never build or run it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ARTIFACTS = Path('/RESEARCH_HOME/gptrace-artifacts')
SOURCE_INDEX = ARTIFACTS / 'reproduce/source_index'
BASE = SOURCE_INDEX / 'poppler-1d23101c'
DESTINATION = SOURCE_INDEX / 'poppler-magma-v1.2.1-patched'
MAGMA = ARTIFACTS / 'reproduce/generate_data_sources/magma'
# Configure this release-local output location for the historical reconstruction.
OUTPUT = Path('/RESEARCH_OUTPUT')
BASE_COMMIT = '1d23101ccebe14261c6afc024ea14f29d209e760'
BASE_TREE = '705396a7b5dbb32ad5debb456131a7f5a5da3206'
MAGMA_COMMIT = '1e4d8c248debacd7a5c4b157a1059d5290bd3a22'
PATCHES = [
    'PDF013', 'PDF011', 'PDF005', 'PDF001', 'PDF014', 'PDF007',
    'PDF009', 'PDF017', 'PDF021', 'PDF022', 'PDF015', 'PDF004',
    'PDF016', 'PDF010', 'PDF006', 'PDF002', 'PDF008', 'PDF019',
    'PDF003', 'PDF018', 'PDF012', 'PDF020',
]


def run(arguments, *, check=True):
    result = subprocess.run(arguments, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(
            f'Command failed ({result.returncode}): {arguments}\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_value(repo, *args):
    return run(['git', '-C', str(repo), *args]).stdout.strip()


def main():
    if DESTINATION.exists():
        raise SystemExit(f'Refusing to overwrite existing destination: {DESTINATION}')
    if git_value(BASE, 'rev-parse', 'HEAD') != BASE_COMMIT:
        raise SystemExit('Unexpected Poppler base commit')
    if git_value(BASE, 'rev-parse', 'HEAD^{tree}') != BASE_TREE:
        raise SystemExit('Unexpected Poppler base tree')
    if git_value(MAGMA, 'rev-parse', 'HEAD') != MAGMA_COMMIT:
        raise SystemExit('Unexpected Magma commit')
    patch_version = run(['patch', '--version']).stdout.splitlines()[0]
    staging = Path(tempfile.mkdtemp(prefix='poppler-magma-building-', dir=SOURCE_INDEX))
    # mkdtemp creates the directory; git clone requires a missing destination.
    staging.rmdir()
    run(['git', 'clone', '--quiet', '--no-hardlinks', str(BASE), str(staging)])
    run(['git', '-C', str(staging), 'checkout', '--quiet', '--detach', BASE_COMMIT])
    logs = []
    try:
        with tempfile.TemporaryDirectory(prefix='rendered-poppler-patches-') as temp_dir:
            temp_root = Path(temp_dir)
            for name in PATCHES:
                original = MAGMA / 'targets/poppler/patches/bugs' / f'{name}.patch'
                if not original.is_file():
                    raise FileNotFoundError(original)
                rendered_text = original.read_text().replace('%MAGMA_BUG%', name)
                if '%MAGMA_BUG%' in rendered_text:
                    raise ValueError(f'Unresolved patch placeholder in {name}')
                rendered = temp_root / f'{name}.patch'
                rendered.write_text(rendered_text)
                common = ['patch', '--batch', '--forward', '-p1', '-d', str(staging), '-i', str(rendered)]
                dry = run([*common[:1], '--dry-run', *common[1:]])
                applied = run(common)
                logs.append({
                    'name': name,
                    'original_sha256': sha256(original),
                    'rendered_sha256': sha256(rendered),
                    'dry_run_stdout': dry.stdout,
                    'dry_run_stderr': dry.stderr,
                    'apply_stdout': applied.stdout,
                    'apply_stderr': applied.stderr,
                })
        diff = run(['git', '-C', str(staging), 'diff', '--binary', '--no-ext-diff']).stdout.encode()
        anchors = {}
        for relative in (
            'poppler/GfxState.cc', 'utils/ImageOutputDev.cc', 'poppler/XRef.cc',
            'poppler/Annot.h', 'poppler/Annot.cc', 'poppler/JBIG2Stream.cc',
        ):
            anchors[relative] = sha256(staging / relative)
        staging.rename(DESTINATION)
        manifest = {
            'status': 'static_source_reconstruction_complete',
            'scope': 'Source mapping only; no configure, build, target, or crashing input executed.',
            'destination': str(DESTINATION),
            'base_commit': BASE_COMMIT, 'base_tree': BASE_TREE,
            'magma_commit': MAGMA_COMMIT, 'patch_program': patch_version,
            'patch_order': PATCHES, 'patches': logs,
            'git_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'git_diff_bytes': len(diff), 'anchor_file_sha256': anchors,
            'limitations': [
                'Patch order is explicitly pinned because upstream apply_patches.sh uses unsorted find output.',
                'Coordinate/function correspondence must still be checked against every saved report.',
                'No hash of the original build container final source was available for byte-for-byte comparison.',
            ],
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / 'poppler-patched-source-manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps({key: value for key, value in manifest.items() if key != 'patches'}, indent=2))
    except Exception:
        print(f'Partial staging directory retained for diagnosis: {staging}')
        raise


if __name__ == '__main__':
    main()
