"""Create patched-source contexts with Magma instrumentation removed from text."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import re

import extract_contexts_v2 as v2
from extract_contexts_v3 import patched_source_maps

WORKSPACE = Path(__file__).resolve().parents[1]
BENCHMARK_MARKER = re.compile(r'\bMAGMA_[A-Z0-9_]+\b|\bPDF\d{3}\b')
DIRECTIVE = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')


def sanitized_location_window(parsed, line, radius=5):
    content = bytearray(parsed.content)
    for node in v2.descendants(parsed.tree.root_node):
        if node.type == 'comment':
            for offset in range(node.start_byte, node.end_byte):
                if content[offset] not in (10, 13):
                    content[offset] = 32
    lines = bytes(content).decode('utf-8', 'replace').splitlines()
    excluded, stack = set(), []
    for index, text in enumerate(lines):
        directive = DIRECTIVE.match(text)
        if directive and directive.group(1) in {'if', 'ifdef', 'ifndef'}:
            block_is_benchmark = bool(BENCHMARK_MARKER.search(text)) or any(stack)
            stack.append(block_is_benchmark)
            if block_is_benchmark:
                excluded.add(index)
            continue
        if directive and directive.group(1) in {'elif', 'else'}:
            if stack and (stack[-1] or BENCHMARK_MARKER.search(text)):
                stack[-1] = True
                excluded.add(index)
            continue
        if directive and directive.group(1) == 'endif':
            if any(stack):
                excluded.add(index)
            if stack:
                stack.pop()
            continue
        if any(stack) or BENCHMARK_MARKER.search(text):
            excluded.add(index)
    focus = line - 1
    if focus in excluded:
        raise ValueError('Crash focus line is benchmark instrumentation')
    start, end = max(0, focus - radius), min(len(lines), focus + radius + 1)
    selected = [('FOCUS: ' if index == focus else '') + lines[index].rstrip()
                for index in range(start, end) if index not in excluded]
    result = '\n'.join(selected)
    if BENCHMARK_MARKER.search(result):
        raise ValueError('Benchmark marker remains in sanitized L text')
    return result


def main():
    v2.source_maps = patched_source_maps
    v2.QUARANTINED_TARGET_PREFIXES = ()
    v2.location_window = sanitized_location_window
    original = [json.loads(line) for line in
                (WORKSPACE / 'outputs/l0-pilot-contexts.jsonl').read_text().splitlines()]
    results = []
    rejected_markers = 0
    for record in original:
        try:
            result = v2.extract(record)
            result['schema_version'] = 'context-v4'
            result['source_text_policy'] = 'comments_removed_and_magma_instrumentation_blocks_excluded'
            if result.get('target') == 'poppler__pdfimages' and result.get('source_validation', '').startswith('function_correspondence'):
                result['source_validation'] = 'magma_build_procedure_equivalent_function_correspondence'
            if any(BENCHMARK_MARKER.search(result.get(field) or '') for field in ('l_text', 'd_text')):
                rejected_markers += 1
                result.update({'status': 'benchmark_marker_rejected', 'l_text': None,
                               'd_text': None, 'l_text_sha256': None, 'd_text_sha256': None})
            else:
                for field in ('l', 'd'):
                    text = result.get(f'{field}_text')
                    if text:
                        result[f'{field}_text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            results.append(result)
        except Exception as error:
            results.append({key: record[key] for key in ('report_id', 'target', 'label', 'report_relative_path')}
                           | {'schema_version': 'context-v4', 'status': 'extraction_error',
                              'error': str(error), 'publication_ready': False,
                              'l_text': None, 'd_text': None})
    marker_occurrences = sum(bool(BENCHMARK_MARKER.search((row.get('l_text') or '') + (row.get('d_text') or '')))
                             for row in results)
    output = WORKSPACE / 'outputs/revised'
    (output / 'contexts-v4.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    summary = {
        'schema_version': 'context-v4', 'records': len(results), 'publication_ready': 0,
        'source_profile': 'Magma-patched Poppler plus original FreeType/SoX release sources',
        'source_text_policy': 'comments removed; MAGMA_* conditional blocks and PDF### identifiers excluded',
        'status_by_target': {target: dict(collections.Counter(row['status'] for row in results if row['target'] == target))
                             for target in sorted({row['target'] for row in results})},
        'ld_reports': sum(bool(row.get('l_text') and row.get('d_text')) for row in results),
        'l_only_reports': sum(bool(row.get('l_text') and not row.get('d_text')) for row in results),
        'neither_reports': sum(not row.get('l_text') and not row.get('d_text') for row in results),
        'benchmark_marker_rejected_records': rejected_markers,
        'benchmark_marker_occurrences_in_output': marker_occurrences,
        'limitations': [
            'Manual semantic review is still pending; all records remain publication_ready=false.',
            'D0 is syntax-based and does not prove runtime dependencies.',
            'Removing benchmark-only preprocessor blocks yields a semantic view, not the exact compiled text.',
        ],
    }
    (output / 'contexts-v4-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
