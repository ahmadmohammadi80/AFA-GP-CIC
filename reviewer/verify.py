"""Verify all 360 seed-method records against pinned 2026-11-18 CI science fields.

Wall clock excluded. Historical CI: 44360746204; source frozen at 498be2bc.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ('greedy', 'joint_mix', 'afa_gated_two_step')
FIELDS = ('seed', 'method', 'certified', 'false_certificate', 'labels',
          'steps', 'winner', 'coverage_breached', 'planner_calls',
          'positive_plans', 'root_overrides', 'online_decisions', 'query_counts')


def validate(folder: Path, manifest_file: Path):
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    report = {'reference_run': manifest['reference_run'], 'datasets': {},
              'all_pass': True, 'total_scientific_rows': 0}
    for name, spec in manifest['datasets'].items():
        file = folder / name
        problems = []
        if not file.is_file():
            problems.append('missing result file')
            report['datasets'][name] = {'pass': False, 'problems': problems}
            report['all_pass'] = False
            continue
        data = json.loads(file.read_text(encoding='utf-8'))
        proto = data.get('protocol', {})
        for k, expected in spec['protocol'].items():
            if proto.get(k) != expected:
                problems.append(f'protocol.{k}: expected {expected!r}, got {proto.get(k)!r}')
        rows = []
        for m in MODES:
            mr = data.get('rows', {}).get(m, [])
            if len(mr) != spec['protocol']['seeds']:
                problems.append(f'{m}: expected {spec["protocol"]["seeds"]} rows, got {len(mr)}')
            counts = sum(bool(r.get('certified')) for r in mr)
            if counts != spec['certified'][m]:
                problems.append(f'{m}: certified {counts}, expected {spec["certified"][m]}')
            means = sum(r.get('labels', 0) for r in mr) / max(len(mr), 1)
            if abs(means - spec['mean_labels'][m]) > 1e-8:
                problems.append(f'{m}: mean labels {means}, expected {spec["mean_labels"][m]}')
            if any(r.get('false_certificate', False) for r in mr):
                problems.append(f'{m}: false_certificate observed')
            for idx, row in enumerate(mr):
                if row.get('method') != m:
                    problems.append(f'{m}[{idx}]: invalid method')
                if row.get('seed') != proto.get('offset', 0) + idx:
                    problems.append(f'{m}[{idx}]: seed order mismatch')
                if row.get('labels', -1) > proto.get('budget', -1):
                    problems.append(f'{m}[{idx}]: exceeded label budget')
                try:
                    rows.append([row[field] for field in FIELDS])
                except KeyError as ex:
                    problems.append(f'{m}[{idx}]: missing {ex}')
        digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        if digest != spec['scientific_rows_sha256']:
            problems.append('scientific per-seed SHA256 mismatch')
        report['datasets'][name] = {'pass': not problems, 'problems': problems,
                                    'records': len(rows), 'scientific_rows_sha256': digest}
        report['total_scientific_rows'] += len(rows)
        if problems:
            report['all_pass'] = False
    if report['total_scientific_rows'] != 360:
        report['all_pass'] = False
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Verify all 360 AFA benchmark seed-method records')
    p.add_argument('--results', type=Path, default=Path('results/reviewer'))
    p.add_argument('--manifest', type=Path, default=HERE / 'reference_manifest.json')
    p.add_argument('--report', type=Path, default=Path('results/reviewer/replay_report.json'))
    args = p.parse_args()
    result = validate(args.results, args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['all_pass'] else 1)
