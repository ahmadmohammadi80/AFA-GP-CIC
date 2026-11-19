"""Single-command reviewer reproduction: tests, four suites, fail-closed verification."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('online_primary40_3methods.json', 40, 20261116, []),
    ('online_holdout40_3methods.json', 40, 20270101, []),
    ('online_cyclic20_3methods.json', 20, 20261116, ['--cyclic']),
    ('online_null20_3methods.json', 20, 20261116, ['--null']),
]


def main():
    ap = argparse.ArgumentParser(description='Rerun frozen AFA-GP-CIC online experiments')
    ap.add_argument('--results', type=Path, default=ROOT / 'results' / 'reviewer')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    out = args.results.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {'python': platform.python_version(), 'numpy': version('numpy'),
           'scipy': version('scipy'), 'platform': platform.platform()}
    (out / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')
    if tuple(sys.version_info[:2]) != (3, 13) or (env['numpy'], env['scipy']) != ('2.3.5', '1.17.0'):
        raise SystemExit(f'Environment mismatch {env}: use Python 3.13 and reviewer/requirements.txt')
    if not args.verify_only:
        def call(command, cwd=ROOT):
            print('$', ' '.join(str(x) for x in command), flush=True)
            subprocess.run(command, cwd=cwd, check=True)
        call([sys.executable, '-m', 'unittest', '-v',
              'test_integrated_two_step_certificate_20261118',
              'test_online_two_step_benchmark_20261118'],
             cwd=ROOT / 'research' / 'six_issues_20261116')
        source = ROOT / 'research' / 'six_issues_20261116' / 'online_two_step_benchmark_20261118.py'
        for filename, n, offset, flags in CASES:
            call([sys.executable, str(source), '--seeds', str(n), '--budget', '1510',
                  '--offset', str(offset), *flags, '--out', str(out / filename)])
    call = [sys.executable, str(ROOT / 'reviewer' / 'verify.py'),
            '--results', str(out), '--report', str(out / 'replay_report.json')]
    print('$', ' '.join(call), flush=True)
    subprocess.run(call, cwd=ROOT, check=True)
    print('PASS: all 360 scientific method-seed records matched reference fingerprints')


if __name__ == '__main__':
    main()
