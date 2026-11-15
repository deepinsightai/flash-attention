"""Small, dependency-free checks for a benchmark CSV before analysis."""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

REQUIRED = ('method', 'N', 'runtime_ms')


def audit_csv(path):
    """Return a report of data-quality issues without modifying the input."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED if name not in columns]
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(missing))
        issues = []
        counts = Counter()
        total = 0
        for line, row in enumerate(reader, start=2):
            total += 1
            method = (row.get('method') or '').strip()
            if not method:
                issues.append(f'line {line}: empty method')
            try:
                n = int(row['N'])
                if n <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                issues.append(f'line {line}: N must be a positive integer')
            try:
                runtime = float(row['runtime_ms'])
                if not math.isfinite(runtime) or runtime <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                issues.append(f'line {line}: runtime_ms must be finite and positive')
            if method:
                counts[method] += 1
        return {'rows': total, 'methods': dict(sorted(counts.items())),
                'issues': issues, 'columns': columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', help='CSV containing method, N, runtime_ms')
    args = parser.parse_args()
    try:
        report = audit_csv(args.csv_file)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Cannot audit CSV: {exc}\n')
    print(f"Rows: {report['rows']}")
    for method, count in report['methods'].items():
        print(f'  {method}: {count}')
    if report['issues']:
        print('Issues:')
        for issue in report['issues']:
            print('  ' + issue)
        parser.exit(1)
    print('No row-level issues detected in the required fields.')


if __name__ == '__main__':
    main()
