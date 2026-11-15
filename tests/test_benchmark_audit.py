"""Tests for the optional benchmark CSV audit."""
import csv
import tempfile
import unittest
from pathlib import Path
from analysis.benchmark_audit import audit_csv


class AuditTests(unittest.TestCase):
    def write_csv(self, rows, columns=('method', 'N', 'runtime_ms')):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'runs.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_rows(self):
        path = self.write_csv([
            {'method': 'naive', 'N': '1024', 'runtime_ms': '3.5'},
            {'method': 'flash', 'N': '1024', 'runtime_ms': '1.2'},
        ])
        result = audit_csv(path)
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['methods']['flash'], 1)

    def test_invalid_numeric_values(self):
        path = self.write_csv([
            {'method': 'naive', 'N': '0', 'runtime_ms': 'nan'},
            {'method': '', 'N': 'bad', 'runtime_ms': '-1'},
        ])
        self.assertEqual(len(audit_csv(path)['issues']), 5)

    def test_missing_required_column(self):
        path = self.write_csv([], columns=('method', 'N'))
        with self.assertRaisesRegex(ValueError, 'runtime_ms'):
            audit_csv(path)


if __name__ == '__main__':
    unittest.main()
