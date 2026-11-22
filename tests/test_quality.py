import unittest
import tempfile
from pathlib import Path
from study.quality import check_results

class QualityTests(unittest.TestCase):
    def test_fails_when_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            q=check_results(Path(folder))
            self.assertEqual(q['status'],'FAIL')
    def test_fails_nonexistent_output(self):
        q=check_results(Path('/some/path/does/not/exist'))
        self.assertEqual(q['status'],'FAIL')

if __name__=='__main__':unittest.main()
