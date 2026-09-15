import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux server handoff')
class QueueHandoff(unittest.TestCase):
    def test_successor_starts_only_after_predecessor_naturally_finishes(self):
        script=Path(__file__).resolve().parents[1]/'src/queue_after_experiment.py'
        with tempfile.TemporaryDirectory(prefix='geometry_handoff_check_') as tmp:
            root=Path(tmp)/'output';root.mkdir()
            release=Path(tmp)/'release';(release/'src').mkdir(parents=True)
            (release/'src/run_unified_resident.py').write_text(
                "import sys\nfrom pathlib import Path\n"
                "p=Path(sys.argv[sys.argv.index('--root')+1])\n"
                "(p/'successor_started').write_text('ok')\n")
            predecessor=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'])
            watcher=subprocess.Popen([sys.executable,str(script),'--after-pid',str(predecessor.pid),
                '--release',str(release),'--root',str(root),'--python',sys.executable])
            try:
                time.sleep(.15)
                self.assertIsNone(predecessor.poll())
                self.assertFalse((root/'successor_started').exists())
                predecessor.wait(timeout=5)
                self.assertEqual(watcher.wait(timeout=8),0,(root/'run.log').read_text())
                self.assertTrue((root/'successor_started').exists())
                self.assertEqual(json.loads((root/'queue_status.json').read_text())['status'],'complete')
            finally:
                for process in (predecessor,watcher):
                    if process.poll() is None:process.terminate();process.wait(timeout=5)


if __name__=='__main__':unittest.main()
