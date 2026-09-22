import contextlib, hashlib, io, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from causal_schedule_lab.m3.adaptive_search import begin_episode
from causal_schedule_lab.m3 import run_logging

class Tests(unittest.TestCase):
    def test_manifest(self):
        bank=ROOT/'data/train128';x=json.loads((bank/'protocol.json').read_text())
        rows=x['entries'];self.assertEqual(len(rows),128)
        self.assertEqual(len({r['instance_id'] for r in rows}),128)
        self.assertEqual(sum(r['source']=='retained26' for r in rows),26)
        self.assertEqual(sum(r['source']=='old200_train_snapshot' for r in rows),102)
        self.assertEqual(len({r['problem_fingerprint'] for r in rows}),128)
        for r in rows:self.assertEqual(hashlib.sha256((bank/r['file']).read_bytes()).hexdigest(),r['sha256'])
    def test_clock(self):
        ids=list(map(str,range(128)));c=dict(episode=0,updates=0,budgets={})
        for episode in range(3):
            self.assertTrue(begin_episode(c,ids,128,200));self.assertEqual(c['episode'],episode)
            self.assertEqual(set(c['budgets']),set(ids))
            for _ in range(10):
                for iid in ids:c['budgets'][iid]+=20
            self.assertEqual(set(c['budgets'].values()),{200})
    def test_summary_logging(self):
        with tempfile.TemporaryDirectory() as d:
            run_logging.configure(Path(d));console=io.StringIO()
            with contextlib.redirect_stdout(console):
                run_logging.log('[collect] suppressed');run_logging.log('[summary] visible')
                run_logging.log('[parity] OOM: important')
            self.assertNotIn('suppressed',console.getvalue())
            self.assertIn('important',console.getvalue());self.assertIn('visible',console.getvalue())
            self.assertIn('suppressed',(Path(d)/'detail.log').read_text())
            run_logging._file.close();run_logging._file=None

if __name__=='__main__':unittest.main()
