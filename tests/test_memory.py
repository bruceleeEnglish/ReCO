import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from savemv.data.memory import load_locomo, load_longmemeval
from savemv.experiments.memory import calibrated_stops
from savemv.experiments.qa import task_info, token_f1
from savemv.reco.hazard import HazardSettings

LONGMEMEVAL = [
    {"question_id": "q1", "question_type": "single-session-user", "question": "Where do I work?",
     "answer": "Acme", "haystack_session_ids": ["s1", "s2"], "answer_session_ids": ["s2"],
     "haystack_sessions": [
         [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
         [{"role": "user", "content": "I work at Acme", "has_answer": True},
          {"role": "assistant", "content": "nice"}, {"role": "user", "content": "thanks"}]]},
    {"question_id": "q2", "question_type": "multi-session", "question": "Hi?", "answer": "x",
     "haystack_session_ids": ["s1"], "answer_session_ids": ["s1"],
     "haystack_sessions": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]},
    {"question_id": "q3_abs", "question_type": "temporal-reasoning", "question": "?", "answer": "",
     "haystack_session_ids": ["s1"], "answer_session_ids": [], "haystack_sessions": [[]]},
]

LOCOMO = [{"sample_id": "conv-1", "conversation": {
    "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "d1", "session_2_date_time": "d2",
    "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "later"}],
    "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"},
                  {"speaker": "B", "dia_id": "D1:2", "text": "photo", "blip_caption": "a dog"}]},
    "qa": [{"question": "dog?", "answer": "yes", "evidence": ["D1:2"], "category": 4},
           {"question": "both?", "answer": "yes", "evidence": ["D1:1; D2:1"], "category": 1},
           {"question": "none?", "answer": "no", "evidence": [], "category": 5}]}]


class MemoryLoaderTest(unittest.TestCase):
    def _write(self, obj) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(obj, f)
        f.close()
        return Path(f.name)

    def test_longmemeval_rounds_dedup_and_evidence(self) -> None:
        bench = load_longmemeval(self._write(LONGMEMEVAL))
        self.assertEqual([t.qid for t in bench.tasks], ["q1", "q2"])             # abstention skipped
        self.assertEqual(len(bench.texts), 3)                                    # s1 round shared by q1 and q2
        q1, q2 = bench.tasks
        self.assertEqual(len(q1.units), 3)
        self.assertEqual([bench.texts[i] for i in q1.relevant], ["user: I work at Acme\nassistant: nice"])
        self.assertEqual(q1.relevant_sessions, frozenset({"s2"}))
        self.assertEqual(list(q2.units), [0])
        self.assertEqual(len(q2.relevant), 0)                                    # no has_answer flag

    def test_locomo_turns_and_evidence(self) -> None:
        bench = load_locomo(self._write(LOCOMO))
        self.assertEqual(len(bench.texts), 3)
        self.assertEqual(bench.texts[1], "B: photo [shares a dog]")
        self.assertEqual(len(bench.tasks), 2)                                    # question without evidence skipped
        self.assertEqual([bench.texts[i] for i in bench.tasks[1].relevant], ["A: hello", "B: later"])
        self.assertEqual(bench.tasks[0].category, "4")

    def test_qa_task_info_and_f1(self) -> None:
        info = task_info("locomo", str(self._write(LOCOMO)))
        self.assertEqual(sorted(info), ["conv-1-0", "conv-1-1"])                 # adversarial (5) skipped
        self.assertEqual(info["conv-1-0"]["dates"]["0:session_1"], "d1")
        info = task_info("longmemeval", str(self._write(LONGMEMEVAL)))
        self.assertEqual(info["q1"]["answer"], "Acme")
        self.assertEqual(info["q1"]["dates"], {"s1": None, "s2": None})               # no dates in this fixture
        self.assertAlmostEqual(token_f1("The cat sat.", "cat sat"), 1.0)
        self.assertAlmostEqual(token_f1("dog", "cat"), 0.0)


class MemoryStopTest(unittest.TestCase):
    def test_calibrated_stops_keep_recall(self) -> None:
        """Per-user corpora of different sizes: both rules reach the recall target on held-out users."""
        rng = np.random.default_rng(0)
        est_rows, exact_rows = [], []
        for _ in range(150):
            n = int(rng.integers(60, 240))
            x = rng.standard_normal(n) * rng.uniform(0.5, 2.0)
            est_rows.append((x + rng.uniform(0.2, 1.0) * rng.standard_normal(n)).astype(np.float32))
            exact_rows.append(x.astype(np.float32))
        stops = calibrated_stops(est_rows, exact_rows, 0.1, HazardSettings(horizon=16, ahead=16))
        sizes = np.array([len(e) for e in est_rows])
        for rule, st in stops.items():
            self.assertTrue(np.all(st <= sizes), rule)
            recall = np.mean([np.mean(np.isin(np.argsort(-x, kind="stable")[:10], np.argsort(-e, kind="stable")[:s]))
                              for e, x, s in zip(est_rows, exact_rows, st)])
            self.assertGreaterEqual(recall, 0.85, rule)


if __name__ == "__main__":
    unittest.main()
