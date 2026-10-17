import math
import unittest

from sddrec_core.generation import (
    Trie, get_metrics_results, get_topk_results, prefix_allowed_tokens_fn,
)


class GenerationTests(unittest.TestCase):
    def test_trie_restricts_candidates(self):
        trie = Trie([[0, 2, 3, 1], [0, 2, 4, 1], [0, 5, 1]])
        self.assertEqual(set(trie.get([0])), {2, 5})
        self.assertEqual(set(trie.get([0, 2])), {3, 4})
        self.assertEqual(trie.get([0, 9]), [])
        self.assertEqual(trie.get([0, 2, 3, 1]), [])

    def test_prefix_callback(self):
        class TokenSequence:
            def tolist(self):
                return [0, 2]
        callback = prefix_allowed_tokens_fn(Trie([[0, 2, 3, 1]]))
        self.assertEqual(callback(0, TokenSequence()), [3])

    def test_score_sorting_and_single_target_metrics(self):
        predictions = ['<a_1>', '<a_2>', '<a_3>', '<a_4>', '<a_5>', '<a_6>']
        scores = [0.2, 0.9, 0.1, 0.8, 0.2, 0.1]
        results = get_topk_results(predictions, scores, ['<a_1>', '<a_4>'], 3)
        self.assertEqual(results, [[0, 1, 0], [1, 0, 0]])
        metrics = get_metrics_results(results, ['recall@1', 'recall@3', 'ndcg@3'])
        self.assertEqual(metrics['recall@1'], 0.5)
        self.assertEqual(metrics['recall@3'], 1.0)
        self.assertAlmostEqual(metrics['ndcg@3'], (1 / math.log2(3) + 1) / 2)

    def test_invalid_candidate_receives_low_score(self):
        results = get_topk_results(['invalid', 'valid'], [1.0, 0.1], ['valid'],
                                   2, all_items={'valid'})
        self.assertEqual(results, [[1, 0]])


if __name__ == '__main__':
    unittest.main()
