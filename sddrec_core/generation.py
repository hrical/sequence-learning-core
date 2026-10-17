# Selected research implementation. See PROVENANCE.md and IMPLEMENTATION_NOTES.md.
import math
from typing import Dict, List


class Trie(object):

    def __init__(self, sequences: List[List[int]]=[]):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum((1 for _ in trie))
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence: List[int], trie_dict: Dict, append_trie=None, bos_token_id: int=None):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(prefix_sequence[1:], trie_dict[prefix_sequence[0]], append_trie, bos_token_id)
        elif append_trie:
            return append_trie.get(prefix_sequence)
        else:
            return []

    def __iter__(self):

        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(prefix_sequence + [next_token], trie_dict[next_token])
            else:
                yield prefix_sequence
        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

def prefix_allowed_tokens_fn(candidate_trie):

    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out
    return prefix_allowed_tokens

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(' ', '').split('Response:')[-1] for _ in predictions]
    targets = [_.strip().replace(' ', '') for _ in targets]
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
    for b in range(B):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)
    return results

def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg / len(topk_results)

def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit / len(topk_results)

def get_metrics_results(topk_results, metrics):
    res = {}
    res['samples'] = len(topk_results)
    for m in metrics:
        if m.lower().startswith('recall'):
            k = int(m.split('@')[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith('ndcg'):
            k = int(m.split('@')[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError
    return res
