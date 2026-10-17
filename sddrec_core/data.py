# Selected research implementation. See PROVENANCE.md and IMPLEMENTATION_NOTES.md.
import json
import os
import torch
from torch.utils.data import Dataset


class SemanticsTokenDataset(Dataset):

    def __init__(self, args, mode='train', **kwargs):
        self.mode = mode
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.all_items = None
        self.max_his_len = args.max_his_len
        self.new_tokens = None
        self._load_data()
        self._remap_items()
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + '.inter.json'), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + '.index.json'), 'r') as f:
            self.indices = json.load(f)

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items
        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add(''.join(index))
        return self.all_items

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def _remap_items(self):
        """
        "0":
        ['<a_183><b_70><c_232><d_6>', # 代表一个item
        '<a_183><b_122><c_96><d_235>',
        '<a_52><b_170><c_54><d_27>',
        '<a_110><b_228><c_51><d_223>',
        '<a_9><b_135><c_195><d_111>']
        """
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = [''.join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data['item'] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data['inters'] = ''.join(history)
                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data['item'] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data['inters'] = ''.join(history)
            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data['item'] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data['inters'] = ''.join(history)
            inter_data.append(one_data)
        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d['inters'], labels=d['item'])

class MultiTokenDataset(SemanticsTokenDataset):

    def __init__(self, args, mode='train', **kwargs):
        super().__init__(args, mode=mode, **kwargs)

    def _process_train_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data['item'] = items[i]
                one_data['multi_token_label'] = []
                for j in range(self.args.multi_token_num):
                    if i + j + 1 < len(items):
                        one_data['multi_token_label'].append(items[i + j + 1])
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data['inters'] = ''.join(history)
                inter_data.append(one_data)
        return inter_data

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d['inters'], labels=d['item'], multi_token_labels=d.get('multi_token_label', []))

class SemanticsTokenCollator(object):

    def __init__(self, args, tokenizer, **kwargs):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d['input_ids'] for d in batch]
        label_texts = [d['labels'] for d in batch]
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding='longest', max_length=self.tokenizer.model_max_length, truncation=True, return_attention_mask=True)
        labels = self.tokenizer(label_texts, return_tensors='pt', padding='longest', max_length=self.tokenizer.model_max_length, truncation=True, return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        "\n        {\n                'input_ids': tensor([[101, 202, 303, 404,   0,   0],\n                                    [105, 206, 307, 408, 109,   0]]), # (B,L)\n                'attention_mask': tensor([[1, 1, 1, 1, 0, 0],\n                                        [1, 1, 1, 1, 1, 0]]), # (B,L)\n                'labels':\n\n            }\n        "
        return inputs

class MultiTokenCollator(SemanticsTokenCollator):

    def __call__(self, batch):
        inputs = super().__call__(batch)
        for i in range(self.args.multi_token_num):
            multi_token_labels = [d['multi_token_labels'][i] if i < len(d['multi_token_labels']) else '' for d in batch]
            label_tensor = self.tokenizer(multi_token_labels, return_tensors='pt', padding='max_length', max_length=self.args.sem_token_len + 1, truncation=True)['input_ids']
            inputs[f'multi_token_label_{i}_labels'] = label_tensor
            label_tensor[label_tensor[:, 0] == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
            inputs[f'multi_token_label_{i}_inputs'] = torch.zeros_like(label_tensor)
            inputs[f'multi_token_label_{i}_inputs'][:] = self.tokenizer.pad_token_id
            inputs[f'multi_token_label_{i}_inputs'][:, 1:] = label_tensor[:, :-1]
            inputs[f'multi_token_label_{i}_labels'][inputs[f'multi_token_label_{i}_labels'] == self.tokenizer.pad_token_id] = -100
        return inputs
