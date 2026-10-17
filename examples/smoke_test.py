"""Small synthetic runtime check; requires the model dependencies."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import T5Config

from sddrec_core.model import TIGERMTPV1


def main():
    torch.manual_seed(7)
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(**json.loads((root / 'configs/core_example.json').read_text()))
    config = T5Config(
        vocab_size=32, d_model=32, d_kv=8, d_ff=64, num_layers=1,
        num_decoder_layers=1, num_heads=4, dropout_rate=0.0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        use_cache=False,
        other_args_multi_token_num=args.multi_token_num,
        other_args_mtp_num_decoder_layers=args.mtp_num_decoder_layers,
    )
    model = TIGERMTPV1(config)
    model.set_hyper(args, tokenizer=None)
    model.set_epoch(0)
    model.train()
    batch = {'input_ids': torch.randint(2, 32, (4, 8)), 'attention_mask': torch.ones(4, 8)}
    labels = torch.randint(2, 32, (4, 5))
    labels[:, -1] = 1
    batch['labels'] = labels
    for i in range(args.multi_token_num):
        target = torch.randint(2, 32, (4, 5))
        target[:, -1] = 1
        batch[f'multi_token_label_{i}_labels'] = target
        batch[f'multi_token_label_{i}_inputs'] = model._shift_right(target)
    outputs = model(**batch)
    loss = outputs.loss
    for i, weight in enumerate(args.multi_token_loss_lambda):
        loss = loss + weight * outputs[f'multi_token_loss_{i}']
    assert torch.isfinite(loss), 'Non-finite training loss'
    loss.backward()
    assert model.mtp_block[0].block[0].layer[0].SelfAttention.q.weight.grad is not None
    model.eval()
    with torch.no_grad():
        result = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                max_new_tokens=5, num_beams=2)
    assert result.shape[0] == 4
    print('Synthetic forward, backward, and inference checks passed.')


if __name__ == '__main__':
    main()
