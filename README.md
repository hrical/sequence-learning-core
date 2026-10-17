# Sequence Learning Core

Selected core modules from an existing sequential recommendation research project. This repository contains the T5-based model, multi-step supervision, sample selection, data collation, and generation utilities. It is a limited experimental source snapshot, not a complete implementation of the final manuscript or a package for reproducing the reported results.

## Contents

| File | Purpose |
| --- | --- |
| `sddrec_core/model.py` | Original `TIGER` and active `TIGERMTPV1` model classes, including shared T5 components, auxiliary decoders, and filtering |
| `sddrec_core/data.py` | Semantic-ID sequences, leave-one-out splits, future targets, and collators |
| `sddrec_core/trainer.py` | Original loss aggregation and prediction-step customization |
| `sddrec_core/generation.py` | Candidate trie, prefix constraints, Recall and NDCG utilities |
| `configs/core_example.json` | Portable illustrative settings, without machine-specific paths |
| `examples/smoke_test.py` | Small synthetic forward, backward, and inference check; no dataset or pretrained checkpoint |
| `IMPLEMENTATION_NOTES.md` | Existing implementation behavior and differences from the manuscript |
| `PROVENANCE.md` | Extraction boundaries and attribution |

Raw data, item metadata, checkpoints, cached reference scores, experiment records, notebooks, manuscripts, account settings, and unrelated baselines are excluded.

## Environment

Python 3.10 or newer is required. The selected source targets Transformers 4.52.3. Training and model imports require PyTorch; install the appropriate PyTorch build for your own hardware.

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -e .
python examples/smoke_test.py
python -m unittest discover -s tests -v
```

The dependency-light unit tests cover generation and ranking utilities. The synthetic model check requires the model dependencies and does not validate recommendation quality. See `VALIDATION.md` for checks actually completed for this snapshot.

## Model interface

The original class names and argument spelling are retained so the selected code can be traced to the research source. `TIGERMTPV1` is the active multi-step variant. `TIGERMultiToken` only supplies its initialization and forward method and is not a standalone model in this snapshot.

The model receives a `T5Config` containing `other_args_multi_token_num` and `other_args_mtp_num_decoder_layers`. Call `model.set_hyper(args, tokenizer)` and `model.set_epoch(epoch)` before computing training losses. `WXTrainer` requires `trainer.set_hyper(args)`.

`multi_token_num` counts **additional** prediction horizons. A value of 2 means one main next-item task plus two auxiliary tasks, for three horizons in total. Auxiliary losses are returned separately and are combined by `WXTrainer.compute_loss`; simply backpropagating `outputs.loss` omits auxiliary supervision.

The example configuration defaults to no filtering because no reference cache is distributed. Its 25% reference-score retention value is illustrative, not a claim that this snapshot reproduces the manuscript setting. Enabling reference filtering requires a compatible trusted cache and the key format described in `IMPLEMENTATION_NOTES.md`.

## External data interface

Datasets are supplied separately under `data/<dataset>/`:

- `<dataset>.inter.json`: user IDs mapped to chronological item-ID lists.
- `<dataset>.index.json`: item IDs mapped to lists of semantic-ID strings, such as `<a_1>`, `<b_2>`, `<c_3>`, and `<d_4>`.

Training uses all but the last two interactions. The penultimate interaction is the validation target, and the last interaction is the test target. Semantic tokens must be added to the tokenizer and the model embeddings resized consistently. This snapshot does not include the item-text embedding or RQ-VAE training pipeline.

## Inference

Set the model to evaluation mode. Auxiliary branches are skipped and the main decoder produces the next item. The trie and `prefix_allowed_tokens_fn` helper can constrain beam search to known semantic IDs. Reported ranking metrics depend on providing valid, consistently tokenized candidates.

## Research status and attribution

No acceptance or publication status is claimed. A paper citation can be added when an authoritative public reference is available. Consult `PROVENANCE.md` before redistribution; no new license is assigned to the authors' research code by this snapshot.
