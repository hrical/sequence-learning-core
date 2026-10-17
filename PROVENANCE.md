# Source provenance

The selected definitions originate from the supplied `hf_transformer_recsys/acdemic_2025/acdemic_2025` project. `SOURCE_MANIFEST.json` records the original filenames, line locations, SHA-256 hashes, and selected symbols. Source paths in the manifest are relative and contain no machine-specific user directories.

## Extraction boundaries

- `baseline_model.py`: `TIGER`, `TIGERMTPV1`, and the `__init__` and `forward` methods of `TIGERMultiToken`.
- `data.py`: `SemanticsTokenDataset`, `MultiTokenDataset`, `SemanticsTokenCollator`, and `MultiTokenCollator`.
- `trainer_wx.py`: `WXTrainer.set_hyper`, `compute_loss`, and `prediction_step`.
- `utils.py`: the candidate trie and selected generation and metric helpers.

The extraction normalizes formatting through Python's abstract syntax tree and replaces broad project imports with explicit imports. Selected method bodies retain their original executable structure. The inactive older multi-token loss implementation and the copied evaluation loop are excluded. Baseline models, prompts for unrelated tasks, account initialization, experiment orchestration, and notebooks are excluded.

Documentation, illustrative configuration, packaging metadata, and small verification examples were added for this release. They do not introduce a replacement training algorithm.

## Third-party components

The model and trainer extend Hugging Face Transformers. Portions of their forward and trainer methods follow Transformers implementations. Transformers is distributed under Apache License 2.0. The upstream license is included in `licenses/Apache-2.0.txt`, and attribution is retained in `NOTICE`. See the upstream version at https://github.com/huggingface/transformers/tree/v4.52.3. PyTorch is an external dependency rather than vendored source.

No project-wide license for the authors' research contributions was found in the selected source project. This preparation does not grant a new license or claim ownership of third-party material. Select an appropriate research-code license and preserve applicable upstream notices before offering a broader public release.
