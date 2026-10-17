# Implementation notes

This snapshot preserves the selected executable definitions from the supplied experiment source. The following behavior should be accounted for before describing it as a final-paper implementation.

## Reference criterion

The reference-filter branch computes `reference_item_probability - target_item_probability` and retains the largest values. Each item probability is a product of its token probabilities, including the supplied end token. The paper's excess-loss expression instead corresponds to `log(reference_item_probability / target_item_probability)`. These criteria are not generally rank-equivalent.

Reference scores are read from a trusted pickle file identified by `seqlabel_2_score_dict_path`. Keys are `(history_key, label_key)`, where each key is a hyphen-joined sequence of token IDs excluding zero IDs. Values contain `(token_probabilities, item_probability)`. The original offline notebook that builds the cache is not distributed, and no cache is included. Do not load untrusted pickle files.

The original filtering routine reloads the cache during loss evaluation. It assumes every queried sample key exists, including any samples evaluated with reference filtering enabled. It does not supply a missing-key fallback, probability stabilization, or a minimum retained count. Callers must ensure their batch size and `rho_save_ration` retain at least one valid sample.

## Auxiliary filtering

The main loss receives `input_ids`, which enables reference-cache lookup. The active auxiliary loss call does not pass `input_ids`, so it does not enter the reference-filter branch. Auxiliary tasks can use self-loss filtering through `noise_filter_eta` and `noise_filter_eta_list`.

Self-loss filtering discards the configured fraction of lowest-probability samples, equivalent to the highest-loss samples. Reference filtering instead retains `rho_save_ration` of samples. These arguments therefore have different meanings.

## Architecture and step count

The main task uses the shared T5 decoder and language-model head. Each auxiliary task runs the shared decoder on its own shifted target sequence and then applies its additional task-specific T5 stack. The main next-item task has no additional task-specific stack in this source.

`multi_token_num` is the count of auxiliary tasks. Total horizons equal `1 + multi_token_num`. Evaluation skips auxiliary branches and uses the main task. Training invokes the shared decoder separately for the auxiliary horizons; it does not pack all horizons into a single decoder call with an explicit block attention mask.

## Initialization and training configuration

Constructing `TIGERMTPV1(T5Config(...))` initializes model parameters rather than loading pretrained weights. Pretrained initialization requires a separately verified loading procedure. The original T5 factory also instantiated from configuration; the LoRA setup in the supplied utility module applied to the LCREC branch, not the T5 branch. This repository does not imply that T5-Large pretrained initialization or T5 LoRA was already implemented.

The example settings are deliberately portable and illustrative. They are not a recovered final experiment configuration. No full two-stage launcher, original hyperparameter search, training logs, or recommendation results are included.

## Existing edge cases

The selected source assumes semantic-ID labels have the expected lengths. Callers should validate padding, ignored labels, reference-cache coverage, and all-empty auxiliary batches. The trainer skips NaN auxiliary losses but does not provide a general empty-batch policy. The trie and metric helpers assume unique candidates and one ground-truth item per user.
