# Selected research implementation. See PROVENANCE.md and IMPLEMENTATION_NOTES.md.
import math
from typing import Any, Optional, Union
import torch
from torch import nn
import transformers
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class WXTrainer(transformers.Trainer):

    def set_hyper(self, args):
        self.args_ = args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.args_.model_code in ['tigerMulti', 'TIGERMTPV1', 'tiger']:
            if (self.label_smoother is not None or self.compute_loss_func is not None) and 'labels' in inputs:
                labels = inputs.pop('labels')
            else:
                labels = None
            if self.model_accepts_loss_kwargs:
                loss_kwargs = {}
                if num_items_in_batch is not None:
                    loss_kwargs['num_items_in_batch'] = num_items_in_batch
                inputs = {**inputs, **loss_kwargs}
            model.set_epoch(self.state.epoch)
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if self.compute_loss_func is not None:
                    loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and 'loss' not in outputs:
                    raise ValueError(f"The model did not return a loss from the inputs, only the following keys: {','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                loss_log = {}
                training_flag = False
                if self.args_.noise_filter_eta > 0 and 'real_loss' in outputs:
                    loss_log['real_loss'] = outputs['real_loss'].item()
                    training_flag = True
                origin_loss = loss.item()
                loss_log['origin_loss'] = origin_loss
                if 'multi_token_num' in self.args_.__dict__ and self.args_.multi_token_num > 0:
                    for i in range(getattr(self.args_, 'multi_token_num', 0)):
                        if f'multi_token_loss_{i}' in outputs and (not math.isnan(outputs[f'multi_token_loss_{i}'].item())):
                            loss += self.args_.multi_token_loss_lambda[i] * outputs[f'multi_token_loss_{i}']
                            loss_log[f'multi_token_loss_{i}'] = outputs[f'multi_token_loss_{i}'].item()
                            training_flag = True
                        if f'multi_token_real_loss_{i}' in outputs and (not math.isnan(outputs[f'multi_token_real_loss_{i}'].item())):
                            loss_log[f'multi_token_real_loss_{i}'] = outputs[f'multi_token_real_loss_{i}'].item()
                            training_flag = True
                if training_flag:
                    self.log({f'train_{k}': v for k, v in loss_log.items()})
            if self.args.average_tokens_across_devices and (self.model_accepts_loss_kwargs or self.compute_loss_func) and (num_items_in_batch is not None):
                loss *= self.accelerator.num_processes
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[list[str]]=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args_.model_code in ['tigerMulti', 'TIGERMTPV1']:
            tmp_label_names = self.label_names
            self.label_names = ['labels']
            res = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            self.label_names = tmp_label_names
            return res
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
