# Selected research implementation. See PROVENANCE.md and IMPLEMENTATION_NOTES.md.
import copy
import pickle
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
from transformers.modeling_outputs import BaseModelOutput, ModelOutput


class TIGER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.temperature = 1.0

    def set_hyper(self, args, tokenizer):
        self.temperature = args.temperature
        self.args = args
        self.tokenizer = tokenizer

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def ranking_loss(self, lm_logits, labels, input_ids=None):
        if labels is not None:
            t_logits = lm_logits / self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
            if getattr(self.args, 'seqlabel_2_score_dict_path', None) is not None and input_ids is not None:
                with open(self.args.seqlabel_2_score_dict_path, 'rb') as f:
                    seqlabel_2_score_dict = pickle.load(f)
                item_pro_list = []
                for input_id, label in zip(input_ids, labels):
                    input_id_key = '-'.join([str(id.item()) for id in input_id if id != 0])
                    label_key = '-'.join([str(id.item()) for id in label if id != 0])
                    final_key = (input_id_key, label_key)
                    _, item_pro = seqlabel_2_score_dict[final_key]
                    item_pro_list.append(item_pro)
                label_nonzero_indices = labels[:, 0] != -100
                labels = labels[label_nonzero_indices]
                t_logits = t_logits[label_nonzero_indices]
                if labels.size(0) == 0:
                    return loss
                with torch.no_grad():
                    label_pro = torch.gather(F.softmax(t_logits, dim=-1), 2, labels.unsqueeze(-1)).squeeze(-1).prod(dim=-1)
                    item_pro_list = torch.tensor(item_pro_list, device=label_pro.device)
                    excess_pro = item_pro_list[label_nonzero_indices] - label_pro
                    sorted_excess_pro, indices = torch.sort(excess_pro, descending=True)
                    save_num = int(self.args.rho_save_ration * labels.size(0))
                    selected_indices = indices[:save_num]
                s_logits = t_logits[selected_indices]
                s_labels = labels[selected_indices]
                s_loss = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))
                return (loss, s_loss)
            elif self.args.noise_filter_eta > 0 and self.training and (self.cur_epoch >= getattr(self.args, 'noise_filter_start_epoch', 0)):
                label_nonzero_indices = labels[:, 0] != -100
                labels = labels[label_nonzero_indices]
                t_logits = t_logits[label_nonzero_indices]
                if labels.size(0) == 0:
                    return loss
                with torch.no_grad():
                    label_logit = torch.log(torch.gather(F.softmax(t_logits, dim=-1), 2, labels.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
                    sorted_label_logit, indices = torch.sort(label_logit, descending=False)
                    filter_num = int(self.args.noise_filter_eta * labels.size(0))
                    selected_indices = indices[filter_num:]
                s_logits = t_logits[selected_indices]
                s_labels = labels[selected_indices]
                s_loss = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))
                return (loss, s_loss)
        return loss

    def total_loss(self, lm_logits, labels, decoder_input_ids, input_ids=None):
        loss = self.ranking_loss(lm_logits, labels, input_ids)
        return loss

    def forward(self, input_ids=None, whole_word_ids=None, attention_mask=None, encoder_outputs=None, decoder_input_ids=None, decoder_attention_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, labels=None, inputs_embeds=None, decoder_inputs_embeds=None, head_mask=None, decoder_head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, reduce_loss=False, return_hidden_state=False, **kwargs):
        """
            input_ids [B,L]  attention_mask [B,L] labels [B,5] (4 id + 1 eos) decoder_input_ids== labels shift
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids, input_ids)
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return ModelOutput(**{k: v for k, v in dict(loss=loss[-1] if isinstance(loss, tuple) else loss, real_loss=loss[0] if isinstance(loss, tuple) else loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions).items() if v is not None})

class TIGERMultiToken(TIGER):

    def __init__(self, config):
        super().__init__(config)
        self.linear_mtp = nn.ModuleList([nn.Linear(self.config.d_model * 2, self.config.d_model) for i in range(config.other_args_multi_token_num)])
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.other_args_mtp_num_decoder_layers
        self.mtp_block = nn.ModuleList([T5Stack(decoder_config) for i in range(config.other_args_multi_token_num)])

    def forward(self, input_ids=None, whole_word_ids=None, attention_mask=None, encoder_outputs=None, decoder_input_ids=None, decoder_attention_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, labels=None, inputs_embeds=None, decoder_inputs_embeds=None, head_mask=None, decoder_head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, reduce_loss=False, return_hidden_state=False, return_loss=True, **kwargs):
        """
            input_ids [B,L]  attention_mask [B,L] labels [B,5] (4 id + 1 eos) decoder_input_ids== labels shift
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids, input_ids)
        multi_token_loss = self.calc_multitoken_loss(kwargs, encoder_attention_mask=attention_mask, encoder_hidden_states=hidden_states, ntp_decoder_sequence_output=sequence_output)
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return ModelOutput(**{k: v for k, v in dict(loss=loss[-1] if isinstance(loss, tuple) else loss, real_loss=loss[0] if isinstance(loss, tuple) else loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, **multi_token_loss).items() if v is not None})

class TIGERMTPV1(TIGERMultiToken):

    def __init__(self, config):
        super().__init__(config)
        del self.linear_mtp

    def calc_multitoken_loss(self, kwargs, encoder_attention_mask, encoder_hidden_states, ntp_decoder_sequence_output):
        multi_token_loss_dict = {}
        if self.training:
            for i in range(self.args.multi_token_num):
                decoder_input_ids = kwargs[f'multi_token_label_{i}_inputs']
                labels = kwargs[f'multi_token_label_{i}_labels']
                decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=None, inputs_embeds=None, past_key_values=None, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=None, cross_attn_head_mask=None, use_cache=False, output_attentions=None, output_hidden_states=None, return_dict=True)
                sequence_output = decoder_outputs[0]
                mtp_decoder_sequence_output = sequence_output
                mtp_decoder_sequence_output = self.mtp_block[i](input_ids=None, attention_mask=None, inputs_embeds=mtp_decoder_sequence_output, past_key_values=None, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=None, cross_attn_head_mask=None, use_cache=False, output_attentions=None, output_hidden_states=None, return_dict=True)[0]
                if self.config.tie_word_embeddings:
                    mtp_decoder_sequence_output = mtp_decoder_sequence_output * self.model_dim ** (-0.5)
                lm_logits = self.lm_head(mtp_decoder_sequence_output)
                noise_filter_eta_tmp = 0
                if self.args.noise_filter_eta > 0:
                    noise_filter_eta_tmp = self.args.noise_filter_eta
                    self.args.noise_filter_eta = self.args.noise_filter_eta_list[i]
                loss = self.total_loss(lm_logits, labels, decoder_input_ids)
                if noise_filter_eta_tmp > 0:
                    self.args.noise_filter_eta = noise_filter_eta_tmp
                if isinstance(loss, tuple):
                    multi_token_loss_dict[f'multi_token_real_loss_{i}'] = loss[0]
                    multi_token_loss_dict[f'multi_token_loss_{i}'] = loss[-1]
                else:
                    multi_token_loss_dict[f'multi_token_loss_{i}'] = loss
        return multi_token_loss_dict
