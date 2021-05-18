__all__ = ['ValueHead', 'GPT2HeadWithValueModel', 'respond_to_batch']

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class GPT2HeadWithValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)

        return outputs

def add_blanks(input_ids, attention_mask, n, bos_token):
    zeros = torch.zeros((input_ids.shape[0], n))
    attention_mask = torch.cat([attention_mask, zeros], dim=-1)
    zeros[:,:] = bos_token
    input_ids = torch.cat([input_ids, zeros], dim=-1)
    return input_ids, attention_mask

def respond_to_batch(model, queries, attention_mask=None, txt_len=20, top_k=0, top_p=1.0, bos_token = -1):
    """Sample text from language model."""
    input_ids = queries
    batch_size, start_len = queries.shape
    if attention_mask is not None:
        ones = torch.ones((attention_mask.shape[0], 1),
                          dtype=attention_mask.dtype,
                          device=attention_mask.device)
        batch_eos = torch.zeros((attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device)
        response_mask = list()

    for i in range(txt_len):
        # Get Logits
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        if attention_mask is not None:
            batch_eos = torch.where(batch_eos == bos_token, ones, batch_eos)
            attention_mask = torch.cat([attention_mask, 1-batch_eos], dim=-1)
            if torch.all(batch_eos == 1):
                input_ids, attention_mask = add_blanks(input_ids, attention_mask, txt_len-i-1, bos_token)
                break

    if attention_mask is not None:
        return input_ids[:, start_len:], attention_mask

    return input_ids[:, start_len:]