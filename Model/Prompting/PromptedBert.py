from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertEncoder, BertLayer
from transformers import BertModel, BertConfig
from Prompting.EPrompt import E_Prompt
from Prompting.G_Prompt import G_Prompt
import torch
import torch.nn as nn
import math

class PromptedBertLayer(BertLayer):
    """
    Subclass of BertLayer that prepends G-Prompt and E-Prompt to the key/value
    tensors inside the self-attention mechanism.
    """
    def __init__(self, config):
        super().__init__(config)
        self.attention = PromptedBertAttention(config)

        # We do NOT define G_Prompt/E_Prompt here because we’ll get them from the parent model.

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        g_key_prefix=None,
        g_value_prefix=None,
        e_key_prefix=None,
        e_value_prefix=None,
        *args,
        **kwargs
    ):
        """
        The main difference from the standard BertLayer forward is that we
        concatenate g_key_prefix & e_key_prefix to the 'key' and similarly for value.
        """
        # Standard BertLayer forward except for hooking the self-attention step:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            # We’ll pass g/e prompts into the self-attention module via kwargs
            g_key_prefix=g_key_prefix,
            g_value_prefix=g_value_prefix,
            e_key_prefix=e_key_prefix,
            e_value_prefix=e_value_prefix,
            **kwargs
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # attention_probs, etc.

        # Pass the result to the intermediate+output sub-layers
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class PromptedBertEncoder(BertEncoder):
    """
    Subclass of BertEncoder that passes the G-Prompt and E-Prompt down into each BertLayer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([PromptedBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        g_key_prefix=None,
        g_value_prefix=None,
        e_key_prefix=None,
        e_value_prefix=None,
        *args,
        **kwargs
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                g_key_prefix=g_key_prefix,
                g_value_prefix=g_value_prefix,
                e_key_prefix=e_key_prefix,
                e_value_prefix=e_value_prefix,
                *args,
                **kwargs
            )
            hidden_states = layer_outputs[0]
        return (hidden_states,)


class PromptedBertModel(nn.Module):
    """
    A BERT backbone that injects G-Prompt and E-Prompt inside each encoder layer.
    """
    def __init__(self, config, pretrained_weights=None, enable_g_prompt=True, enable_e_prompt=True):
        super().__init__()

        # Load the BERT config and model
        bert_config = BertConfig.from_pretrained(pretrained_weights)
        self.embeddings = BertModel.from_pretrained(pretrained_weights).embeddings

        # Build a custom encoder that inherits from the standard BertEncoder
        self.encoder = PromptedBertEncoder(bert_config)

        # Tie or freeze layers if desired:
        # Freeze the first half of layers
        huggingface_bert = BertModel.from_pretrained(pretrained_weights)
        for name, param in huggingface_bert.encoder.named_parameters():
            if "layer." in name:  # Check if this parameter belongs to a layer
                layer_num = int(name.split("layer.")[1].split(".")[0])
                if layer_num < bert_config.num_hidden_layers // 2:
                    param.requires_grad = False
        
        # Copy the weights from the huggingface_bert encoder
        for i, layer in enumerate(self.encoder.layer):
            layer.load_state_dict(huggingface_bert.encoder.layer[i].state_dict())
        
        # Initialize G-Prompt if enabled
        self.enable_g_prompt = enable_g_prompt
        if enable_g_prompt:
            self.g_prompt = G_Prompt(
                num_heads=config.num_heads,
                input_size=config.hidden_dim,
                layers=['k', 'v'],
                prompt_init='uniform',
                length=config.g_prompt_length,
                prefix=config.prefix_tune
            )
            self.g_prompt.init_g_prompt()

        # Initialize E-Prompt if enabled
        self.enable_e_prompt = enable_e_prompt
        if enable_e_prompt:
            self.e_prompt = E_Prompt(
                num_heads=config.num_heads,
                input_size=config.hidden_dim,
                prompt_key_init='uniform',
                layers=['k', 'v'],
                prompt_init='uniform',
                length=config.e_prompt_length,
                prompt_prefix_size=config.prompt_prefix_size,
                prefix=config.prefix_tune
            )

        # Move entire model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.cuda.current_device())

    def forward(self, input_ids, attention_mask, customer_type=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embeddings(input_ids=input_ids)

        # G-Prompt
        g_key_prefix, g_value_prefix = None, None
        if self.enable_g_prompt:
            # Pass batch_size to G_Prompt
            g_key_prefix, g_value_prefix = self.g_prompt.get_g_prompt(batch_size)

        # E-Prompt
        e_key_prefix, e_value_prefix = None, None
        if self.enable_e_prompt and customer_type is not None:
            e_prefix = self.e_prompt.get_e_prompt(customer_type)
            e_key_prefix, e_value_prefix = e_prefix, e_prefix

        # Pass everything through custom BertEncoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            g_key_prefix=g_key_prefix,
            g_value_prefix=g_value_prefix,
            e_key_prefix=e_key_prefix,
            e_value_prefix=e_value_prefix,
        )

        last_hidden_state = encoder_outputs[0]
        return last_hidden_state



class PromptedBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_heads = config.num_attention_heads  # Get from BERT config
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        g_key_prefix=None,
        g_value_prefix=None,
        e_key_prefix=None,
        e_value_prefix=None,
        output_attentions=False,
    ):
        batch_size = hidden_states.size(0)
        device = hidden_states.device  # Get device from input tensor

        # Regular BERT attention computation
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, num_heads, seq_len, head_dim]
        key_layer = self.transpose_for_scores(mixed_key_layer)      # [batch_size, num_heads, seq_len, head_dim]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch_size, num_heads, seq_len, head_dim]

        # Handle G-Prompt
        if g_key_prefix is not None:
            g_key = g_key_prefix.to(device)
            g_value = g_value_prefix.to(device)

            # Now g_key => [B, dup, 1, num_heads, length, head_dim]
            g_key = g_key.squeeze(2)  # => [B, dup, num_heads, length, head_dim]
            g_value = g_value.squeeze(2)

            # Permute to [B, num_heads, dup, length, head_dim]
            g_key = g_key.permute(0, 2, 1, 3, 4)
            g_value = g_value.permute(0, 2, 1, 3, 4)
            
            # Flatten dup*length => new_seq_len
            B, H, dup, L, D = g_key.shape
            g_key = g_key.reshape(B, H, dup * L, D)
            g_value = g_value.reshape(B, H, dup * L, D)

            # Then concatenate
            key_layer = torch.cat([g_key, key_layer], dim=2)
            value_layer = torch.cat([g_value, value_layer], dim=2)
            

        # E-Prompt logic
        if e_key_prefix is not None:
            # e_key_prefix => shape [B, dup, 1, H, L, D]
            e_key_prefix = e_key_prefix.to(device)
            e_value_prefix = e_value_prefix.to(device)

            # Remove the extra dim=2 which is "1"
            e_key_prefix = e_key_prefix.squeeze(2)   # => [B, dup, H, L, D]
            e_value_prefix = e_value_prefix.squeeze(2)

            # Permute to [B, H, dup, L, D]
            e_key_prefix = e_key_prefix.permute(0, 2, 1, 3, 4)
            e_value_prefix = e_value_prefix.permute(0, 2, 1, 3, 4)

            # Flatten (dup * L) => new_seq_len
            B, H, dup, L, D = e_key_prefix.shape
            e_key_prefix = e_key_prefix.reshape(B, H, dup * L, D)
            e_value_prefix = e_value_prefix.reshape(B, H, dup * L, D)

            # Concatenate along the sequence dimension
            key_layer = torch.cat([e_key_prefix, key_layer], dim=2)
            value_layer = torch.cat([e_value_prefix, value_layer], dim=2)


        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch_size, num_heads, seq_len + prompt_len_g + prompt_len_e , seq_len + prompt_len_g + prompt_len_e]
        attention_scores /= math.sqrt(self.attention_head_size)

        # Suppose key_layer is now [B, heads, final_seq_len, head_dim]
        final_seq_len = key_layer.size(2)

        if attention_mask is not None:
            orig_seq_len = attention_mask.size(1)  # The original length
            added_prompt_len = final_seq_len - orig_seq_len

            if added_prompt_len < 0:
                raise ValueError("Prompt length mismatch: final_seq_len < orig_seq_len?")
            
            # create 1s for the newly prepended prompt tokens
            prompt_mask = torch.ones(
                (batch_size, added_prompt_len), 
                device=device, dtype=attention_mask.dtype
            )
            updated_mask = torch.cat([prompt_mask, attention_mask], dim=1)   # shape => [B, final_seq_len]
            
            # for BERT attn, expand to [B,1,1, final_seq_len]
            extended_attention_mask = updated_mask.unsqueeze(1).unsqueeze(2).float()
            attention_scores += extended_attention_mask


        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_heads, seq_len + prompt_len_g + prompt_len_e , head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,) if not output_attentions else (context_layer, attention_probs)
        return outputs


class PromptedBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = PromptedBertSelfAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        g_key_prefix=None,
        g_value_prefix=None,
        e_key_prefix=None,
        e_value_prefix=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            g_key_prefix=g_key_prefix,
            g_value_prefix=g_value_prefix,
            e_key_prefix=e_key_prefix,
            e_value_prefix=e_value_prefix,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
