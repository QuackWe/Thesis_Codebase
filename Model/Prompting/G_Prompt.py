# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:55:50 2023

@author: Tamara Verbeek
"""
import torch
import torch.nn as nn
import copy


class G_Prompt(nn.Module):
    def __init__(self, num_heads, input_size, layers, prompt_init, length, prefix):
        super(G_Prompt, self).__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.layers = layers
        self.prompt_init = prompt_init
        self.storage = nn.ParameterDict()
        self.prompt = None
        self.size = length
        self.prefix_tune = prefix

    def init_g_prompt(self):
        print('Initializing the G-Prompt')
        if self.prefix_tune:
            dup = 2
        else:
            dup = 3
        prompt_pool_shape = (dup, 1, self.num_heads, self.size, self.input_size // self.num_heads)
        assert all(dim > 0 for dim in prompt_pool_shape), f"Invalid prompt shape: {prompt_pool_shape}"
        if self.prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif self.prompt_init == 'uniform':
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)
        self.storage['0'] = self.prompt

    def get_g_prompt(self, batch_size: int):
        """
        Return a G-prompt repeated for each sample in the batch.
        Shape => [B, dup, 1, num_heads, prompt_length, head_dim]
        """
        # 1. Fetch the stored prompt (e.g., shape [dup, 1, num_heads, length, head_dim])
        prompt = self.storage['0']

        # 2. Insert a batch dimension and 'expand' so that we get one prompt per sample
        #    prompt.unsqueeze(0) => [1, dup, 1, num_heads, length, head_dim]
        #    .expand(batch_size, -1, -1, -1, -1, -1) => [B, dup, 1, num_heads, length, head_dim]
        prompt_batched = prompt.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)

        return prompt_batched, prompt_batched