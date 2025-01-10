import torch
import torch.nn as nn

class E_Prompt(nn.Module):
    def __init__(self, num_heads, input_size, prompt_key_init, layers, prompt_init, length, prompt_prefix_size, prefix):
        super(E_Prompt, self).__init__()
        self.num_heads = num_heads
        self.input_size = input_size
        self.prompt_init = prompt_init
        self.size = length
        self.prompt_prefix_size = prompt_prefix_size
        self.prefix_tune = prefix
        
        self.prompt_key_init = prompt_key_init  
        self.layers = layers  # keep for consistency

        self.task_storage = nn.ParameterDict()  # concept_id -> prompt
        self.concept_mapping = {}               # str(customer_type) -> str(concept_id)
        self.next_concept_id = 1

    def init_e_prompt(self, concept_id):
        print(f"Creating new E-Prompt for concept {concept_id}")
        dup = 2 if self.prefix_tune else 3
        prompt_shape = (dup, 1, self.num_heads, self.size, self.input_size // self.num_heads)
        assert all(dim > 0 for dim in prompt_shape), f"Invalid prompt shape: {prompt_shape}"

        if self.prompt_init == 'zero':
            prompt = nn.Parameter(torch.zeros(prompt_shape))
        else:  # 'uniform'
            prompt = nn.Parameter(torch.randn(prompt_shape))
            nn.init.uniform_(prompt, -1, 1)
        
        self.task_storage[str(concept_id)] = prompt

    def get_or_create_concept_id(self, ctype_scalar):
        """
        ctype_scalar: a single integer/float representing the customer_type
        """
        ctype_int = int(ctype_scalar)
        ctype_str = str(ctype_int)
        if ctype_str not in self.concept_mapping:
            concept_id = str(self.next_concept_id)
            print(f"Creating new E-Prompt for customer type {ctype_str} (concept {concept_id})")
            self.concept_mapping[ctype_str] = concept_id
            self.init_e_prompt(concept_id)
            self.next_concept_id += 1
        return self.concept_mapping[ctype_str]

    def get_e_prompt(self, customer_type_batch):
        """
        customer_type_batch: shape [batch_size] (tensor)
        Return shape => [batch_size, dup, 1, num_heads, prompt_length, head_dim]
        
        Each sample i in the batch gets its own concept's 5D prompt.
        """
        if not isinstance(customer_type_batch, torch.Tensor):
            raise TypeError("customer_type_batch must be a PyTorch tensor (shape [batch_size]).")

        device = customer_type_batch.device
        batch_size = customer_type_batch.size(0)
        
        # If we haven't created any concept yet, forcibly create one for the first sample
        if len(self.task_storage) == 0:
            first_ctype = int(customer_type_batch[0].item())
            self.get_or_create_concept_id(first_ctype)

        # Grab a reference prompt for shape
        first_key = next(iter(self.task_storage.keys()))
        ref_prompt = self.task_storage[first_key]  # shape => [dup,1,num_heads,prompt_len,head_dim]

        # Build a [B, dup,1,num_heads,prompt_len,head_dim] output
        batch_prompts_shape = (batch_size,) + ref_prompt.shape
        batch_prompts = torch.zeros(batch_prompts_shape, device=device, dtype=ref_prompt.dtype)

        # Fill each sample
        for i in range(batch_size):
            ctype_scalar = customer_type_batch[i].item()
            concept_id = self.get_or_create_concept_id(ctype_scalar)
            batch_prompts[i] = self.task_storage[concept_id]

        return batch_prompts

    def save_prompts(self, save_path):
        # Convert ParameterDict to regular state dict before saving
        task_storage_state = {}
        for key, param in self.task_storage.items():
            task_storage_state[key] = param.data  # Save tensor data

        torch.save({
            'task_storage_state': task_storage_state,
            'concept_mapping': self.concept_mapping,
            'next_concept_id': self.next_concept_id
        }, f"{save_path}/eprompts.pt")

    def load_prompts(self, save_path):
        checkpoint = torch.load(f"{save_path}/eprompts.pt")
        
        # Reconstruct task_storage from saved state
        self.task_storage = nn.ParameterDict()
        for key, tensor in checkpoint['task_storage_state'].items():
            self.task_storage[key] = nn.Parameter(tensor)
        
        self.concept_mapping = checkpoint['concept_mapping']
        self.next_concept_id = checkpoint['next_concept_id']
