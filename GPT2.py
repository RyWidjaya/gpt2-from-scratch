from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import inspect

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Architecture
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class GPTConfig: 
    block_size: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 12 
    n_head: int = 12 
    n_embd: int = 768 


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
 
        # n_embd must divide evenly across all heads, otherwise concatenating
        # the heads back together at the end won't reconstruct n_embd exactly
        assert config.n_embd % config.n_head == 0
 
         # key, query, value projections for all heads, but in a batch
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head  # size of each individual head's q/k/v vectors
 

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
 
        # output projection: mixes information ACROSS heads after they've been computed independently and concatenated back together. Shape is already n_embd -> n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.NANOGPT_SCALE_INIT = 1 # attaching an extra attribute

        # causal mask: a lower-triangular matrix of 1s and 0s, built ONCE at the maximum possible sequence length (block_size). 1 = "allowed to attend to this position",
        # 0 = "this is a future position, must be masked out". Stored as a buffer (not a parameter since it remains constant)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
 
    def forward(self, x):
        # x.shape = (B, T, C) = (no.examples, block_size_used, n_embd)
        B, T, C = x.shape
 
        # apply linear projections to x to get the full-size key, query and value matrices
        k = self.key(x)    # k.shape = (B, T, n_embd)
        q = self.query(x)  # q.shape = (B, T, n_embd)
        v = self.value(x)  # v.shape = (B, T, n_embd)
 
        # split the last dimension (n_embd) into (n_head, head_size), then move n_head into the batch-like dimension via transpose so that matrix multiplication treats each head completely independently.
        # .view(...): (B, T, n_embd) -> (B, T, n_head, head_size)
        # .transpose(1, 2): (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # PyTorch has a trick that treats every dim except for the last 2 as batch dimensions 

        # Line-by-line Self-Attention
        # compute raw attention scores ("affinities") between every position's query and every position's key, independently within each head.
        "att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)"
        "att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))"
        "att = F.softmax(att, dim=-1)" 
        "y = att @ v"

        "Flash Attention Shortcut"
        # Flash Attention computes the exact same softmax(QK^T/sqrt(d))V result as the manual implementation above, 
        # but restructures the computation into tiles processed via a fused GPU kernel, using a running-softmax trick so the full (T,T) attention matrix is never fully materialized in slow VRAM.
        # This eliminates the memory read/write bottleneck of the standard approach (which is the real limiting factor, not raw compute), 
        # making it significantly faster and more memory-efficient without changing the mathematical output at all.
        # The math is identical, what changes is purely the computational strategy for executing that math on the GPU - specifically how data moves between fast on-chip memory and slower VRAM during the calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
 
        # recombine all heads back into a single n_embd-sized vector per position.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
 
        # final output projection: mixes information across the (currently independent, just-concatenated) heads together into a genuinely blended representation.
        y = self.proj(y)
 
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define the layers we want to implement int he MLP portion of the block 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # gelu is just a relu that does not have a flat tail - gets rid of the dead neuron problem
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd) # this projection is what transforms the output back to the original shape (B, T, C) to allow for residual addition

        self.proj.NANOGPT_SCALE_INIT = 1 # attaching an extra attribute

    def forward(self, x):
        # implement the MLP as x flows through it
        # x passes through each individual layer, reassigns the value to x, and returns the final x 
        x = self.c_fc(x) 
        x = self.gelu(x) 
        x = self.proj(x) 
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # layernorm 1 
        # Keeps each token's value consistent and controlled (gaussian) at every layer, preventing the scale drift across depth that would cause softmax saturation and vanishing/exploding gradients
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        # Self-attention
        # Enables tokens to communicate with one another
        self.attn = CausalSelfAttention(config) # different object separate for attention itself
        # layernorm 2
        self.ln_2 = nn.LayerNorm(config.n_embd) # second layernorm 
        # MLP/FNN
        # Takes what attention gathered (cross-token context) and processes it further, per token (no more cross-token mixing), via mathematical computations
        self.mlp = MLP(config) # different object separate for MLP
    
    # in a single forward pass of the loop perform the following layers in order: layernorm -> attention -> residual add -> layer norm -> feedforward -> residual add
    def forward(self, x):
        # residual addition (accumulated into x so the final output contains all oeprations from the transformer block)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) 
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # word token embedding (embedding word tokens to vectors)
            wte = nn.Embedding(config.vocab_size, config.n_embd), # key=value is another way to setup a dict using dict() 
            # word position embedding (embedding position of word to vectors)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # stack of transformer blocks which consists of layernorm, multihead self-attention, residual add, MLP 
            # creates a list of instances of Block - 1 Block object for each loop of the block layer
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            # layernorm layer
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weights)

    # default initialisation
    # helps to guarantee that the weights and parameters upon initialisation produce reasonable losses that aren't inflated
    def __init_weights(self, module):
        if isinstance(module, nn.Linear): # if a Linear module
            std = 0.02 
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # check if it has the manual tag - we only consider attn and mlp outputs since they are accumulated onto x, other outputs are not involved involved in residual additions to x
                std *= (2 * self.config.n_layer) ** -0.5 # if it has the tag then scale down the std by some variable amount (shrink the default 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # initialises the module's weights by randomly sampling values from a normal dist. with mean=0.0 and std scaled down
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # if an embedding module
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # a
        B, T = idx.shape
        # T cannot be greater than the blocK_size, that is the maximum length of the context window that we can provide
        assert T <= self.config.block_size, f"Cannot forward sequence length of {T}, block_size is only {self.config.block_size}"

        # forward the token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # creates an embedding vector for each of the 0 to T positions in a sequence
        x = tok_emb + pos_emb # combines position and token embeddings to get a vector representetive of the meaning and position

        # forward the blocks for the transformer
        # transformer.h stores all the blocks and we are simply iterating over all the blocks and passing x through them
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # softmax applied when calculating loss use F.cross_entropy
        loss = None 
        if targets is not None: 
            # flattens logits and targets from 3D to 2D and 1D respectively to calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    # Hyper Parameter Tuning - Weight Decay + FusedAdamW
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad) before we split into params that are 2D or larger
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ] # stores a list of 2 dicts with params and weight decay keys

        # find no_ parameters being decayed and not being decayed
        num_decay_params = sum(p.numel() for p in decay_params) # get numel() for each parameter in decay_params and sum the number of elements for each param to get total no. parameters
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # checks if the installed PyTorch version's AdamW supports the 'fused' argument, to avoid crashing on older versions that don't
        use_fused = fused_available and device == "cuda"
        optimizer = torch.optim.AdamW(params=optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        # we can pass in a list of dictionaries consisting of the params and the corresponding weight_decay to params as an argument
        # torch.optim.AdamW will automatically process the required weight decays
        # on every optimizer.step() call for a decay-group parameter the update is weight = weight - lr * gradient_update - lr * weight_decay * weight
        return optimizer