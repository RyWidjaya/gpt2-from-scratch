from GPT2 import GPT, GPTConfig
import torch
import tiktoken

# this loads in the weights for GPT-2 from huggingface
# this is not used for training and does not replace training, instead we load
# these weights in to check that our model architecture works as intended and
# matches GPT-2 exactly
def from_pretrained(model_type):
    """Loads pretrained GPT-2 model weights from huggingface into our GPT class"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

    # NOTE: our CausalSelfAttention uses 3 separate Linear layers (key, query, value, proj)
    # instead of HF's fused c_attn/c_proj, AND our MLP now uses "proj" instead of "c_proj".
    # None of these can be matched 1:1 by name against HF's keys, so we exclude all of them
    # here and handle every one manually further down.
    attn_keys = [k for k in sd_keys if '.attn.key.' in k or '.attn.query.' in k
                    or '.attn.value.' in k or '.attn.proj.' in k]
    mlp_proj_keys = [k for k in sd_keys if '.mlp.proj.' in k]
    sd_keys = [k for k in sd_keys if k not in attn_keys and k not in mlp_proj_keys]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

    # same reasoning: c_attn/c_proj (attn) and mlp.c_proj on the HF side have no 1:1
    # counterpart in our naming, so exclude them here too and handle manually below
    attn_keys_hf = [k for k in sd_keys_hf if '.attn.c_attn.' in k or '.attn.c_proj.' in k]
    mlp_proj_keys_hf = [k for k in sd_keys_hf if '.mlp.c_proj.' in k]
    sd_keys_hf = [k for k in sd_keys_hf if k not in attn_keys_hf and k not in mlp_proj_keys_hf]

    transposed = ['mlp.c_fc.weight']  # mlp.c_proj removed -- now handled manually below
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    # --- manual attention weight loading ---
    # HF stores q, k, v fused as one Conv1D weight (n_embd, 3*n_embd) per block.
    # Our model keeps them as 3 separate Linear layers (n_embd, n_embd) each, so we split
    # HF's fused tensor into 3 equal chunks and route each into its matching layer,
    # transposing since Conv1D is (in, out) and Linear expects (out, in).
    for i in range(config.n_layer):
        w_hf = sd_hf[f'transformer.h.{i}.attn.c_attn.weight']
        b_hf = sd_hf[f'transformer.h.{i}.attn.c_attn.bias']

        w_q, w_k, w_v = w_hf.split(config.n_embd, dim=1)
        b_q, b_k, b_v = b_hf.split(config.n_embd, dim=0)

        with torch.no_grad():
            sd[f'transformer.h.{i}.attn.query.weight'].copy_(w_q.t())
            sd[f'transformer.h.{i}.attn.key.weight'].copy_(w_k.t())
            sd[f'transformer.h.{i}.attn.value.weight'].copy_(w_v.t())
            sd[f'transformer.h.{i}.attn.query.bias'].copy_(b_q)
            sd[f'transformer.h.{i}.attn.key.bias'].copy_(b_k)
            sd[f'transformer.h.{i}.attn.value.bias'].copy_(b_v)

        # attn.c_proj -> our attn.proj (1:1 shape, just needs the Conv1D transpose)
        w_proj_hf = sd_hf[f'transformer.h.{i}.attn.c_proj.weight']
        b_proj_hf = sd_hf[f'transformer.h.{i}.attn.c_proj.bias']
        with torch.no_grad():
            sd[f'transformer.h.{i}.attn.proj.weight'].copy_(w_proj_hf.t())
            sd[f'transformer.h.{i}.attn.proj.bias'].copy_(b_proj_hf)

        # mlp.c_proj -> our mlp.proj (same story, name-only mismatch)
        w_mlp_proj_hf = sd_hf[f'transformer.h.{i}.mlp.c_proj.weight']
        b_mlp_proj_hf = sd_hf[f'transformer.h.{i}.mlp.c_proj.bias']
        with torch.no_grad():
            sd[f'transformer.h.{i}.mlp.proj.weight'].copy_(w_mlp_proj_hf.t())
            sd[f'transformer.h.{i}.mlp.proj.bias'].copy_(b_mlp_proj_hf)

    return model


def test_matches_huggingface_gpt2():
    """Correctness test: loads real GPT-2 weights into our from-scratch model
    and checks its output logits match HuggingFace's GPT2LMHeadModel on the
    same input, within float tolerance. This is the test that actually proves
    the architecture is right -- a successful training run only tells you the
    code runs, not that it's correct."""
    from transformers import GPT2LMHeadModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    enc = tiktoken.get_encoding("gpt2")
    prompt = "The quick brown fox jumps over the lazy"
    tokens = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)

    ours = from_pretrained('gpt2').to(device).eval()
    theirs = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

    with torch.no_grad():
        our_logits, _ = ours(tokens)
        their_logits = theirs(tokens).logits

    max_abs_diff = (our_logits - their_logits).abs().max().item()
    print(f"max abs logit difference: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, f"logits diverge too much: {max_abs_diff}"

    our_next = our_logits[0, -1].argmax().item()
    their_next = their_logits[0, -1].argmax().item()
    assert our_next == their_next, (
        f"predicted next token differs: ours={enc.decode([our_next])!r} "
        f"vs theirs={enc.decode([their_next])!r}"
    )


def test_overfit_single_batch():
    # Sanity check: model should be able to drive loss to ~0 on one tiny
    # repeated batch. If it can't, something in the forward/backward pass or
    # optimizer setup is broken, independent of any real training data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    config = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-2, device=device)

    x = torch.randint(0, 65, (2, 16)).to(device)
    y = torch.randint(0, 65, (2, 16)).to(device)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.1, "model failed to overfit a single tiny batch"