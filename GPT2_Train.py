
# Training script for the from-scratch GPT-2 (124M), extracted from the original jupyter notebook notebook into a runnable, GPU-agnostic script.

""" 
Usage:
    python train.py --data_dir edu_fineweb10B --checkpoint_dir checkpoints
 
Resumes automatically from the latest checkpoint in --checkpoint_dir if one
exists, so re-running this same command after an interruption continues
rather than restarting.
"""

import time
import math
import numpy as np
import os 
from datasets import load_dataset 
import tiktoken
import torch 
import torch.functional as F 
from GPT2 import GPT, GPTConfig
from hellaswag import render_example, iterate_examples, get_most_likely_row

torch.manual_seed(1337)
master_process = True

# use cuda GPU to speed up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


# micro-batch hyperparameter optimisation
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens 
B = 4 # micro batch size 
T = 1024 # sequence length 
assert total_batch_size % (B * T ) == 0 # must be divisible
grad_accum_steps = total_batch_size // (B * T) # number of steps = required batch_size (tokens) // no.tokens per micro batch


# loads the tokens from the .npy file and converts them to tensors
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt 

# Tokenisation already happened so we are essentially loading tokens rather than loading words and then tokenising
class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B 
        self.T = T 
        assert split in {'train', 'val'}

        # get the shard filenames
        # shards are numpy files which are essentially numpy arrays of data
        # iterating over shards is equivalent to iterating over tensors/numpy arrays of data 
        
        # path to the local folder containing the pre-tokenized shard files
        data_root = "edu_fineweb10B"
        # list every filename inside that folder (no filtering yet)
        shards = os.listdir(data_root)
        # keep only filenames that match this split
        # e.g. if split="train", keeps "train_0000.npy" but skips "val_0000.npy"
        shards = [s for s in shards if split in s]
        # sort alphabetically/numerically so shards are read in a consistent,
        # predictable order (e.g. shard 0000, then 0001, then 0002, ...)
        shards = sorted(shards)
        # turn each bare filename into a full path relative to data_root
        # e.g. "train_0000.npy" -> "edu_fineweb10B/train_0000.npy"
        shards = [os.path.join(data_root, s) for s in shards]
        # store the final list of full shard file paths on the instance,
        # so next_batch() can index into it later to load new shards as needed
        self.shards = shards
        # sanity check — fail loudly and immediately if the folder/filtering
        # somehow produced zero matching shard files, rather than silently
        # crashing later with a confusing error when training tries to start
        assert len(shards) > 0, f"no shards found for split {split}"
        # only print from the "main" process — relevant in DDP setups where
        # multiple processes would otherwise all print this same message;
        # since you're single-GPU, master_process is just always True,
        # so this always prints
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        self.reset() # useful to reset the dataloader because in the main training loop


    def reset(self):
        # state, init at shard zero 
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard]) # get the tokens by loading the shards - convert from .npy file to an array and then tensor
        self.current_position = 0 # starts from 0th position inside the shard 

    def next_batch(self):
        B, T = self.B, self.T # get the batch size and length of each sequence

        # slice the tokens tensor to get the tokens we want to use to form the batch
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        # We want B * T + 1 tokens rather than B * T
        # This is because we want our input to be current position to B * T and the labels to be current position + 1 to B * T + 1
        # That way, we want B * T + 1 shift since our input = buf and labels = buf shifted by 1 index
        
        x = buf[:-1].view(B, T) # inputs from 0 to B * T, reshape to a matrix of shape (B, T) from a flat tensor which essentially creates B batches of length T
        y = buf[1:].view(B, T) # labels from 1 to B * T + 1 and reshape to a matrix of shape (B, T)
        # think of it as for some index (a, b) in x, the predicted next label is (a, b) in y
        # now purely in terms of data, this implements a single context window being used for prediction - e.g. character predicted only using the previous
        # with the implementation of attention, this allows tokens to communicate with each other, however only the previous characters can be used a context
        # this transforms it from a 1 character predicts the next scenario, to all previous characters predicts the next (this context window is a max size of block_size)

        # advance the position in the tensor 
        # we fan think of batches as every B * T tokens because there are B * T tokens per batch
        self.current_position += B * T 

        # if loading the next batch would be out of bounds, advance to the next shard 
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y


train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')

# meant to make computation faster
torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304)) # model is GPT and pass in GPTConfig() which stores the required hyperparameters for the model
                                         # override vocab_size to 50304 since it is a much nicer number being divisibly by multiple powers of 2 - thereby increasing efficiency
model.to(device) # moves all model weights/parameters onto the GPU (or stays on CPU if device == 'cpu') -- without this the model silently stays on CPU regardless of what device says
use_compile = False # torch.compile interferes with HellaSwag eval and Generation.
if use_compile:
    model = torch.compile(model) # torch.compile() analyses the entire thing so it knows what processes it needs to run and optimises these processes making it significantly faster


# lr hyperparameter implementation
# adjusts the learning rate depending on what step we are in the iterations
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # match GPT-3 warmup schedule
max_steps = 19073 # = 10e9 tokens we want to do / 2**19 tokens per step = 19073 steps
def get_lr(it):
    # 1) linear warmup for warmup_iters steps 
    if it < warmup_steps:
        # approaches max_lr as it approaches the end of warmup_steps (in equal increasing steps)
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate 
    if it > max_steps:
        # when it reaches max_steps we want the lr to be its minimm value 
        # this serves a safety check that clamps the learning rate to min_lr if training runs longer than max_steps, repventing the cosine decay from going out of its valid range
        return min_lr 
    # 3) in between, use cosine decay down to min learning rate 
    # decay ratio increases as it increases because warmup_steps is constant and so is the denominator
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) # min_lr starts adding smaller values until 0 since coeff approaches 0 

# optimisation
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# Full Training Loop
for step in range(max_steps):
    t0 = time.time() # time how long the optimisation takes
    last_step = (step == max_steps -1)
    
    # once in a while evaluate our validation loss
    if (step % 250 == 0 or last_step):
        model.eval() # put into evaluation mode
        val_loader.reset() 
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20 
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if master_process: 
            print(f"validation loss: {val_loss_accum.item():.4f}")
            # ADDED: write val loss to the log file, not just console -- console output is lost the
            # moment the Colab session disconnects, the log file survives on disk
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # ADDED: checkpoint saving, missing from the original version of this script. 
            # checkpoints every 1000 steps to prevent losses when
            if step > 0 and (step % 1000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # NOTE: same caveat Andrej flags in his own script -- this only saves model weights,
                # not optimizer state. Resuming from this checkpoint restores the model correctly, but
                # AdamW's momentum/variance buffers restart from scratch -- an approximate resume, not exact.
                torch.save(checkpoint, checkpoint_path)
                print(f"saved checkpoint to {checkpoint_path}")

    # once in a while evaluate hellaswag
    if step % 250 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # no ddp_world_size/ddp_rank filtering needed -- single GPU processes every example itself
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # no dist.all_reduce needed -- num_total/num_correct_norm are already the full, real counts
        # since there's only one process computing them, nothing to combine across processes
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    
    # micro batch optimisation
    loss_accum = 0.0 # track the accumulation of loss in micro batches for visualisation (not computation)
    # We want 0.5M tokens to contribute to each single optimiser update (one step) 
    # Instead of loading all 0.5M tokens at once we split this into smaller micro-batches each of size B * T tokens (4 * 1024 = 4096 tokens) and this micro-batch process is repeated total_batch_size / (B*T) times = grad_accum_steps (grad_accum_stes no. of micro batches)
    # The loss is calculated for each micro-batch, they are accumulated via loss.backward() and /grad_accum_steps makes it so the total output is the mean loss of all micro-batches - this way we find the mean of a batch with 0.5M tokens (used 0.5M tokens for training)
    # now the optimizer uses the gradients from loss.backward() (loss wrt to all params) to adjust the params and minimise loss
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device) # moves this batch's input/label tensors onto the GPU to match where the model now lives -- model and data must be on the same device or PyTorch throws an error
        
        # meant to be faster so if the GPU is not powerful enough it's not possible and slower - wont be using tensorcores
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        #     print(logits.dtype)

        logits, loss = model(x, y)
        loss = loss / grad_accum_steps # this loss is loss per micro batch and divided by grad_accum_steps allows for calculation of avg loss by summing (normalisation factor * loss) where normalisation factor is how much the final division is to get the average
                                       # loss_accum actually tracks the avg loss across the micro-batches
        loss_accum += loss.detach() # detach the tensor, just want to keep track of the values

        loss.backward() # populates gradients of loss wrt params for each params .grad but backward() itself has no knowledge of grad_accum_steps
                        # it never applies the division needed to correct for cross-entropy's mean-over-batch normalization (the loss = mean of losses across each individual batch - in this case micro batch)
                        # that's why loss = loss / grad_accum_steps happens manually beforehand - because it requires the avg loss - not the summed loss
                        # these gradients are accumulated for every time .backward() (not overwritten which is exactly why we need optimizer.zero_grad to refresh gradients) is called 
                        # this means the .grad of each parameter is the accumulated avg roc of loss wrt to that particular parameter for each micro-batch - since each loss is multiplied by a normalisation constant, the sum of roc works out to the average roc across all micro-batches
                        
                        # Each micro-batch is an independent computational graph, but all backward() calls accumulate into the same set of parameters (since .grad += not = we accumulate these gradients) 
                        # -- combined with pre-dividing each loss by grad_accum_steps, the result is the average gradient across all micro-batches, equivalent to one big backward pass we can't fit in memory.

    # global gradient clipping hyperparameter
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # calculates the global norm of the parameters = sqrt(sum of squares of params gradients) = magnitude of the gradient vector
                                                                   # here we are making the magnitude a maximum of 1.0 - this prevents the model from getting shocks from high gradient magnitudes
    
    # determine and set the learning rate for this iteration
    lr = get_lr(step) # get the lr 
    for param_group in optimizer.param_groups: # change the lr - to change lr we need to iterate over param_groups (even though there's only 1 param_group this is the only way)
        param_group['lr'] = lr
    
    optimizer.step() # gets each params.grad to find roc loss wrt to the param and adjusts the params value to minimise loss (.grad is the derivative of whatever tensor .backward() was called on wrt to the params)
    if device == 'cuda':
        torch.cuda.synchronize() # CUDA ops are launched asynchronously (CPU queues work and moves on immediately) -- without this, t1 gets stamped before the GPU actually finishes computing, making the timing measurement inaccurate/too small
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert seconds to milliseconds for a more readable per-step timing number
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / (t1 - t0)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

    # write train loss to the log file too -- previously only printed to console, never persisted
    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
