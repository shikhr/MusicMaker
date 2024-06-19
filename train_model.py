import os
import time
import math
from contextlib import nullcontext
import torch
from dataclasses import dataclass


from model import GPT
from miditok import MusicTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator, split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Load the tokenizer
tokenizer = MusicTokenizer.from_pretrained("miditok2_12k")

# load paths to the midi files
midi_paths = list(Path("adl-piano-midi").glob("**/*.mid"))
midi_paths.sort()

# create a directory to store the dataset chunks
os.rmdir("chunked")
os.makedirs("chunked", exist_ok=True)

dataset_chunks_dir = Path("chunked")

# split the dataset into chunks
split_files_for_training(
    files_paths=midi_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=1024,
)

# create the dataset
dataset = DatasetMIDI(
    files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)

# create the dataloader
collator = DataCollator(
    pad_token_id=tokenizer.pad_token_id, copy_inputs_as_labels=True, labels_pad_idx=-1
)
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collator, drop_last=True
)

# set the device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
print(device, ptdtype)
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
scaler.is_enabled()


# training settings
learning_rate = 6e-4  # max learning rate
max_iters = 1800  # total number of training iterations
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 30  # how many steps to warm up for
lr_decay_iters = 1800  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

citer = 1
accum = 4


# learning rate schedule
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# instantiate the model
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        tokenizer.vocab_size
    )  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = (
        False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


model = GPT(GPTConfig())
model.to(device)


# instantiate the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# training loop
def train():
    t0 = time.time()
    for epoch in range(10):
        print("epoch:", epoch)
        for bid, batch in enumerate(dataloader):
            x, y = batch["input_ids"], batch["labels"]
            x, y = x.to(device), y.to(device)
            with ctx:
                logits, loss = model(x, attn_mask=None, targets=y)
                loss = loss / accum
            scaler.scale(loss).backward()

            if (bid + 1) % accum != 0:
                continue
            lr = get_lr(citer)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if citer % 30 == 0:
                print(
                    f"iter {citer}: loss {loss.item()*accum:.4f}, time {dt*1000:.2f}ms, lr {lr:.10f}"
                )
            citer += 1


# train the model
train()

# save the model
torch.save(model.state_dict(), "m1.pt")
