from model import GPT
from miditok import MusicTokenizer
from pathlib import Path
import torch
from dataclasses import dataclass
from midi2audio import FluidSynth

# Load the tokenizer
tokenizer = MusicTokenizer.from_pretrained("miditok2_12k")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = tokenizer.vocab_size
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False


model = GPT(GPTConfig())
sd = torch.load("m1.pt", map_location=device)
model.load_state_dict(sd)
model.to(device)

# Generate some music
out = model.generate(
    torch.tensor([[1]]).to(device), max_new_tokens=800, temperature=1.0, top_k=None
)

# Save the generated MIDI
tokenizer(out[0].tolist()).dump_midi("generated.mid")

# Convert the MIDI to audio
FluidSynth().midi_to_audio("generated.mid", f"generated.wav")
