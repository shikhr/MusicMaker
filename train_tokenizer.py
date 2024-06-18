from miditok import REMI, TokenizerConfig
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(".env"))

midi_paths = list(Path("data/adl-piano-midi").glob("**/*.mid"))

hf_username = "shikhr"


# Define the training function
def train_tokenizer(midi_paths, vocab_size):
    print("starting training")
    config = TokenizerConfig(
        num_velocities=16, use_chords=True, use_programs=True, use_tempos=True
    )
    tokenizer = REMI(config)
    tokenizer.train(vocab_size=vocab_size, files_paths=midi_paths)
    print("ending training")

    tokenizer.push_to_hub(
        f"{hf_username}/miditok_mega{i}k",
        private=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.save_params(Path(f"tokenizer2_mega{i}k.json"))


# Define the tokenizer sizes
tokenizer_size = [6, 12, 18, 24]

# Train the tokenizer
for i in tokenizer_size:
    train_tokenizer(midi_paths, i * 1000)
