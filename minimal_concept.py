#%% [markdown]
# # Week‑4 Prefix‑LM Captioning 🪄 CLIP Patches → Tiny Decoder (v3)
#
# **v3 key ideas**
# * Use **all CLIP ViT patch embeddings** as a *frozen prefix*.
# * Re‑use CLIP's **frozen token‑embedding matrix** for the decoder.
# * One single Transformer stream:  
#   `[patch₀ … patchₙ] BOS  caption_tokens … EOS  PAD …`
# * Decoder predicts *only* the caption tokens; patches are context.
#
# Check console logs for:
# * patch count `P`, caption length `L_max`
# * loss dropping from ≈ 20 → single digits
# * validation BLEU increasing each epoch

#%%
import torch, torch.nn as nn
from transformers import CLIPConfig, CLIPModel
from contextlib import nullcontext
import random
torch.manual_seed(42)
random.seed(42)
import numpy as np
np.random.seed(42)
import evaluate

#%% [markdown]
# ## Stage 1 – Tokenizer & Vocabulary (from CLIP)
#
# We re‑use the exact BPE vocabulary that CLIP's text tower was trained on.
# A `<pad>` token is appended because the original model never needed one.
# BOS and EOS are already present as special tokens in the pretrained config.
#%%

from transformers import CLIPTokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ------------------------------------------------------------------
# Ensure PAD token has its **own** unique ID (not shared with EOS)
# ------------------------------------------------------------------
if clip_tokenizer.pad_token_id is None or clip_tokenizer.pad_token_id == clip_tokenizer.eos_token_id:
    # Pick a pad token string that is guaranteed not to exist
    pad_token_str = "<|pad_extra|>"
    clip_tokenizer.add_tokens([pad_token_str])         # append to vocab
    clip_tokenizer.pad_token = pad_token_str           # register as pad
PAD_ID = clip_tokenizer.pad_token_id

BOS_ID = clip_tokenizer.bos_token_id      # 49406  <|startoftext|>
EOS_ID = clip_tokenizer.eos_token_id      # 49407  <|endoftext|>

print(f"Vocab size with PAD: {len(clip_tokenizer)}")
print(f"Special IDs – PAD:{PAD_ID}, BOS:{BOS_ID}, EOS:{EOS_ID}")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    try:
        torch.empty(1, device="mps")
        device = "mps"
    except RuntimeError:
        device = "cpu"
else:
    device = "cpu"

# --- Load full CLIP (vision + text) once ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
for p in clip_model.parameters():
    p.requires_grad = False

vision = clip_model.vision_model  
clip_text_emb = clip_model.text_model.embeddings.token_embedding.weight  # (49408,512)
PATCH_DIM = clip_text_emb.size(1)   
# ------------------------------------------------------------
# Helper to fetch patch sequence from CLIP ViT
# ------------------------------------------------------------
def get_patch_sequence(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, P, 512) patch embeddings from CLIP ViT.
    P ≈ 50 for 224×224 images.
    """
    with torch.no_grad():
        return vision(pixel_values).last_hidden_state

#%% [markdown]
# ## Stage 2 – Load Flickr30k & Build a Caption DataLoader
#
# We fetch the HuggingFace parquet version of Flickr30k (≈ 31 k images).
# For the demo we load only 1 % of the training split to keep runtime light.
# The collate function:
# * preprocesses the image with CLIP's own resize + normalise;
# * builds `decoder_input_ids` (BOS+tokens) and `target_ids` (tokens+EOS);
# * right‑pads to the batch max length with `PAD_ID`.
#%%
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# CLIP image preprocessing
preprocess_clip = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

dataset_full = load_dataset("nlphuji/flickr30k", split="test[:1%]")
print("Loaded subset size:", len(dataset_full))

def collate_caption(batch):
    """Collate a batch of Flickr30k samples for CLIP captioning."""
    image_tensors = []
    decoder_inputs = []
    label_targets = []

    for item in batch:
        # Use first caption; clip_tokenizer handles truncation
        caption_tokens = clip_tokenizer(
            item["caption"][0],
            truncation=True,
            max_length=64
        ).input_ids

        # Build BOS + caption  / target = caption + EOS
        decoder_in  = [BOS_ID] + caption_tokens
        target_out  = caption_tokens + [EOS_ID]

        # The image is already a PIL image, just convert to RGB if needed
        image = item["image"].convert("RGB") 
        image_tensors.append(preprocess_clip(image))

        decoder_inputs.append(torch.tensor(decoder_in, dtype=torch.long))
        label_targets.append(torch.tensor(target_out, dtype=torch.long))

    # Pad sequences
    decoder_inputs_padded = pad_sequence(decoder_inputs,
                                         batch_first=True,
                                         padding_value=PAD_ID)
    label_targets_padded  = pad_sequence(label_targets,
                                         batch_first=True,
                                         padding_value=PAD_ID)

    return (
        torch.stack(image_tensors),
        decoder_inputs_padded,
        label_targets_padded
    )

dataloader = DataLoader(
    dataset_full,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_caption
)
images_batch, dec_inputs_batch, targets_batch = next(iter(dataloader))
print("Batch shapes – images:", images_batch.shape,
      "dec_in:", dec_inputs_batch.shape,
      "targets:", targets_batch.shape)

# ------------------------------------------------------------
# 2.  Tiny decoder with automatic projection layer
# ------------------------------------------------------------
class TinyDecoder(nn.Module):
    """
    A minimal prefix-LM decoder: attends to ViT patch embeddings, autoregressively generates text.
    Args:
        enc_dim: Patch embedding dim (should match PATCH_DIM).
        dec_dim: Decoder d_model.
        vocab:   Vocabulary size.
        n_layers / n_heads: Transformer depth & width.
        max_len: Max sequence length (controls positional embedding & mask).
    """
    def __init__(self, enc_dim=PATCH_DIM, dec_dim=768, vocab=1000,
                 n_layers=1, n_heads=8, max_len=256):
        super().__init__()
        assert dec_dim % n_heads == 0, "n_heads must divide d_model"
        self.proj = (nn.Identity() if enc_dim == dec_dim else nn.Linear(enc_dim, dec_dim, bias=False))
        self.tok_emb = nn.Embedding(vocab, dec_dim)
        with torch.no_grad():
            self.tok_emb.weight[: clip_text_emb.size(0)].copy_(clip_text_emb)
        for p in self.tok_emb.parameters():
            p.requires_grad = False
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dec_dim))
        dec_layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=n_heads, batch_first=True)
        if hasattr(dec_layer, "_set_gradient_checkpointing"):
            dec_layer._set_gradient_checkpointing(True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dec_dim, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.register_buffer(
            "causal_mask",
            torch.ones(max_len, max_len, dtype=torch.bool).triu(1)
        )

    def forward(self, patch_emb: torch.Tensor, caption_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            patch_emb: (B, P, D) frozen prefix from ViT.
            caption_ids: (B, L) caption token IDs (BOS + words + PAD)
        Returns:
            logits (B, P+L, vocab)
        """
        B, P, _ = patch_emb.shape
        L = caption_ids.size(1)

        patch_emb = self.proj(patch_emb)  # Project patches from 768→512
        text_emb = self.tok_emb(caption_ids)
        seq = torch.cat([patch_emb, text_emb], dim=1)          # (B, P+L, D)
        seq = seq + self.pos_emb[:, :P+L]

        # causal mask
        mask = self.causal_mask[:P+L, :P+L].clone()
        mask[:P, :] = False                                    # patches are visible

        pad_mask = torch.zeros(B, P+L, dtype=torch.bool, device=seq.device)
        pad_mask[:, P:] = (caption_ids == PAD_ID)

        out = self.decoder(seq, seq,
                           tgt_mask=mask,
                           tgt_key_padding_mask=pad_mask)
        return self.lm_head(out)

# ------------------------------------------------------------
# 2b.  Greedy generation helper
# ------------------------------------------------------------
@torch.no_grad()
def greedy_generate(model: TinyDecoder,
                    vision_fn,
                    image: torch.Tensor,
                    bos_token: int = BOS_ID,
                    eos_token: int = EOS_ID,
                    max_len: int = 20) -> torch.LongTensor:
    """
    Run naive greedy decoding given a single image tensor on `device`.
    Returns a 1‑D tensor of generated token IDs (including EOS).
    """
    patch_seq = vision_fn(image.to(device).unsqueeze(0))
    generated = [bos_token]
    for _ in range(max_len):
        inp = torch.tensor(generated, device=patch_seq.device).unsqueeze(0)
        logits = model(patch_seq, inp)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == eos_token:
            break
    return torch.tensor(generated, dtype=torch.long)

# ------------------------------------------------------------
# Utility – BLEU evaluator on a DataLoader
# ------------------------------------------------------------
bleu_metric = evaluate.load("bleu")

@torch.no_grad()
def compute_bleu(model, vision_fn, dataloader, max_batches: int = 25):
    """
    Compute corpus BLEU-4 on the dataloader (truncated to max_batches for speed).
    """
    model.eval()
    preds, refs = [], []
    for b, (imgs, _, labels) in enumerate(dataloader):
        if b >= max_batches: break
        imgs = imgs.to(device)
        ids  = greedy_generate(model, get_patch_sequence, imgs[0])
        preds.append(clip_tokenizer.decode(ids.tolist(), skip_special_tokens=True))
        ref_txt = clip_tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)
        refs.append([ref_txt])
    score = bleu_metric.compute(predictions=preds, references=refs)["bleu"]
    return score

#%% [markdown]
# ## Stage 3 – Mini Train / Validation / Test Splits
#
# To keep development light on an M‑series laptop, we sample tiny subsets:
# * **Train** 5 % of Flickr30k train split  
# * **Val** 1 % of Flickr30k validation split  
# * **Test**  1 % of Flickr30k test split  
#
# Use `NUM_EPOCHS = 2` and `BATCH_SIZE = 4` for a quick sanity‑run. Feel free
# to enlarge once the loop is stable.
#%%

SUBSET_TRAIN = "test[:5%]"     # First 20% for training
SUBSET_VAL   = "test[5%:6%]"   # Next 2% for validation
SUBSET_TEST  = "test[6%:7%]"   # Next 2% for testing

NUM_EPOCHS  = 3
BATCH_SIZE  = 4
LR          = 3e-4

train_set = load_dataset("nlphuji/flickr30k", split=SUBSET_TRAIN)
val_set   = load_dataset("nlphuji/flickr30k", split=SUBSET_VAL)
test_set  = load_dataset("nlphuji/flickr30k", split=SUBSET_TEST)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_caption)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_caption)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_caption)

print(f"Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)}")

# Display a sample caption from the training set
raw_caption = train_set[0]["caption"][0]
print("Raw Flickr caption:", raw_caption)

# Get a batch to see tokenized version
sample_batch = next(iter(train_loader))
dec_inputs_batch = sample_batch[1]
print("Tokenizer decode :", clip_tokenizer.decode(
         dec_inputs_batch[0].tolist(), skip_special_tokens=True))

#%% [markdown]
# ## Stage 4 – Tiny Training Loop
#
# We train for `NUM_EPOCHS` and log train / val loss each epoch.  
# No fancy schedulers yet – keep the loop minimal.
#%%
model = TinyDecoder(enc_dim=768,  # Patch embedding dim is 768
                    dec_dim=512,  # Text embedding dim is 512
                    n_layers=3,
                    n_heads=8,
                    vocab=len(clip_tokenizer)).to(device)

loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def run_epoch(loader, train: bool):
    running_loss = 0.0
    steps = 0
    model.train(mode=train)
    for images, dec_in, targets in loader:
        images, dec_in, targets = (images.to(device),
                                   dec_in.to(device),
                                   targets.to(device))
        patch_seq = get_patch_sequence(images)                 # (B,P,512)
        with torch.set_grad_enabled(train):
            if steps == 0:
                print(f"Patch seq shape: {patch_seq.shape}")
            logits   = model(patch_seq, dec_in)

            B, P, _ = patch_seq.shape
            labels_full = torch.full((B, P + dec_in.size(1)),
                                     -100, device=device)
            labels_full[:, P:] = targets
            loss     = loss_fn(logits.view(-1, logits.size(-1)),
                               labels_full.view(-1))
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        running_loss += loss.item()
        steps += 1
    return running_loss / steps

best_bleu = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader,   train=False)
    bleu_val = compute_bleu(model, get_patch_sequence, val_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | train {train_loss:.2f} | val {val_loss:.2f} | BLEU {bleu_val:.3f}")
    print(f"... BLEU {bleu_val:.3f} | best {best_bleu:.3f}")
    if bleu_val >= best_bleu:
        best_bleu = bleu_val
        torch.save(model.state_dict(), "tiny_decoder_best.pt")
        print(f"✓ New best BLEU {best_bleu:.3f} – checkpoint saved.")

#%% [markdown]
# ## Stage 5 – Quick Test‑set Inference
#%%
model.eval()
images_test, _, _ = next(iter(test_loader))
sample_caption_ids = greedy_generate(model, get_patch_sequence, images_test[0])
print("Generated IDs:", sample_caption_ids.tolist())
print("→", clip_tokenizer.decode(sample_caption_ids.tolist(),
                                 skip_special_tokens=True))
