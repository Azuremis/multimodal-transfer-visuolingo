#%% [markdown]
# # Week‑4 Vision‑Language Captioning 🪄 CLIP → Tiny‑Decoder (v2)
# 
# **What changed in v2**
# 1. Loads **CLIPTokenizer** and extends it with a dedicated `<pad>` token.
# 2. Uses that vocabulary (≈ 49 k) for the decoder.
# 3. Adds a Flickr30k loader & collate function for caption fine‑tuning.
# 4. Demonstrates a single mini‑batch forward + loss with the new data.
# 
# Run each stage top‑to‑bottom; cells are grouped logically with headings.

#%%
import torch, torch.nn as nn
from transformers import CLIPConfig, CLIPModel, ViTConfig, ViTModel
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

# ------------------------------------------------------------
# 1.  Encoder loader (falls back to random weights offline)
# ------------------------------------------------------------

def get_enc_dim(enc):
    # CLIP
    if hasattr(enc.config, "vision_embed_dim"):
        return enc.config.vision_embed_dim
    if hasattr(enc.config, "vision_config"):
        return enc.config.vision_config.hidden_size
    # ViT or others
    return enc.config.hidden_size

def load_encoder(which: str = "clip"):
    """
    Load a frozen vision encoder (CLIP or ViT).

    Args:
        which: "clip" or "vit".

    Returns:
        vision_fwd: Callable that maps pixel tensor -> pooled features.
        hidden_dim: Output feature dimension of the encoder.
    """
    if which == "clip":
        try:
            enc = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
        except Exception:
            print("⚠️  Offline – using random CLIP weights"); enc = CLIPModel(CLIPConfig()).to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
        # we only keep the vision tower → pool to (B, D)
        def vision_fwd(pix):
            outputs = enc.get_image_features(pixel_values=pix)
            return outputs
    else:
        try:
            enc = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
        except Exception:
            print("⚠️  Offline – using random ViT weights"); enc = ViTModel(ViTConfig()).to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
        def vision_fwd(pix):
            outputs = enc(pixel_values=pix)
            return outputs.pooler_output        # (B, D)
    
    # For safety, we'll verify the actual output dimension (outside the function definition)
    with torch.no_grad():
        sample_out = vision_fwd(torch.randn(1, 3, 224, 224, device=device))
        actual_dim = sample_out.shape[-1]
        print(f"Debug - Actual output dimension: {actual_dim}, Config dimension: {get_enc_dim(enc)}")
        if which == "clip":
            print(f"Debug - CLIP features shape: {sample_out.shape}")
        else:
            print(f"Debug - ViT features shape: {sample_out.shape}")
        hidden = actual_dim
    
    return vision_fwd, hidden

vision, ENC_DIM = load_encoder("clip")   # swap to "vit" if you like
print(f"Encoder hidden size = {ENC_DIM}")

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
    A minimal cross‑attention decoder that turns an image embedding into
    an autoregressive text sequence.

    Args:
        enc_dim: Dimension of the encoder CLS/vector.
        dec_dim: Model d_model for the text decoder.
        vocab:   Vocabulary size including special tokens.
        n_layers / n_heads: Transformer depth & width.
        max_len: Max sequence length (controls positional embedding & mask).
    """
    def __init__(self, enc_dim, dec_dim=512, vocab=1000, n_layers=1, n_heads=8, max_len=128):
        super().__init__()
        assert dec_dim % n_heads == 0, "n_heads must divide d_model"
        self.proj = (nn.Identity() if enc_dim == dec_dim else nn.Linear(enc_dim, dec_dim, bias=False))
        self.tok_emb = nn.Embedding(vocab, dec_dim)
        # Initialise embeddings with a smaller std so early loss isn't gigantic
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dec_dim))
        dec_layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=n_heads, batch_first=True)
        # Enable gradient checkpointing to save memory (PyTorch ≥ 2.3)
        if hasattr(dec_layer, "_set_gradient_checkpointing"):
            dec_layer._set_gradient_checkpointing(True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dec_dim, vocab, bias=False)
        
        # Tie output projection weights to token embedding
        self.lm_head.weight = self.tok_emb.weight
        
        # Pre-cache the causal mask at max size
        self.register_buffer(
            "causal_mask",
            torch.ones(max_len, max_len, dtype=torch.bool).triu(1)
        )

    def forward(self, img_emb: torch.Tensor, tgt_ids: torch.LongTensor) -> torch.Tensor:
        """Run one full decoder pass.

        Args:
            img_emb: (B, D_enc) pooled image embeddings.
            tgt_ids: (B, L) token ids including <bos> and possibly <pad>.

        Returns:
            Logits of shape (B, L, vocab).
        """
        B, L = tgt_ids.shape
        assert L <= self.pos_emb.size(1), "Sequence length exceeds max_len"
        img = self.proj(img_emb).unsqueeze(1)           # (B,1,D_dec)
        tgt = self.tok_emb(tgt_ids) + self.pos_emb[:, :L]
        
        # Use the pre-cached mask, sliced to the current sequence length
        mask = self.causal_mask[:L, :L]
        # padding mask: True where tgt_ids is PAD_ID
        pad_mask = (tgt_ids == PAD_ID)
        out = self.decoder(
            tgt, img,
            tgt_mask=mask,
            tgt_key_padding_mask=pad_mask            # prevents attention to <pad>
        )
        return self.lm_head(out)                         # (B,L,V)

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
    img_feat = vision_fn(image.to(device).unsqueeze(0))   # (1, D)
    generated = [bos_token]
    for _ in range(max_len):
        inp = torch.tensor(generated, device=device).unsqueeze(0)  # (1, t)
        logits = model(img_feat, inp)                              # (1,t,V)
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
        ids  = greedy_generate(model, vision_fn, imgs[0])
        preds.append(clip_tokenizer.decode(ids.tolist(), skip_special_tokens=True))
        ref_txt = clip_tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)
        refs.append([ref_txt])
    score = bleu_metric.compute(predictions=preds, references=refs)["bleu"]
    return score

#%% [markdown]
# ## Stage 3 – Mini Train / Validation / Test Splits
#
# To keep development light on an M‑series laptop, we sample tiny subsets:
# * **Train**  5 % of Flickr30k train split  
# * **Val**   1 % of Flickr30k validation split  
# * **Test**   1 % of Flickr30k test split  
#
# Use `NUM_EPOCHS = 2` and `BATCH_SIZE = 4` for a quick sanity‑run. Feel free
# to enlarge once the loop is stable.
#%%

SUBSET_TRAIN = "test[:5%]"     # First 5% for training
SUBSET_VAL   = "test[5%:6%]"   # Next 1% for validation
SUBSET_TEST  = "test[6%:7%]"   # Next 1% for testing

NUM_EPOCHS  = 2
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
model = TinyDecoder(enc_dim=ENC_DIM,
                    dec_dim=512,
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
        with torch.set_grad_enabled(train):
            img_feat = vision(images)
            logits   = model(img_feat, dec_in)
            loss     = loss_fn(logits.view(-1, logits.size(-1)),
                               targets.view(-1))
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
    bleu_val = compute_bleu(model, vision, val_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | train {train_loss:.2f} | val {val_loss:.2f} | BLEU {bleu_val:.3f}")
    if bleu_val > best_bleu:
        best_bleu = bleu_val
        torch.save(model.state_dict(), "tiny_decoder_best.pt")
        print(f"✓ New best BLEU {best_bleu:.3f} – checkpoint saved.")

#%% [markdown]
# ## Stage 5 – Quick Test‑set Inference
#%%
model.eval()
images_test, _, _ = next(iter(test_loader))
sample_caption_ids = greedy_generate(model, vision, images_test[0])
print("Generated IDs:", sample_caption_ids.tolist())
print("→", clip_tokenizer.decode(sample_caption_ids.tolist(),
                                 skip_special_tokens=True))
