#%% [markdown]
# # Week‑4 Minimal Vision‑Language Merge 🪄🖼️→📝 (v1.1)
# 
# **New in v1.1** – automatic projection so the encoder's hidden size (e.g. 768 for CLIP/ViT‑B) always matches the decoder's embedding dim. No more shape‑mismatch errors like `mat1 and mat2 shapes cannot be multiplied (2×768 vs 512×1024)`.
# 
# Run on CPU with synthetic inputs to confirm plumbing; swap the synthetic loader for Flickr30k when you're online.
# 
# ---

#%%
import torch, torch.nn as nn
from transformers import CLIPConfig, CLIPModel, ViTConfig, ViTModel
from contextlib import nullcontext

PAD_ID = 0  # token id used for <pad>

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

def load_encoder(which="clip"):
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

vision, ENC_DIM = load_encoder("vit")   # swap to "vit" if you like
print(f"Encoder hidden size = {ENC_DIM}")

# ------------------------------------------------------------
# 2.  Tiny decoder with automatic projection layer
# ------------------------------------------------------------
class TinyDecoder(nn.Module):
    def __init__(self, enc_dim, dec_dim=512, vocab=1000, n_layers=1, n_heads=8, max_len=128):
        super().__init__()
        assert dec_dim % n_heads == 0, "n_heads must divide d_model"
        self.proj = (nn.Identity() if enc_dim == dec_dim else nn.Linear(enc_dim, dec_dim, bias=False))
        self.tok_emb = nn.Embedding(vocab, dec_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dec_dim))
        dec_layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dec_dim, vocab, bias=False)
        
        # Pre-cache the causal mask at max size
        self.register_buffer(
            "causal_mask",
            torch.ones(max_len, max_len, dtype=torch.bool).triu(1)
        )

    def forward(self, img_emb, tgt_ids):
        """img_emb (B, D_enc) – will be projected to (B, 1, D_dec)"""
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
# 3.  Minimal end‑to‑end smoke test (synthetic)
# ------------------------------------------------------------
B = 2
pixels = torch.randn(B, 3, 224, 224, device=device)
ids    = torch.randint(0, 999, (B, 6), device=device)

# Make sure we use the ACTUAL encoder dimension from the features
model = TinyDecoder(enc_dim=ENC_DIM, dec_dim=512).to(device).eval()

with torch.no_grad():
    img_feats = vision(pixels)                          # (B, ENC_DIM)
    print(f"Image features shape: {img_feats.shape}")
    
    amp_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()
    with amp_ctx:
        logits = model(img_feats, ids)
    
print("✅  Forward pass OK – logits shape:", logits.shape)
