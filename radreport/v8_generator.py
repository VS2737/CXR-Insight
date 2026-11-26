"""Frozen RAD-DINO tokens -> medical report via GPT-2 cross-attention.

Train only on actual IU train split. We deliberately limit trainable decoder
parameters for an 8 GB RTX 4060. Cross-attention starts RANDOM and requires
training; the model is not medically accurate before empirical evaluation.
"""
from __future__ import annotations
from pathlib import Path


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def select_trainable_parameters(model, *, train_last_blocks: int = 2):
    if not 0 <= train_last_blocks <= len(model.decoder.transformer.h):
        raise ValueError('train_last_blocks out of range')
    for p in model.decoder.parameters():
        p.requires_grad_(False)
    # Cross-attention is randomly initialized when loading GPT-2. Train it in
    # every block, plus optional final language blocks and final LayerNorm.
    for block in model.decoder.transformer.h:
        for attr in ('crossattention', 'ln_cross_attn'):
            layer = getattr(block, attr, None)
            if layer is None:
                raise ValueError('GPT-2 cross-attention module missing')
            for p in layer.parameters():
                p.requires_grad_(True)
    if train_last_blocks:
        for block in model.decoder.transformer.h[-train_last_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
    for p in model.decoder.transformer.ln_f.parameters():
        p.requires_grad_(True)
    for p in model.bridge.parameters():
        p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_v8_model(pretrained: bool = True, train_last_blocks: int = 2):
    from torch import nn
    from transformers import GPT2Config, GPT2LMHeadModel

    class _Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.bridge = nn.Sequential(nn.Linear(768, 768), nn.LayerNorm(768))
            if pretrained:
                cfg = GPT2Config.from_pretrained('gpt2')
            else:
                cfg = GPT2Config(n_embd=768, n_layer=12, n_head=12, vocab_size=50257)
            cfg.add_cross_attention = True
            self.decoder = (GPT2LMHeadModel.from_pretrained('gpt2', config=cfg)
                            if pretrained else GPT2LMHeadModel(cfg))

        def forward(self, tokens, ids, mask, labels):
            return self.decoder(input_ids=ids, attention_mask=mask,
                encoder_hidden_states=self.bridge(tokens.float()),
                labels=labels, use_cache=False)

        @__import__('torch').no_grad()
        def sample_reports(self, tokens, tokenizer, n=5, max_new_tokens=160,
                           temperature=.78, top_p=.9):
            import torch
            self.eval()
            if tokens.shape[0] != 1:
                raise ValueError('Only one input X-ray per request')
            mem = self.bridge(tokens.float()).repeat(n, 1, 1)
            eos = tokenizer.eos_token_id
            seq = torch.full((n, 1), eos, dtype=torch.long, device=tokens.device)
            finished = torch.zeros(n, dtype=torch.bool, device=tokens.device)
            past = None
            for _ in range(max_new_tokens):
                out = self.decoder(input_ids=seq[:, -1:] if past is not None else seq,
                                   encoder_hidden_states=mem, past_key_values=past,
                                   use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :] / max(temperature, 1e-4)
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = logits.softmax(dim=-1)
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cutoff = (torch.cumsum(sorted_p, dim=-1) - sorted_p) > top_p
                sorted_p[cutoff] = 0
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                chosen = sorted_i.gather(1, torch.multinomial(sorted_p, 1))
                chosen[finished] = eos
                seq = torch.cat((seq, chosen), 1)
                finished |= (chosen[:, 0] == eos)
                if finished.all():
                    break
            return [s.strip() for s in tokenizer.batch_decode(seq[:,1:].cpu(), skip_special_tokens=True)]

    model = _Generator()
    select_trainable_parameters(model, train_last_blocks=train_last_blocks)
    return model


def save_v8(model, folder: str, metadata: dict):
    import torch, json
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':model.state_dict(), 'metadata':metadata}, out / 'generator.pt')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def load_v8(folder: str, device: str):
    import torch
    p = Path(folder) / 'generator.pt'
    # Only load checkpoints you created or trust.
    try:
        saved = torch.load(p, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot safely load v8 checkpoint: {exc}') from exc
    meta = saved['metadata']
    if meta.get('architecture') != 'rad-dino-gpt2-v8' or meta.get('grid') != 7:
        raise ValueError('Unexpected v8 checkpoint architecture or feature grid')
    model = build_v8_model(pretrained=False, train_last_blocks=meta.get('train_last_blocks',2))
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), meta, get_tokenizer()
