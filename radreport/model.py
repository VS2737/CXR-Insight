"""Research VLM: pretrained EfficientNet-B0 visual tokens conditioned GPT-2 via cross-attention.

Install requirements-ml.txt before importing. Hugging Face weights download on first run.
The cross-attention layers and visual bridge start randomly initialized and MUST
be trained; pretrained GPT-2 is not already a radiology image-to-report model.
"""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ReportGenerator(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.encoder = backbone.features
        self.bridge = nn.Sequential(nn.Linear(1280, 768), nn.LayerNorm(768))
        config = GPT2Config.from_pretrained('gpt2') if pretrained else GPT2Config(
            n_embd=768, n_layer=12, n_head=12, add_cross_attention=True)
        config.add_cross_attention = True
        self.decoder = (GPT2LMHeadModel.from_pretrained('gpt2', config=config)
                        if pretrained else GPT2LMHeadModel(config))
        if freeze_encoder:
            for param in self.encoder.parameters(): param.requires_grad_(False)
        self.freeze_encoder = freeze_encoder

    def visual_tokens(self, images):
        if self.freeze_encoder:
            self.encoder.eval()  # do not update ImageNet BatchNorm statistics
            with torch.no_grad():
                z = self.encoder(images)
        else:
            z = self.encoder(images)
        return self.bridge(z.flatten(2).transpose(1, 2))  # B, 49, 768

    def forward(self, images, input_ids, attention_mask, labels):
        memory = self.visual_tokens(images)
        return self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                            encoder_hidden_states=memory, labels=labels, use_cache=False)

    @torch.no_grad()
    def sample_reports(self, image_batch, tokenizer, n=5, max_new_tokens=65, temperature=0.85, top_p=0.9):
        """Five independent autoregressive samples with KV cache; samples share image features."""
        self.eval()
        assert image_batch.shape[0] == 1, 'Inference supports one X-ray per request.'
        memory = self.visual_tokens(image_batch).repeat(n, 1, 1)
        bos = tokenizer.eos_token_id
        tokens = torch.full((n, 1), bos, dtype=torch.long, device=image_batch.device)
        past = None
        finished = torch.zeros(n, dtype=torch.bool, device=image_batch.device)
        for _ in range(max_new_tokens):
            out = self.decoder(input_ids=tokens[:, -1:] if past is not None else tokens,
                               encoder_hidden_states=memory, past_key_values=past,
                               use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :] / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            mask = (torch.cumsum(sorted_p, dim=-1) - sorted_p) > top_p
            sorted_p[mask] = 0
            sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sorted_p, 1)
            next_token = sorted_i.gather(-1, choice)
            next_token[finished] = bos
            tokens = torch.cat((tokens, next_token), dim=1)
            finished |= (next_token.squeeze(-1) == bos)
            if finished.all(): break
        return [x.strip() for x in tokenizer.batch_decode(tokens[:, 1:].cpu(), skip_special_tokens=True)]


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def save_generator(model, output: str, metadata: dict):
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'freeze_encoder': model.freeze_encoder, 'metadata': metadata}, p / 'generator.pt')
    import json
    (p / 'generator_metadata.json').write_text(json.dumps(metadata, indent=2))


def load_generator(model_dir: str, device: str):
    # checkpoint is local and must be trusted: torch.load can deserialize pickle
    ckpt = torch.load(Path(model_dir) / 'generator.pt', map_location='cpu', weights_only=False)
    model = ReportGenerator(pretrained=False, freeze_encoder=ckpt.get('freeze_encoder', True))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    return model, ckpt['metadata'], get_tokenizer()
