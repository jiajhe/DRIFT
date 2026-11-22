"""Qwen3.5-4B with interleaved, observed DRIFT soft tokens and all-layer LoRA."""

from collections import defaultdict
from functools import lru_cache
import json
import re
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from prompt_templates_v2 import SYSTEM, user_parts, response
from v2_features import ROOT, WIDTHS, Inputs

REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL = ROOT / "cache/hf-models/models--Qwen--Qwen3.5-4B/snapshots" / REV
SEP = "⟦DRIFT_SLOT⟧"
WEIGHTS = ROOT / "checkpoints/TSLM_training/v2.1"


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Linear(
            base.in_features,
            16,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.up = nn.Linear(
            16,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(0.05)
        self.active = False

    def forward(self, x):
        out = self.base(x)
        if not self.active:
            return out
        return out + 2 * self.up(self.down(self.dropout(x))).to(out.dtype)


class Projectors(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.maps = nn.ModuleDict()
        self.positions = nn.ParameterDict()
        counts = dict(
            temporal=5,
            local=1,
            component=3,
            role=1,
            sender_temporal=2,
            edge=5,
            returns=5,
        )
        for key, n in widths.items():
            mid = 512 if key == "component" else 1024
            self.maps[key] = nn.Sequential(
                nn.LayerNorm(n), nn.Linear(n, mid), nn.GELU(), nn.Linear(mid, 2560)
            )
            self.positions[key] = nn.Parameter(torch.randn(counts[key], 2560) * 0.02)

    def forward(self, key, v, pos, rms):
        x = self.maps[key](v) + self.positions[key][pos]
        return (F.rms_norm(x.float(), (2560,)) * rms).bfloat16()


class DriftTSLM(nn.Module):
    def __init__(self, variant="drift"):
        super().__init__()
        self.variant = variant
        self.source = Inputs()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        base = (
            AutoModelForImageTextToText.from_pretrained(
                MODEL,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            .requires_grad_(False)
            .eval()
        )
        self.decoder = base.model.language_model
        self.head = base.lm_head
        self.embed = self.decoder.get_input_embeddings()
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tokenizer.pad_token_id
        self.rms = float(self.embed.weight[::64].float().square().mean().sqrt())
        self.project = Projectors(
            WIDTHS if variant == "drift" else {"returns": 128}
        ).cuda()
        if variant == "returns":
            sys.path.insert(0, str(ROOT / "external/OpenTSLM/src"))
            from opentslm.model.encoder.TransformerCNNEncoder import (
                TransformerCNNEncoder,
            )

            statepath = (
                ROOT
                / "cache/hf-models/models--OpenTSLM--llama-3.2-1b-tsqa-sp/snapshots/1904441f0d87c458d6e9376de04ada5f4c9ab5b7/model_checkpoint.pt"
            )
            ck = torch.load(statepath, map_location="cpu", weights_only=True, mmap=True)
            self.encoder = TransformerCNNEncoder().cuda().requires_grad_(False).eval()
            self.encoder.load_state_dict(ck["encoder_state"])
        attention = []
        for j, layer in enumerate(self.decoder.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, LoRA(getattr(layer.mlp, name)))
            if layer.layer_type == "full_attention":
                attention.append(j)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, name, LoRA(getattr(layer.self_attn, name)))
        if len(self.decoder.layers) != 32 or attention != list(range(3, 32, 4)):
            raise ValueError("Unexpected Qwen layer contract")
        self.stage(1, False)

    def stage(self, s, training):
        self.eval()
        self.project.train(training)
        for layer in self.decoder.modules():
            if isinstance(layer, LoRA):
                layer.active = s == 2
                layer.up.requires_grad_(s == 2)
                layer.down.requires_grad_(s == 2)
                layer.dropout.train(training and s == 2)

    @lru_cache(maxsize=32768)
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    @lru_cache(maxsize=32768)
    def answer_span(self, text):
        boundary = text.index("<answer>") + len("<answer>")
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        # A BPE token can straddle the end of the opening tag and the start of JSON.
        first = next(
            i for i, (_, end) in enumerate(enc.offset_mapping) if end > boundary
        )
        ids = enc.input_ids + [self.eos]
        return ids, len(ids) - first

    def loss_mass(self, row):
        ids, answer_tokens = self.answer_span(response(row))
        return len(ids) + 2 * answer_tokens

    @lru_cache(maxsize=65536)
    def prefix(self, headers, question, counts):
        user = (
            "".join(h + SEP * n for h, n in zip(headers, counts, strict=True))
            + question
        )
        rendered = self.tokenizer.apply_chat_template(
            [dict(role="system", content=SYSTEM), dict(role="user", content=user)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chunks = rendered.split(SEP)
        seq = list(self.encode(chunks[0]))
        for j, chunk in enumerate(chunks[1:]):
            seq.append(-j - 1)
            seq.extend(self.encode(chunk))
        return seq

    def prepare(self, rows, mode, donors, supervised, *, continuations=None):
        seqs = []
        labs = []
        flat = []
        groups = defaultdict(list)
        raw = []
        for batch, r in enumerate(rows):
            heads, q = user_parts(r)
            if mode == "text":
                blocks = [[] for _ in heads]
            elif self.variant == "returns":
                d = self.source.return_day(r["date"])
                symbols = (
                    [r["i"], r["b"]]
                    if r["task"] == 3
                    else [r["i"], *d["senders"][r["i"]]]
                )
                blocks = []
                for j in symbols:
                    raw.append(d["returns_20d"][j])
                    blocks.append([("returns", p, None) for p in range(5)])
            else:
                blocks = self.source.stock(
                    r, donor=None if donors is None else donors[batch]
                )
            prefix = self.prefix(tuple(heads), q, tuple(map(len, blocks)))
            seq = prefix.copy()
            labels = [-100] * len(seq)
            if supervised:
                ans, _ = self.answer_span(response(r))
                seq += ans
                labels += ans
            if continuations is not None:
                if supervised:
                    raise ValueError("A generated prefix is an inference-only input")
                tail = self.encode(continuations[batch])
                seq.extend(tail)
                labels.extend([-100] * len(tail))
            off = len(flat)
            for block in blocks:
                for key, p, v in block:
                    groups[key].append((len(flat), p, v))
                    flat.append(key)
            seqs.append([s if s >= 0 else s - off for s in seq])
            labs.append(labels)
        lens = [len(s) for s in seqs]
        width = max(lens)
        ids = np.full((len(rows), width), self.pad, dtype="int64")
        masks = np.zeros_like(ids)
        labels = np.full_like(ids, -100)
        for i, (seq, lab) in enumerate(zip(seqs, labs, strict=True)):
            sl = slice(0, len(seq)) if supervised else slice(width - len(seq), width)
            ids[i, sl] = seq
            masks[i, sl] = 1
            labels[i, sl] = lab
        ids = torch.as_tensor(ids, device="cuda")
        em = self.embed(ids.clamp_min(0))
        if flat:
            soft = torch.zeros(len(flat), 2560, device="cuda", dtype=em.dtype)
            for key, group in groups.items():
                if mode == "zero" or (
                    mode == "local" and key not in ("temporal", "local")
                ):
                    continue
                loc, pos, values = zip(*group)
                if key == "returns":
                    x = torch.from_numpy(np.stack(raw)).cuda()
                    x = (x - x.mean(1, keepdim=True)) / x.std(
                        1, unbiased=False, keepdim=True
                    ).clamp_min(1e-6)
                    values = self.encoder(x).reshape(-1, 128).float()
                else:
                    values = torch.from_numpy(np.stack(values)).cuda()
                    if mode == "local" and key == "local":
                        # Standardized Z_last and p_local are local. The remaining
                        # columns contain the flow-adjusted score, flow and disagreement.
                        values[:, 65:] = 0
                soft[list(loc)] = self.project(
                    key, values, torch.as_tensor(pos, device="cuda"), self.rms
                )
            where = ids < 0
            em[where] = soft[-1 - ids[where]]
        return (
            em,
            torch.as_tensor(masks, device="cuda"),
            torch.as_tensor(labels, device="cuda"),
        )

    def loss(self, rows):
        em, mask, labels = self.prepare(rows, "real", None, True)
        h = self.decoder(
            inputs_embeds=em, attention_mask=mask, use_cache=False
        ).last_hidden_state
        target = labels[:, 1:]
        keep = target != -100
        per_token = F.cross_entropy(
            self.head(h[:, :-1][keep]).float(), target[keep], reduction="none"
        )
        suffix = torch.tensor(
            [self.answer_span(response(r))[1] for r in rows], device=h.device
        )
        positions = torch.arange(target.shape[1], device=h.device)
        last = (mask.sum(1) - 1)[:, None]
        answer = (positions >= last - suffix[:, None]) & (positions < last)
        weights = 1 + 2 * answer[keep].float()
        return (per_token * weights).sum(), sum(self.loss_mass(r) for r in rows)

    @torch.inference_mode()
    def generate(self, rows, mode="real", donors=None, max_tokens=168):
        em, mask, _ = self.prepare(rows, mode, donors, False)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        out = self.decoder(
            inputs_embeds=em, attention_mask=mask, position_ids=pos, use_cache=True
        )
        done = torch.zeros(len(rows), device="cuda", dtype=torch.bool)
        tokens = []
        for _ in range(max_tokens):
            tok = self.head(out.last_hidden_state[:, -1]).argmax(-1)
            tokens.append(tok)
            done |= tok == self.eos
            if done.all():
                break
            mask = F.pad(mask, (0, 1), value=1)
            out = self.decoder(
                input_ids=tok[:, None],
                attention_mask=mask,
                position_ids=mask.sum(1, keepdim=True) - 1,
                past_key_values=out.past_key_values,
                use_cache=True,
            )
        answers = []
        for seq in torch.stack(tokens, 1).tolist():
            if self.eos in seq:
                seq = seq[: seq.index(self.eos)]
            answers.append(
                self.tokenizer.decode(seq, skip_special_tokens=False).strip()
            )
        return answers

    def learned(self):
        return {
            k: v
            for k, v in self.named_parameters()
            if k.startswith("project.") or k.endswith((".up.weight", ".down.weight"))
        }

    def restore(self, ck):
        if ck["variant"] != self.variant or ck["revision"] != REV:
            raise ValueError("Wrong model contract")
        if self.variant == "drift" and (
            ck.get("recipe") != "v2.1-weighted-answer"
            or ck["data"]["stats"] != self.source.stats_digest
        ):
            raise ValueError(
                "Checkpoint and history-v3 feature normalization do not match"
            )
        params = self.learned()
        if params.keys() != ck["parameters"].keys():
            raise ValueError("Wrong trainable modules")
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(ck["parameters"][k])
        self.stage(ck["stage"], False)


def parse(text):
    match = re.fullmatch(
        r"<reasoning>\s*(.+?)\s*</reasoning>\s*<answer>\s*(\{[^<>]*\})\s*</answer>",
        text,
        re.S,
    )
    if not match:
        return None
    try:
        value = json.loads(match.group(2))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
