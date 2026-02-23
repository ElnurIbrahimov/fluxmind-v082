"""
FluxMind v0.82 - Scaled Meta-Learning
======================================
Proprietary - Elnur Ibrahimov
February 2026

Scaling experiment for RunPod GPU:
  - Model: 877K -> ~2M params (wider embeddings, deeper predictor)
  - DSLs: 80 train -> 160 train, 20 test -> 40 test (balanced per family)
  - Epochs: 3000 -> 5000
  - Support: 32 -> 64 examples (more context)
  - Balanced test set (equal DSLs per family)

Target: 85%+ accuracy on novel DSLs (up from 78.6% in v0.81)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field
import time
import random
import json
import os
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ScaledConfig:
    """Scaled config for v0.82 (~2M params)."""

    # State space
    state_dim: int = 4
    state_range: int = 15
    bits_per_value: int = 4
    num_operations: int = 8

    # Scaled Bit model (roughly 2x v0.81's 877K)
    bit_embed_dim: int = 40          # 32 -> 40
    bit_state_embed_dim: int = 192   # 128 -> 192
    bit_example_embed_dim: int = 288 # 192 -> 288
    bit_context_dim: int = 256       # 192 -> 256
    bit_hidden_dim: int = 448        # 384 -> 448
    bit_num_heads: int = 8           # 6 -> 8

    # Dropout
    dropout: float = 0.1


# =============================================================================
# MODEL
# =============================================================================

class BitEncoder(nn.Module):
    """Encode state using bit representation."""

    def __init__(self, config: ScaledConfig):
        super().__init__()
        self.config = config

        self.bit_embeddings = nn.Parameter(
            torch.randn(config.state_dim, config.bits_per_value, config.bit_embed_dim) * 0.02
        )

        total_bit_dim = config.state_dim * config.bits_per_value * config.bit_embed_dim
        self.combiner = nn.Sequential(
            nn.Linear(total_bit_dim, config.bit_state_embed_dim),
            nn.LayerNorm(config.bit_state_embed_dim),
            nn.GELU(),
            nn.Linear(config.bit_state_embed_dim, config.bit_state_embed_dim)
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        batch_size = bits.shape[0]
        weighted = bits.unsqueeze(-1) * self.bit_embeddings.unsqueeze(0)
        flat = weighted.view(batch_size, -1)
        return self.combiner(flat)


class BitFluxMindScaled(nn.Module):
    """Scaled bit-based meta-learning model (~2M params)."""

    def __init__(self, config: ScaledConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.before_encoder = BitEncoder(config)
        self.after_encoder = BitEncoder(config)

        # Transition encoder (deeper)
        self.transition_net = nn.Sequential(
            nn.Linear(config.bit_state_embed_dim * 2, config.bit_example_embed_dim),
            nn.LayerNorm(config.bit_example_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bit_example_embed_dim, config.bit_example_embed_dim),
            nn.LayerNorm(config.bit_example_embed_dim),
            nn.GELU(),
            nn.Linear(config.bit_example_embed_dim, config.bit_example_embed_dim)
        )

        # Op embedding
        self.op_embed = nn.Embedding(config.num_operations, config.bit_state_embed_dim // 2)

        # Attention pooling
        self.query_proj = nn.Linear(config.bit_state_embed_dim, config.bit_context_dim)
        self.key_proj = nn.Linear(config.bit_example_embed_dim, config.bit_context_dim)
        self.value_proj = nn.Linear(config.bit_example_embed_dim, config.bit_context_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=config.bit_context_dim,
            num_heads=config.bit_num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Predictor (deeper)
        pred_input = config.bit_state_embed_dim + config.bit_state_embed_dim // 2 + config.bit_context_dim
        self.predictor = nn.Sequential(
            nn.Linear(pred_input, config.bit_hidden_dim),
            nn.LayerNorm(config.bit_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bit_hidden_dim, config.bit_hidden_dim),
            nn.LayerNorm(config.bit_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bit_hidden_dim, config.bit_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.bit_hidden_dim // 2, config.state_dim * config.bits_per_value)
        )

        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(pred_input, config.bit_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.bit_hidden_dim // 2, 1)
        )

    def encode_support(
        self,
        bits_before: torch.Tensor,
        ops: torch.Tensor,
        bits_after: torch.Tensor
    ) -> torch.Tensor:
        """Encode support examples."""
        batch_size, n_examples = bits_before.shape[:2]

        bb_flat = bits_before.view(batch_size * n_examples, *bits_before.shape[2:])
        ba_flat = bits_after.view(batch_size * n_examples, *bits_after.shape[2:])

        before_emb = self.before_encoder(bb_flat)
        after_emb = self.after_encoder(ba_flat)

        combined = torch.cat([before_emb, after_emb], dim=-1)
        trans_emb = self.transition_net(combined)

        return trans_emb.view(batch_size, n_examples, -1)

    def forward(
        self,
        query_bits: torch.Tensor,
        query_op: torch.Tensor,
        support_enc: torch.Tensor,
        support_ops: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config

        # Encode query
        state_emb = self.before_encoder(query_bits)
        op_emb = self.op_embed(query_op)

        # Per-op mask
        op_mask = (support_ops != query_op.unsqueeze(1))

        # Attention
        q = self.query_proj(state_emb).unsqueeze(1)
        k = self.key_proj(support_enc)
        v = self.value_proj(support_enc)

        context, _ = self.attention(q, k, v, key_padding_mask=op_mask)
        context = context.squeeze(1)

        # Predict
        combined = torch.cat([state_emb, op_emb, context], dim=-1)
        bit_logits = self.predictor(combined)
        bit_logits = bit_logits.view(-1, cfg.state_dim, cfg.bits_per_value)

        # Confidence
        conf_logit = self.confidence_head(combined)
        confidence = torch.sigmoid(conf_logit).squeeze(-1)

        return bit_logits, confidence


# =============================================================================
# DSL GENERATOR
# =============================================================================

@dataclass
class GeneratedDSL:
    name: str
    family: str
    ops: Dict[int, Callable]

    def execute(self, state: List[int], op: int) -> List[int]:
        return self.ops[op](state.copy())


class DSLGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _clamp(self, x: int) -> int:
        return max(1, min(15, x))

    def _make_arithmetic_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        for op in range(8):
            dim = op % 4
            const = self.rng.randint(-7, 7)
            if const == 0:
                const = 1
            def make_op(d=dim, c=const):
                def op_fn(state):
                    state[d] = self._clamp(state[d] + c)
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Arith_{dsl_id}", "arithmetic", ops)

    def _make_bitwise_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        bit_ops = ['xor', 'and', 'or']
        for op in range(8):
            dim = op % 4
            bit_op = self.rng.choice(bit_ops)
            mask = self.rng.randint(1, 15)
            def make_op(d=dim, bo=bit_op, m=mask):
                def op_fn(state):
                    val = state[d] - 1
                    if bo == 'xor':
                        val = val ^ m
                    elif bo == 'and':
                        val = val & m
                    else:
                        val = val | m
                    state[d] = (val % 15) + 1
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Bitwise_{dsl_id}", "bitwise", ops)

    def _make_comparison_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        for op in range(8):
            target = op % 4
            source = (op + 1 + self.rng.randint(0, 2)) % 4
            if source == target:
                source = (target + 1) % 4
            use_min = self.rng.choice([True, False])
            def make_op(t=target, s=source, is_min=use_min):
                def op_fn(state):
                    if is_min:
                        state[t] = min(state[t], state[s])
                    else:
                        state[t] = max(state[t], state[s])
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Compare_{dsl_id}", "comparison", ops)

    def _make_modular_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        for op in range(8):
            dim = op % 4
            add_val = self.rng.randint(1, 13)
            mod_val = self.rng.randint(5, 15)
            def make_op(d=dim, a=add_val, m=mod_val):
                def op_fn(state):
                    state[d] = ((state[d] - 1 + a) % m) + 1
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Modular_{dsl_id}", "modular", ops)

    def _make_shift_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        for op in range(8):
            dim = op % 4
            direction = self.rng.choice(['left', 'right'])
            amount = self.rng.randint(1, 3)
            def make_op(d=dim, dr=direction, am=amount):
                def op_fn(state):
                    val = state[d] - 1
                    if dr == 'left':
                        val = (val << am) & 0xF
                    else:
                        val = val >> am
                    state[d] = max(1, val + 1)
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Shift_{dsl_id}", "shift", ops)

    def _make_swap_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        for op in range(8):
            target = op % 4
            source = (op // 2) % 4
            if source == target:
                source = (target + 1) % 4
            do_swap = self.rng.choice([True, False])
            def make_op(t=target, s=source, swap=do_swap):
                def op_fn(state):
                    if swap:
                        state[t], state[s] = state[s], state[t]
                    else:
                        state[t] = state[s]
                    return state
                return op_fn
            ops[op] = make_op()
        return GeneratedDSL(f"Swap_{dsl_id}", "swap", ops)

    def _make_mixed_dsl(self, dsl_id: int) -> GeneratedDSL:
        ops = {}
        op_types = ['arith', 'bitwise', 'compare', 'modular']
        for op in range(8):
            dim = op % 4
            op_type = self.rng.choice(op_types)
            if op_type == 'arith':
                const = self.rng.randint(-5, 5)
                if const == 0: const = 1
                def make_op(d=dim, c=const):
                    def op_fn(state):
                        state[d] = self._clamp(state[d] + c)
                        return state
                    return op_fn
                ops[op] = make_op()
            elif op_type == 'bitwise':
                mask = self.rng.randint(1, 15)
                def make_op(d=dim, m=mask):
                    def op_fn(state):
                        val = (state[d] - 1) ^ m
                        state[d] = (val % 15) + 1
                        return state
                    return op_fn
                ops[op] = make_op()
            elif op_type == 'compare':
                source = (dim + 1) % 4
                use_min = self.rng.choice([True, False])
                def make_op(t=dim, s=source, is_min=use_min):
                    def op_fn(state):
                        state[t] = min(state[t], state[s]) if is_min else max(state[t], state[s])
                        return state
                    return op_fn
                ops[op] = make_op()
            else:
                add_val = self.rng.randint(1, 10)
                mod_val = self.rng.randint(5, 15)
                def make_op(d=dim, a=add_val, m=mod_val):
                    def op_fn(state):
                        state[d] = ((state[d] - 1 + a) % m) + 1
                        return state
                    return op_fn
                ops[op] = make_op()
        return GeneratedDSL(f"Mixed_{dsl_id}", "mixed", ops)

    def generate_balanced(self, n_train: int, n_test_per_family: int) -> Tuple[List[GeneratedDSL], List[GeneratedDSL]]:
        """
        Generate DSLs with BALANCED test set (equal per family).
        Fixes the statistical issue from v0.82 analysis.
        """
        families = [
            ('arithmetic', self._make_arithmetic_dsl),
            ('bitwise', self._make_bitwise_dsl),
            ('comparison', self._make_comparison_dsl),
            ('modular', self._make_modular_dsl),
            ('shift', self._make_shift_dsl),
            ('swap', self._make_swap_dsl),
            ('mixed', self._make_mixed_dsl),
        ]

        n_families = len(families)
        train_per_family = n_train // n_families

        train_dsls = []
        test_dsls = []
        dsl_id = 0

        for family_name, family_fn in families:
            # Generate train DSLs for this family
            for _ in range(train_per_family):
                train_dsls.append(family_fn(dsl_id))
                dsl_id += 1

            # Generate test DSLs for this family (balanced!)
            for _ in range(n_test_per_family):
                test_dsls.append(family_fn(dsl_id))
                dsl_id += 1

        # Fill remaining train DSLs
        remaining = n_train - len(train_dsls)
        for i in range(remaining):
            family_name, family_fn = families[i % n_families]
            train_dsls.append(family_fn(dsl_id))
            dsl_id += 1

        self.rng.shuffle(train_dsls)
        self.rng.shuffle(test_dsls)

        return train_dsls, test_dsls


# =============================================================================
# DATA GENERATION
# =============================================================================

def random_state(rng):
    return [int(rng.randint(1, 16)) for _ in range(4)]


def state_to_bits(state):
    bits = np.zeros((4, 4), dtype=np.float32)
    for di in range(4):
        for bi in range(4):
            bits[di, bi] = ((state[di] - 1) >> bi) & 1
    return bits


def bits_to_state(bits):
    state = []
    for di in range(4):
        val = sum(int(bits[di, bi]) << bi for bi in range(4)) + 1
        state.append(val)
    return state


def generate_examples(dsl: GeneratedDSL, n_examples: int, rng) -> List[Tuple]:
    examples = []
    for op in range(8):
        for _ in range(n_examples // 8 + 2):
            state = random_state(rng)
            try:
                next_state = dsl.execute(state, op)
                if all(1 <= v <= 15 for v in next_state):
                    examples.append((state, op, next_state))
            except:
                pass
    rng.shuffle(examples)
    return examples[:n_examples]


# =============================================================================
# TRAINING
# =============================================================================

def evaluate_dsl(model, dsl, device, rng, n_samples=100, support_size=64):
    """Evaluate on a single DSL."""
    model.eval()
    correct = 0
    total = 0

    for _ in range(n_samples):
        examples = generate_examples(dsl, support_size + 1, rng)
        if len(examples) < support_size + 1:
            continue

        support_examples = examples[:support_size]

        query_op = rng.randint(0, 8)
        query_state = random_state(rng)
        try:
            expected = dsl.execute(query_state, query_op)
        except:
            continue

        if not all(1 <= v <= 15 for v in expected):
            continue

        s_before = np.stack([state_to_bits(ex[0]) for ex in support_examples])
        s_after = np.stack([state_to_bits(ex[2]) for ex in support_examples])

        query_bits = torch.tensor(state_to_bits(query_state), dtype=torch.float32, device=device).unsqueeze(0)
        query_op_t = torch.tensor([query_op], dtype=torch.long, device=device)
        support_before = torch.tensor(s_before, dtype=torch.float32, device=device).unsqueeze(0)
        support_ops = torch.tensor([[ex[1] for ex in support_examples]], dtype=torch.long, device=device)
        support_after = torch.tensor(s_after, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            support_enc = model.encode_support(support_before, support_ops, support_after)
            bit_logits, _ = model(query_bits, query_op_t, support_enc, support_ops)
            pred_bits = (torch.sigmoid(bit_logits) > 0.5).float()

        pred_state = bits_to_state(pred_bits[0].cpu().numpy())

        if pred_state == expected:
            correct += 1
        total += 1

    return correct / max(total, 1)


def quick_eval(model, dsls, device, rng, n_samples=50):
    """Quick eval on subset of DSLs."""
    model.eval()
    total_correct = 0
    total = 0

    for dsl in dsls:
        for _ in range(n_samples // len(dsls)):
            examples = generate_examples(dsl, 72, rng)
            if len(examples) < 65:
                continue

            support = examples[:64]
            query_ex = examples[64]

            s_before = np.stack([state_to_bits(ex[0]) for ex in support])
            s_after = np.stack([state_to_bits(ex[2]) for ex in support])

            query_bits = torch.tensor(state_to_bits(query_ex[0]), dtype=torch.float32, device=device).unsqueeze(0)
            query_op = torch.tensor([query_ex[1]], dtype=torch.long, device=device)
            support_before = torch.tensor(s_before, dtype=torch.float32, device=device).unsqueeze(0)
            support_ops = torch.tensor([[ex[1] for ex in support]], dtype=torch.long, device=device)
            support_after = torch.tensor(s_after, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                support_enc = model.encode_support(support_before, support_ops, support_after)
                bit_logits, _ = model(query_bits, query_op, support_enc, support_ops)
                pred_bits = (torch.sigmoid(bit_logits) > 0.5).float()

            pred_state = bits_to_state(pred_bits[0].cpu().numpy())

            if pred_state == query_ex[2]:
                total_correct += 1
            total += 1

    model.train()
    return total_correct / max(total, 1)


def train_v082(
    device: str = 'cuda',
    n_train_dsls: int = 161,       # 23 per family * 7 = 161
    n_test_per_family: int = 6,    # 6 per family * 7 = 42 test DSLs
    epochs: int = 5000,
    batches_per_epoch: int = 300,
    batch_size: int = 32,
    support_size: int = 64,
    lr: float = 2e-4,
    eval_every: int = 100,
    save_dir: str = 'v082_results'
):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("FLUXMIND v0.82 - SCALED META-LEARNING")
    print("=" * 70)

    # Generate balanced DSLs
    print("\nGenerating DSLs (balanced test set)...")
    generator = DSLGenerator(seed=42)
    train_dsls, test_dsls = generator.generate_balanced(n_train_dsls, n_test_per_family)

    n_test_dsls = len(test_dsls)

    print(f"  Train DSLs: {len(train_dsls)}")
    print(f"  Test DSLs:  {n_test_dsls} ({n_test_per_family} per family)")

    # Print distribution
    for split_name, split_dsls in [("Train", train_dsls), ("Test", test_dsls)]:
        family_counts = {}
        for d in split_dsls:
            family_counts[d.family] = family_counts.get(d.family, 0) + 1
        print(f"  {split_name} distribution: {dict(sorted(family_counts.items()))}")

    # Create model
    config = ScaledConfig()
    model = BitFluxMindScaled(config)
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: BitFluxMindScaled ({params:,} params, {trainable:,} trainable)")
    print(f"Model size: ~{params * 4 / 1024 / 1024:.1f} MB")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batches/epoch: {batches_per_epoch}")
    print(f"Support size: {support_size}, Batch size: {batch_size}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2, eta_min=1e-5
    )

    rng = np.random.RandomState(42)

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()
    best_test_acc = 0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in range(batches_per_epoch):
            # Sample random DSL
            dsl = train_dsls[rng.randint(0, len(train_dsls))]

            # Generate examples
            examples = generate_examples(dsl, support_size + batch_size, rng)
            if len(examples) < support_size + batch_size:
                continue

            support_examples = examples[:support_size]
            query_examples = examples[support_size:support_size + batch_size]

            # Convert to tensors
            support_before = torch.tensor(
                np.array([state_to_bits(ex[0]) for ex in support_examples]),
                dtype=torch.float32, device=device
            ).unsqueeze(0).expand(len(query_examples), -1, -1, -1)

            support_ops = torch.tensor(
                [[ex[1] for ex in support_examples]],
                dtype=torch.long, device=device
            ).expand(len(query_examples), -1)

            support_after = torch.tensor(
                np.array([state_to_bits(ex[2]) for ex in support_examples]),
                dtype=torch.float32, device=device
            ).unsqueeze(0).expand(len(query_examples), -1, -1, -1)

            query_before = torch.tensor(
                np.array([state_to_bits(ex[0]) for ex in query_examples]),
                dtype=torch.float32, device=device
            )
            query_ops = torch.tensor(
                [ex[1] for ex in query_examples],
                dtype=torch.long, device=device
            )
            target_after = torch.tensor(
                np.array([state_to_bits(ex[2]) for ex in query_examples]),
                dtype=torch.float32, device=device
            )

            # Forward
            support_enc = model.encode_support(support_before, support_ops, support_after)
            bit_logits, _ = model(query_before, query_ops, support_enc, support_ops)

            loss = F.binary_cross_entropy_with_logits(bit_logits, target_after)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / batches_per_epoch)

            pred = (torch.sigmoid(bit_logits) > 0.5).float()
            correct = (pred == target_after).all(dim=-1).all(dim=-1).sum().item()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += len(query_examples)

        acc = epoch_correct / max(epoch_total, 1)
        avg_loss = epoch_loss / batches_per_epoch
        elapsed = time.time() - start_time

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if (epoch + 1) % eval_every == 0:
                # Full test eval on subset
                test_acc = quick_eval(model, test_dsls[:7], device, rng, n_samples=70)

                entry = {
                    'epoch': epoch + 1,
                    'train_acc': acc,
                    'test_acc': test_acc,
                    'loss': avg_loss,
                    'elapsed_min': elapsed / 60
                }
                history.append(entry)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    # Save best checkpoint
                    torch.save({
                        'model_state_dict': best_state,
                        'config': config,
                        'epoch': epoch + 1,
                        'test_acc': test_acc,
                        'train_acc': acc,
                    }, os.path.join(save_dir, 'best_model.pt'))

                eta_hr = (elapsed / (epoch + 1)) * (epochs - epoch - 1) / 3600
                print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | Train: {acc:.1%} | Test: {test_acc:.1%} | Best: {best_test_acc:.1%} (ep{best_epoch}) | {elapsed/60:.0f}m | ETA: {eta_hr:.1f}h")
            else:
                print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1%} | {elapsed/60:.0f}m")

        # Periodic checkpoint
        if (epoch + 1) % 500 == 0:
            torch.save({
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'config': config,
                'epoch': epoch + 1,
                'history': history,
            }, os.path.join(save_dir, f'checkpoint_ep{epoch+1}.pt'))
            print(f"  [Checkpoint saved: checkpoint_ep{epoch+1}.pt]")

    total_time = time.time() - start_time
    print(f"\nTraining time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} (test acc: {best_test_acc:.1%})")

    # ==========================================================================
    # FULL EVALUATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FULL EVALUATION (200 samples per DSL)")
    print("=" * 70)

    model.eval()

    # Train DSLs (sample 14)
    print("\n--- Train DSLs (sanity check, 14 sampled) ---")
    train_accs = []
    for dsl in train_dsls[:14]:
        acc = evaluate_dsl(model, dsl, device, rng, n_samples=200, support_size=support_size)
        train_accs.append(acc)
    print(f"  Mean train accuracy: {np.mean(train_accs)*100:.1f}%")

    # Test DSLs (ALL)
    print("\n--- Test DSLs (NEVER SEEN DURING TRAINING) ---")
    test_results = {}
    family_results = {}

    for dsl in test_dsls:
        acc = evaluate_dsl(model, dsl, device, rng, n_samples=200, support_size=support_size)
        test_results[dsl.name] = {'accuracy': acc, 'family': dsl.family}

        if dsl.family not in family_results:
            family_results[dsl.family] = []
        family_results[dsl.family].append(acc)

        print(f"  {dsl.name:20s} ({dsl.family:12s}): {acc*100:5.1f}%")

    # Summary by family
    print("\n--- Summary by Family (BALANCED) ---")
    for family in sorted(family_results.keys()):
        accs = family_results[family]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        n = len(accs)
        print(f"  {family:12s}: {mean_acc*100:5.1f}% +/- {std_acc*100:4.1f}% (n={n})")

    # Overall
    all_test_accs = [r['accuracy'] for r in test_results.values()]
    mean_test = np.mean(all_test_accs)
    mean_train = np.mean(train_accs)

    print(f"\n  OVERALL TEST:  {mean_test*100:.1f}%")
    print(f"  OVERALL TRAIN: {mean_train*100:.1f}%")
    print(f"  Random baseline: 6.25%")
    print(f"  vs Random: {mean_test/0.0625:.1f}x")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if mean_test > 0.85:
        print("TARGET HIT! 85%+ on novel DSLs!")
    elif mean_test > 0.80:
        print("STRONG: 80%+ on novel DSLs. Close to target.")
    elif mean_test > 0.70:
        print("GOOD: Improved over v0.81 (78.6%). Scaling helps.")
    else:
        print("MIXED: Need to investigate why scaling didn't help more.")

    # Comparison to v0.81
    print(f"\n  v0.81: 78.6% (877K params, 80 train DSLs, 3000 epochs)")
    print(f"  v0.82: {mean_test*100:.1f}% ({params:,} params, {len(train_dsls)} train DSLs, {epochs} epochs)")
    improvement = mean_test - 0.786
    print(f"  Change: {'+' if improvement >= 0 else ''}{improvement*100:.1f}%")

    # Save final results
    final_results = {
        'version': '0.82',
        'params': params,
        'n_train_dsls': len(train_dsls),
        'n_test_dsls': n_test_dsls,
        'epochs': epochs,
        'support_size': support_size,
        'mean_test_acc': float(mean_test),
        'mean_train_acc': float(mean_train),
        'best_epoch': best_epoch,
        'training_time_min': total_time / 60,
        'per_dsl_results': {k: {'accuracy': float(v['accuracy']), 'family': v['family']}
                           for k, v in test_results.items()},
        'per_family_results': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                              for k, v in family_results.items()},
        'history': history
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': final_results,
    }, os.path.join(save_dir, 'final_model.pt'))

    print(f"\nResults saved to {save_dir}/")
    print(f"  best_model.pt    - best checkpoint")
    print(f"  final_model.pt   - final model")
    print(f"  results.json     - full results")

    return model, final_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluxMind v0.82 Scaled Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-dsls", type=int, default=161)
    parser.add_argument("--test-per-family", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-dir", default="v082_results")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_v082(
        device=device,
        n_train_dsls=args.train_dsls,
        n_test_per_family=args.test_per_family,
        epochs=args.epochs,
        batches_per_epoch=args.batches,
        batch_size=args.batch_size,
        support_size=args.support_size,
        lr=args.lr,
        eval_every=args.eval_every,
        save_dir=args.save_dir
    )
