"""
FluxMind v0.82 - Evaluation Only
=================================
Loads best_model.pt and runs full evaluation on all 42 test DSLs.
"""

import torch
import json
import os
import numpy as np
from train_v082_scaled import (
    ScaledConfig, BitFluxMindScaled, DSLGenerator,
    evaluate_dsl, state_to_bits, bits_to_state, generate_examples, random_state
)


def run_eval(
    checkpoint_path: str = 'v082_results/best_model.pt',
    device: str = 'cuda',
    n_train_dsls: int = 161,
    n_test_per_family: int = 6,
    support_size: int = 64,
    n_samples: int = 200,
    save_dir: str = 'v082_results'
):
    print("=" * 70)
    print("FLUXMIND v0.82 - FULL EVALUATION")
    print("=" * 70)

    # Load checkpoint
    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', ScaledConfig())

    model = BitFluxMindScaled(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"Checkpoint train acc: {checkpoint.get('train_acc', '?')}")
    print(f"Checkpoint test acc (quick): {checkpoint.get('test_acc', '?')}")

    # Generate same DSLs
    print("\nGenerating DSLs (balanced)...")
    generator = DSLGenerator(seed=42)
    train_dsls, test_dsls = generator.generate_balanced(n_train_dsls, n_test_per_family)
    print(f"  Train: {len(train_dsls)}, Test: {len(test_dsls)}")

    rng = np.random.RandomState(123)  # Different seed for eval

    # Evaluate train DSLs (sample)
    print(f"\n--- Train DSLs (14 sampled, {n_samples} samples each) ---")
    train_accs = []
    for dsl in train_dsls[:14]:
        acc = evaluate_dsl(model, dsl, device, rng, n_samples=n_samples, support_size=support_size)
        train_accs.append(acc)
        print(f"  {dsl.name:20s} ({dsl.family:12s}): {acc*100:5.1f}%")
    print(f"  Mean train: {np.mean(train_accs)*100:.1f}%")

    # Evaluate ALL test DSLs
    print(f"\n--- Test DSLs (ALL {len(test_dsls)}, {n_samples} samples each) ---")
    test_results = {}
    family_results = {}

    for dsl in test_dsls:
        acc = evaluate_dsl(model, dsl, device, rng, n_samples=n_samples, support_size=support_size)
        test_results[dsl.name] = {'accuracy': acc, 'family': dsl.family}

        if dsl.family not in family_results:
            family_results[dsl.family] = []
        family_results[dsl.family].append(acc)

        print(f"  {dsl.name:20s} ({dsl.family:12s}): {acc*100:5.1f}%")

    # Summary by family
    print(f"\n--- Summary by Family (BALANCED, {n_test_per_family} per family) ---")
    for family in sorted(family_results.keys()):
        accs = family_results[family]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  {family:12s}: {mean_acc*100:5.1f}% +/- {std_acc*100:4.1f}% (n={len(accs)})")

    # Overall
    all_test_accs = [r['accuracy'] for r in test_results.values()]
    mean_test = np.mean(all_test_accs)
    mean_train = np.mean(train_accs)

    print(f"\n{'='*70}")
    print(f"  OVERALL TEST:  {mean_test*100:.1f}%")
    print(f"  OVERALL TRAIN: {mean_train*100:.1f}%")
    print(f"  Random baseline: 6.25%")
    print(f"  vs Random: {mean_test/0.0625:.1f}x")
    print(f"\n  v0.81: 78.6% (877K params)")
    print(f"  v0.82: {mean_test*100:.1f}% ({params:,} params)")
    print(f"  Change: +{(mean_test - 0.786)*100:.1f}%")
    print(f"{'='*70}")

    # Save
    results = {
        'version': '0.82',
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'params': params,
        'n_test_dsls': len(test_dsls),
        'n_samples_per_dsl': n_samples,
        'support_size': support_size,
        'mean_test_acc': float(mean_test),
        'mean_train_acc': float(mean_train),
        'per_dsl': {k: {'accuracy': float(v['accuracy']), 'family': v['family']}
                    for k, v in test_results.items()},
        'per_family': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                       for k, v in family_results.items()},
    }

    out_path = os.path.join(save_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="v082_results/best_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    run_eval(
        checkpoint_path=args.checkpoint,
        device=device,
        n_samples=args.samples
    )
