"""
Parse result.txt from tuning runs, pick best LR per dataset/pred_len.
Usage: python parse_tuning.py
"""
import re
from collections import defaultdict

def parse_results(filepath='result.txt'):
    results = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and 'mse:' not in line and line != '':
            # This is a setting line
            setting = line
            # Next non-empty line should have mse/mae
            i += 1
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            if i < len(lines) and 'mse:' in lines[i]:
                metric_line = lines[i].strip()
                mse_match = re.search(r'mse:([\d.]+)', metric_line)
                mae_match = re.search(r'mae:([\d.]+)', metric_line)
                if mse_match and mae_match:
                    mse = float(mse_match.group(1))
                    mae = float(mae_match.group(1))
                    results.append((setting, mse, mae))
        i += 1
    
    return results

def extract_info(setting):
    """Extract dataset, lr, pred_len from setting string."""
    # Format: ETTh1_lr0.0003_96_ema_GLPatch_ETTh1_ftM_sl96_ll48_pl96_Exp_0
    # or: weather_lr0.001_720_ema_GLPatch_custom_ftM_sl96_ll48_pl720_Exp_0
    
    parts = setting.split('_')
    
    # Find LR
    lr = None
    for p in parts:
        if p.startswith('lr'):
            lr = p[2:]
            break
    
    # Find pred_len from pl field
    pred_len = None
    for p in parts:
        if p.startswith('pl'):
            pred_len = int(p[2:])
            break
    
    # Find dataset from the tag prefix (before lr)
    dataset = None
    tag_parts = []
    for p in parts:
        if p.startswith('lr'):
            break
        tag_parts.append(p)
    dataset = '_'.join(tag_parts) if tag_parts else parts[0]
    
    # Normalize dataset names
    dataset_map = {
        'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2',
        'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2',
        'weather': 'weather', 'exchange': 'exchange',
        'ili': 'ili', 'solar': 'solar',
        'traffic': 'traffic', 'electricity': 'electricity'
    }
    for key in dataset_map:
        if key.lower() in dataset.lower():
            dataset = dataset_map[key]
            break
    
    return dataset, lr, pred_len

# xPatch Table 13 baselines (seq_len=96)
xpatch_baselines = {
    ('ETTh1', 96): 0.376, ('ETTh1', 192): 0.417, ('ETTh1', 336): 0.449, ('ETTh1', 720): 0.470,
    ('ETTh2', 96): 0.233, ('ETTh2', 192): 0.291, ('ETTh2', 336): 0.344, ('ETTh2', 720): 0.407,
    ('ETTm1', 96): 0.311, ('ETTm1', 192): 0.348, ('ETTm1', 336): 0.388, ('ETTm1', 720): 0.461,
    ('ETTm2', 96): 0.164, ('ETTm2', 192): 0.230, ('ETTm2', 336): 0.292, ('ETTm2', 720): 0.381,
    ('weather', 96): 0.168, ('weather', 192): 0.214, ('weather', 336): 0.236, ('weather', 720): 0.309,
    ('exchange', 96): 0.082, ('exchange', 192): 0.177, ('exchange', 336): 0.349, ('exchange', 720): 0.891,
    ('ili', 24): 1.378, ('ili', 36): 1.315, ('ili', 48): 1.459, ('ili', 60): 1.616,
}

if __name__ == '__main__':
    results = parse_results()
    
    # Group by (dataset, pred_len) → list of (lr, mse, mae)
    grouped = defaultdict(list)
    for setting, mse, mae in results:
        dataset, lr, pred_len = extract_info(setting)
        if lr and pred_len:
            grouped[(dataset, pred_len)].append((lr, mse, mae))
    
    # Find best LR per (dataset, pred_len)
    print("=" * 90)
    print(f"{'Dataset':<12} {'PL':>4} {'Best LR':>10} {'Best MSE':>10} {'xPatch':>10} {'Diff':>10} {'Pct':>8}")
    print("=" * 90)
    
    best_config = {}
    for (dataset, pred_len), lr_results in sorted(grouped.items()):
        # Sort by MSE
        lr_results.sort(key=lambda x: x[1])
        best_lr, best_mse, best_mae = lr_results[0]
        
        xp = xpatch_baselines.get((dataset, pred_len))
        if xp:
            diff = best_mse - xp
            pct = (diff / xp) * 100
            marker = "✓" if diff < 0 else "✗"
            print(f"{dataset:<12} {pred_len:>4} {best_lr:>10} {best_mse:>10.4f} {xp:>10.3f} {diff:>+10.4f} {pct:>+7.2f}% {marker}")
        else:
            print(f"{dataset:<12} {pred_len:>4} {best_lr:>10} {best_mse:>10.4f}")
        
        best_config[(dataset, pred_len)] = (best_lr, best_mse, best_mae)
        
        # Show all LRs for comparison
        for lr, mse, mae in lr_results:
            flag = " <-- best" if lr == best_lr else ""
            print(f"{'':>29} lr={lr:>10} mse={mse:.4f}{flag}")
    
    # Summary
    print("\n" + "=" * 90)
    print("BEST CONFIG SUMMARY (for final run script):")
    print("=" * 90)
    
    wins = 0
    total = 0
    for (dataset, pred_len), (best_lr, best_mse, best_mae) in sorted(best_config.items()):
        xp = xpatch_baselines.get((dataset, pred_len))
        if xp:
            total += 1
            if best_mse < xp:
                wins += 1
    
    print(f"\nWins: {wins}/{total}")
    
    # Group best LR by dataset
    print("\nBest LR per dataset:")
    dataset_lrs = defaultdict(list)
    for (dataset, pred_len), (best_lr, _, _) in best_config.items():
        dataset_lrs[dataset].append((pred_len, best_lr))
    
    for dataset, pl_lrs in sorted(dataset_lrs.items()):
        lrs = [lr for _, lr in pl_lrs]
        # Most common LR
        from collections import Counter
        most_common = Counter(lrs).most_common(1)[0][0]
        print(f"  {dataset}: {most_common} (per-pl: {dict(sorted(pl_lrs))})")
