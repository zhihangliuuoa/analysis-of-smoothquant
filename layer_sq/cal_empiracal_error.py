"""
Empirical Error Analysis

python test_error_analysis_simple.py \
    --pile-path /path/to/val.jsonl.zst \
    --model-name meta-llama/Llama-3.2-1B
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def load_pile_dataset(pile_path, num_samples=512):
    import zstandard as zstd
    
    print(f"Loading Pile dataset from {pile_path}...")
    
    if not os.path.exists(pile_path):
        raise FileNotFoundError(
            f"Pile dataset not found at {pile_path}\n"
            f"Download: https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
    
    texts = []
    
    with open(pile_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            
            for i, line in enumerate(text_stream.split('\n')):
                if i >= num_samples:
                    break
                
                if line.strip():
                    try:
                        data = json.loads(line)
                        texts.append(data['text'])
                    except json.JSONDecodeError:
                        continue
    
    print(f"Loaded {len(texts)} samples")
    return texts


def compute_act_scales_streaming(model, tokenizer, pile_texts, selected_layers, seq_len=512):
    device = next(model.parameters()).device
    act_max_dict = {}
    
    print(f"\nComputing activation scales (streaming)...")
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            
            # [batch, seq, channels] -> [channels]
            current_max = x.abs().max(dim=0)[0].max(dim=0)[0]
            
            if name not in act_max_dict:
                act_max_dict[name] = current_max.detach().cpu().float()
            else:
                act_max_dict[name] = torch.maximum(
                    act_max_dict[name],
                    current_max.detach().cpu().float()
                )
        return hook
    
    hooks = []

    for layer_idx in selected_layers:
        key = f"model.layers.{layer_idx}.self_attn.q_proj"
        for name, module in model.named_modules():
            if name == key:
                hooks.append(module.register_forward_hook(make_hook(name)))
                break
    
    model.eval()
    
    for i in tqdm(range(len(pile_texts)), desc="Computing X_max"):
        input_ids = tokenizer(
            pile_texts[i],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding='max_length'
        ).input_ids.to(device)
        
        with torch.no_grad():
            model(input_ids)
        
        if i % 50 == 0:
            torch.cuda.empty_cache()
    
    for h in hooks:
        h.remove()
    
    print(f"✓ Computed scales for {len(act_max_dict)} layers")
    
    return act_max_dict


def compute_empirical_errors_streaming(model, tokenizer, pile_texts, act_scales,
                                       selected_layers, alphas, seq_len=512,
                                       max_samples=64):
    device = next(model.parameters()).device
    
    print(f"\nComputing empirical errors (max {max_samples} samples per layer)...")
    
    results = {}
    
    for layer_idx in tqdm(selected_layers, desc="Processing layers"):
        key = f"model.layers.{layer_idx}.self_attn.q_proj"
        
        if key not in act_scales:
            print(f"Warning: No act_scales for layer {layer_idx}")
            continue
        
        act_samples_batch = []
        
        def collect_hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            batch_size_curr, seq_len_curr, channels = x.shape
            x_flat = x.reshape(-1, channels)
            act_samples_batch.append(x_flat.detach().cpu().float())
        
        target_module = None
        for name, module in model.named_modules():
            if name == key:
                target_module = module
                break
        
        if target_module is None:
            print(f"Warning: Module not found for layer {layer_idx}")
            continue
        
        hook = target_module.register_forward_hook(collect_hook)
        
        model.eval()
        num_to_collect = min(max_samples, len(pile_texts))
        
        for i in range(num_to_collect):
            input_ids = tokenizer(
                pile_texts[i],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding='max_length'
            ).input_ids.to(device)
            
            with torch.no_grad():
                model(input_ids)
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        hook.remove()
        
        if len(act_samples_batch) == 0:
            print(f"Warning: No samples collected for layer {layer_idx}")
            continue
        
        act_samples = torch.cat(act_samples_batch, dim=0)
        
        weight = target_module.weight.data.cpu().float()
        
        act_max = act_scales[key]
        weight_max = weight.abs().max(dim=0)[0]
        
        errors = []
        
        for alpha in tqdm(alphas, desc=f"  Layer {layer_idx}", leave=False):
            # SmoothQuant
            s = (act_max ** alpha) / (weight_max ** (1 - alpha) + 1e-8)
            
            act_scaled = act_samples / s
            weight_scaled = weight * s.unsqueeze(0)
            
            # Fake INT8
            def fake_quant(x):
                qmax = 127.0
                scale = x.abs().max() / qmax + 1e-8
                x_q = torch.clamp((x / scale).round(), -qmax, qmax)
                return x_q * scale
            
            act_q = fake_quant(act_scaled)
            weight_q = fake_quant(weight_scaled)
            
            max_tokens = min(1000, act_samples.shape[0])
            if act_samples.shape[0] > max_tokens:
                indices = torch.randperm(act_samples.shape[0])[:max_tokens]
                act_sub = act_samples[indices]
                act_q_sub = act_q[indices]
            else:
                act_sub = act_samples
                act_q_sub = act_q
            
            y_fp = act_sub @ weight.T
            y_q = act_q_sub @ weight_q.T
            
            error = (y_q - y_fp).pow(2).mean().item()
            errors.append(error)
        
        results[layer_idx] = errors
        
        del act_samples, act_scaled, weight_scaled, act_q, weight_q
        torch.cuda.empty_cache()
    
    return results


def compute_theoretical_errors(model, act_scales, selected_layers, alphas):
    print(f"\nComputing theoretical errors...")
    
    results = {}
    
    for layer_idx in tqdm(selected_layers, desc="Computing theory"):
        key = f"model.layers.{layer_idx}.self_attn.q_proj"
        
        if key not in act_scales:
            continue
        
        weight = None
        for name, module in model.named_modules():
            if name == key:
                weight = module.weight.data.cpu().float()
                break
        
        if weight is None:
            continue
        
        act_max = act_scales[key]
        weight_max = weight.abs().max(dim=0)[0]
        act_var = (act_max ** 2) / 3.0
        weight_var = weight.var(dim=0)
        
        errors = []
        
        for alpha in alphas:
            s = (act_max ** alpha) / (weight_max ** (1 - alpha) + 1e-8)
            
            act_max_smoothed = act_max / s
            weight_max_smoothed = weight_max * s
            
            sigma_X_sq = (act_max_smoothed ** 2) / 12.0
            sigma_W_sq = (weight_max_smoothed ** 2) / 12.0
            
            error = (weight_var * sigma_X_sq).sum() + (act_var * sigma_W_sq).sum()
            errors.append(error.item())
        
        results[layer_idx] = errors
    
    return results


def plot_empirical_only(empirical_results, alphas, selected_layers, suffix=""):
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_layers)))
    
    for idx, layer_idx in enumerate(selected_layers):
        if layer_idx not in empirical_results:
            continue
        
        emp = np.array(empirical_results[layer_idx])
        emp_norm = emp / np.max(emp)
        
        opt_alpha = alphas[np.argmin(emp)]
        
        ax.plot(alphas, emp_norm, '-o', color=colors[idx],
               label=f'Layer {layer_idx} (α*={opt_alpha:.3f})',
               linewidth=2.5, markersize=6)
        
        ax.axvline(opt_alpha, color=colors[idx], linestyle=':', alpha=0.3)
    
    ax.set_xlabel('α', fontweight='bold')
    ax.set_ylabel('Normalized Error', fontweight='bold')

    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'empirical_errors{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def save_results_to_file(theoretical_results, empirical_results, alphas, suffix=""):
    data = {
        'theoretical': theoretical_results,
        'empirical': empirical_results,
        'alphas': alphas.tolist()
    }
    
    filename = f'error_results{suffix}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved: {filename}")
    
    txt_filename = f'error_results{suffix}.txt'
    
    with open(txt_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ERROR ANALYSIS RESULTS{suffix}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Alpha values: {alphas.tolist()}\n\n")
        
        for layer_idx in sorted(theoretical_results.keys()):
            f.write(f"\nLayer {layer_idx}:\n")
            f.write("-"*60 + "\n")
            
            if layer_idx in empirical_results:
                theo = theoretical_results[layer_idx]
                emp = empirical_results[layer_idx]
                
                opt_alpha_theo = alphas[np.argmin(theo)]
                opt_alpha_emp = alphas[np.argmin(emp)]
                
                f.write(f"Optimal α (theoretical): {opt_alpha_theo:.4f}\n")
                f.write(f"Optimal α (empirical):   {opt_alpha_emp:.4f}\n")
                f.write(f"Difference:              {abs(opt_alpha_theo - opt_alpha_emp):.4f}\n\n")
                
                f.write(f"Theoretical errors: {theo}\n\n")
                f.write(f"Empirical errors:   {emp}\n")
    
    print(f"✓ Saved: {txt_filename}")


def main():
    parser = argparse.ArgumentParser(description='Simplified empirical error analysis')
    parser.add_argument('--pile-path', type=str, default="../dataset/val.jsonl.zst")
    parser.add_argument('--model-name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--num-samples', type=int, default=256,
                       help='Number of Pile samples for computing X_max')
    parser.add_argument('--max-samples-per-layer', type=int, default=64,
                       help='Number of samples per layer for empirical error')
    parser.add_argument('--selected-layers', type=int, nargs='+', default=None)
    
    args = parser.parse_args()
    
    print("="*80)
    print("SIMPLIFIED EMPIRICAL ERROR ANALYSIS")
    print("="*80)
    print("\n setting：")
    print(f"  • Alpha range 1: [0.1, 0.9] step=0.1")
    print(f"  • Alpha range 2: [0.3, 0.6] step=0.02")
    print(f"  • Max samples per layer: {args.max_samples_per_layer}")
    print("="*80)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    num_layers = len([m for m in model.modules() 
                      if m.__class__.__name__ == 'LlamaDecoderLayer'])
    
    if args.selected_layers is None:
        if num_layers == 16:
            selected_layers = [0, 4, 8, 12, 15]
        elif num_layers == 32:
            selected_layers = [0, 8, 15, 23, 31]
        else:
            selected_layers = [0,9,19,29,39]
    else:
        selected_layers = args.selected_layers
    
    print(f"Selected layers: {selected_layers}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pile_texts = load_pile_dataset(args.pile_path, num_samples=args.num_samples)
    
    act_scales = compute_act_scales_streaming(
        model, tokenizer, pile_texts, selected_layers
    )

    
    alphas_fine = np.arange(0.3, 0.65, 0.02)
    
    theo_results_fine = compute_theoretical_errors(
        model, act_scales, selected_layers, alphas_fine
    )
    
    emp_results_fine = compute_empirical_errors_streaming(
        model, tokenizer, pile_texts, act_scales,
        selected_layers, alphas_fine,
        max_samples=args.max_samples_per_layer
    )
    
    plot_empirical_only(emp_results_fine, alphas_fine, 
                       selected_layers, suffix="_fine")
    
    save_results_to_file(theo_results_fine, emp_results_fine, 
                        alphas_fine, suffix="_fine")

if __name__ == "__main__":
    main()