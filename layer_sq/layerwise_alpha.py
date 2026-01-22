import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict

def compute_layer_error_corrected(act_max, weight_max, act_var, weight_var, alpha):
    """
    - X' = X / s, 其中 s = X_max^α / W_max^(1-α)
    - W' = W * s
    
    - Δ_X' ∝ max(X') = X_max / s = X_max^(1-α) * W_max^(1-α)
    - Δ_W' ∝ max(W') = W_max * s = X_max^α * W_max^(1-α)
    
    - σ_X'² = (Δ_X')² / 12
    - σ_W'² = (Δ_W')² / 12
    """
    s = (act_max ** alpha) / (weight_max ** (1 - alpha))
    
    act_max_smoothed = act_max / s  # = X_max^(1-α) * W_max^(1-α)
    weight_max_smoothed = weight_max * s  # = X_max^α * W_max^(1-α)
    
    sigma_X_sq = act_max_smoothed ** 2
    sigma_W_sq = weight_max_smoothed ** 2
    
    # E = Σ[v_W * σ_X² + v_X * σ_W²]
    error = (weight_var * sigma_X_sq).sum() + (act_var * sigma_W_sq).sum()
    
    return error.item()


def debug_first_layer_corrected(model, act_scales):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            key = name + ".self_attn.q_proj"
            if key not in act_scales:
                continue
            
            act_max = act_scales[key].cpu().float()
            weight = module.self_attn.q_proj.weight.data.cpu().float()
            weight_max = weight.abs().max(dim=0)[0]
            act_var = (act_max ** 2) / 3.0
            weight_var = weight.var(dim=0)
            
            print(f"\n=== Debug Layer (Corrected): {name} ===")
            print(f"act_max: mean={act_max.mean():.3f}, max={act_max.max():.3f}")
            print(f"weight_max: mean={weight_max.mean():.3f}, max={weight_max.max():.3f}")
            print(f"Outlier ratio: {(act_max > 3 * act_max.median()).float().mean():.2%}")
            
            print(f"\nError vs Alpha (Corrected):")
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            errors = []
            
            for alpha in alphas:
                error = compute_layer_error_corrected(act_max, weight_max, act_var, weight_var, alpha)
                errors.append(error)

                print(
                    f"α={alpha:.1f} | "
                    f"Proxy={error:.4e} | "
                )
            
            min_proxy_idx = np.argmin(errors)
            print(f"\nOptimal α using proxy= {alphas[min_proxy_idx]:.1f}")
            
            break
    
    return


def compute_layerwise_alphas_corrected(model, act_scales):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    debug_first_layer_corrected(model, act_scales)
    
    layer_alphas = {}
    
    print("\n=== Computing layer-wise optimal α (Corrected Theory) ===")
    
    for name, module in model.named_modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue
        
        key_attn = name + ".self_attn.q_proj"
        key_ffn = name + ".mlp.gate_proj"
        
        if key_attn not in act_scales:
            continue
        
        # Attention
        act_max_attn = act_scales[key_attn].cpu().float()
        weight_attn = module.self_attn.q_proj.weight.data.cpu().float()
        
        if weight_attn.shape[1] != act_max_attn.shape[0]:
            continue
        
        weight_max_attn = weight_attn.abs().max(dim=0)[0]
        act_var_attn = (act_max_attn ** 2) / 3.0
        weight_var_attn = weight_attn.var(dim=0)
        
        def objective_attn(alpha):
            return compute_layer_error_corrected(
                act_max_attn, weight_max_attn, act_var_attn, weight_var_attn, alpha
            )
        # act_samples_att = act_samples_dict[key_attn].cpu().float()
        # def objective_attn(alpha):
        #     return compute_layer_output_error(
        #             module.self_attn.q_proj,
        #             act_samples_att,
        #             act_max_attn,
        #             alpha
        #         )
        
        result_attn = minimize_scalar(objective_attn, bounds=(0.0, 1.0), method='bounded')
        alpha_attn = result_attn.x
        
        # FFN
        act_max_ffn = act_scales[key_ffn].cpu().float()
        weight_ffn = module.mlp.gate_proj.weight.data.cpu().float()
        
        if weight_ffn.shape[1] != act_max_ffn.shape[0]:
            continue
        
        weight_max_ffn = weight_ffn.abs().max(dim=0)[0]
        act_var_ffn = (act_max_ffn ** 2) / 3.0
        weight_var_ffn = weight_ffn.var(dim=0)
        
        def objective_ffn(alpha):
            return compute_layer_error_corrected(
                act_max_ffn, weight_max_ffn, act_var_ffn, weight_var_ffn, alpha
            )
        # act_samples_ffn = act_samples_dict[key_ffn].cpu().float()
        # def objective_ffn(alpha):
        #     return compute_layer_output_error(
        #             module.mlp.gate_proj,
        #             act_samples_ffn,
        #             act_max_ffn,
        #             alpha
        #         )
        
        result_ffn = minimize_scalar(objective_ffn, bounds=(0.0, 1.0), method='bounded')
        alpha_ffn = result_ffn.x
        
        layer_alphas[name] = (alpha_attn + alpha_ffn) / 2.0
        # layer_alphas[name + '.attn'] = alpha_attn
        # layer_alphas[name + '.ffn'] = alpha_ffn
        
        print(f"{name}: α_attn={alpha_attn:.3f}, α_ffn={alpha_ffn:.3f}, α_avg={layer_alphas[name]:.3f}")
    
    return layer_alphas
