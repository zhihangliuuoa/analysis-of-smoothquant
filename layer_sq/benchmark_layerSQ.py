#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_llama_like
from layerwise_alpha import compute_layerwise_alphas_corrected
from huggingface_hub import login
# login(token="") # need puting your huggingface token here


def load_evaluation_dataset(task_name):

    print(f"Loading dataset: {task_name}...")
    
    try:
        if task_name == "lambada":
            dataset = load_dataset("EleutherAI/lambada_openai", "en", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "piqa":
            dataset = load_dataset("ybisk/piqa", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "winogrande":
            dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "openbookqa":
            dataset = load_dataset("allenai/openbookqa", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "copa":
            dataset = load_dataset("super_glue", "copa", split="validation[0:1000]", trust_remote_code=True)
        elif task_name == "rte":
            dataset = load_dataset("super_glue", "rte", split="validation[0:1000]", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        print(f"Successfully loaded {task_name} ({len(dataset)} samples)")
        return dataset
        
    except Exception as e:
        print(f"Failed with new path: {e}")
        print(f"Trying legacy loading method...")

        try:
            if task_name == "lambada":
                dataset = load_dataset("lambada", split="validation[0:1000]")
            elif task_name == "hellaswag":
                dataset = load_dataset("hellaswag", split="validation[0:1000]")
            elif task_name == "piqa":
                dataset = load_dataset("piqa", split="validation[0:1000]")
            elif task_name == "winogrande":
                dataset = load_dataset("winogrande", "winogrande_xl", split="validation[0:1000]")
            elif task_name == "openbookqa":
                dataset = load_dataset("openbookqa", "main", split="validation[0:1000]")
            elif task_name == "copa":
                dataset = load_dataset("super_glue", "copa", split="validation[0:1000]")
            elif task_name == "rte":
                dataset = load_dataset("super_glue", "rte", split="validation[0:1000]")
            
            print(f"Successfully loaded {task_name} with legacy method ({len(dataset)} samples)")
            return dataset
            
        except Exception as e2:
            print(f"Failed to load {task_name}: {e2}")
            raise RuntimeError(f"Cannot load dataset {task_name}. Please check datasets library version or network connection.")


# ==================== Evaluators ====================

class LAMBADAEvaluator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = load_evaluation_dataset("lambada")
        
        def tokenize_function(examples):
            return self.tokenizer(examples['text'])
        
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for batch in tqdm(self.dataset, desc="LAMBADA", ncols=80):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            if input_ids.shape[1] < 2:
                continue
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += 1
            hit += (pred == label).item()
        return hit / total if total > 0 else 0


class MultipleChoiceEvaluator:
    def __init__(self, tokenizer, device, task_name):
        self.tokenizer = tokenizer
        self.device = device
        self.task_name = task_name
        self.dataset = load_evaluation_dataset(task_name)
    
    def _get_choices_and_label(self, sample):
        if self.task_name == "hellaswag":
            ctx = sample["ctx"]
            choices = [ctx + " " + ending for ending in sample["endings"]]
            label = int(sample["label"])
            return choices, label
        
        elif self.task_name == "piqa":
            goal = sample["goal"]
            choices = [goal + " " + sample["sol1"], goal + " " + sample["sol2"]]
            label = sample["label"]
            return choices, label
        
        elif self.task_name == "winogrande":
            sentence = sample["sentence"]
            option1 = sample["option1"]
            option2 = sample["option2"]
            choices = [sentence.replace("_", option1), sentence.replace("_", option2)]
            label = int(sample["answer"]) - 1
            return choices, label
        
        elif self.task_name == "openbookqa":
            question = sample["question_stem"]
            choice_texts = sample["choices"]["text"]
            choices = [question + " " + choice for choice in choice_texts]
            label = sample["choices"]["label"].index(sample["answerKey"])
            return choices, label
        
        elif self.task_name == "copa":
            premise = sample["premise"]
            question = sample["question"]
            connector = "because" if question == "cause" else "so"
            choices = [
                premise + " " + connector + " " + sample["choice1"],
                premise + " " + connector + " " + sample["choice2"]
            ]
            label = sample["label"]
            return choices, label
        
        elif self.task_name == "rte":
            premise = sample["premise"]
            hypothesis = sample["hypothesis"]
            choices = [
                premise + " Therefore, " + hypothesis,
                premise + " However, " + hypothesis
            ]
            label = sample["label"]
            return choices, label
    
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, correct = 0, 0
        
        for sample in tqdm(self.dataset, desc=self.task_name.upper(), ncols=80):
            try:
                choices, label = self._get_choices_and_label(sample)
                
                choice_losses = []
                for choice in choices:
                    tokens = self.tokenizer(choice, return_tensors="pt").input_ids.to(self.device)
                    
                    if tokens.shape[1] < 2:
                        choice_losses.append(float('inf'))
                        continue
                    
                    outputs = model(tokens)
                    logits = outputs.logits
                    
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = tokens[:, 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                    choice_losses.append(loss.item())
                
                pred = choice_losses.index(min(choice_losses))
                
                total += 1
                correct += (pred == label)
                
            except Exception as e:
                print(f"  ⚠ Error on sample: {e}")
                continue
        
        return correct / total if total > 0 else 0


# ==================== Layer-wise Smoothing ====================

def smooth_lm_layerwise(model, act_scales, layer_alphas):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from smoothquant.smooth import smooth_ln_fcs_llama_like
    
    for name, module in model.named_modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue
        if name not in layer_alphas:
            continue
        
        alpha = layer_alphas[name]
        
        attn_ln = module.input_layernorm
        qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        qkv_input_scales = act_scales[name + ".self_attn.q_proj"]
        smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
        
        ffn_ln = module.post_attention_layernorm
        fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        fcs_input_scales = act_scales[name + ".mlp.gate_proj"]
        smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)


# ==================== main evaluater ====================

def evaluate_model(model_name, act_scales_path, alpha=0.85, use_layerwise=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Method: {'Layer-wise SmoothQuant' if use_layerwise else f'Uniform SmoothQuant (α={alpha})'}")
    print(f"{'='*70}\n")
    
    # 1. Load model
    print("Step 1/6: Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Load activation scales
    print(f"Step 2/6: Loading activation scales...")
    act_scales = torch.load(act_scales_path)
    
    # 3. Apply SmoothQuant
    if use_layerwise:
        print("Step 3/6: Computing layer-wise optimal α...")
        layer_alphas = compute_layerwise_alphas_corrected(model, act_scales)
        alpha_mean = sum(layer_alphas.values()) / len(layer_alphas)
        print(f"  α: Mean={alpha_mean:.3f}, Min={min(layer_alphas.values()):.3f}, Max={max(layer_alphas.values()):.3f}")
        
        print("Step 4/6: Applying layer-wise smoothing...")
        smooth_lm_layerwise(model, act_scales, layer_alphas)
    else:
        print(f"Step 3/6: Applying uniform smoothing (α={alpha})...")
        smooth_lm(model, act_scales, alpha)
    
    # 4. Quantize
    print("Step 4/6: Quantizing to W8A8...")
    model = quantize_llama_like(model)
    
    # 5. Evaluate
    print("Step 5/6: Evaluating on all tasks...\n")
    
    results = {}
    
    # LAMBADA
    print(">>> Evaluating LAMBADA...")
    try:
        lambada_eval = LAMBADAEvaluator(tokenizer, device)
        results["lambada_openai"] = lambada_eval.evaluate(model) * 100
        print(f"    ✓ Accuracy: {results['lambada_openai']:.2f}%\n")
    except Exception as e:
        print(f"    ✗ Failed: {e}\n")
        results["lambada_openai"] = 0.0
    
    # Other tasks
    tasks = ["hellaswag", "winogrande", "openbookqa", "rte", "copa"]
    for task in tasks:
        print(f">>> Evaluating {task.upper()}...")
        try:
            evaluator = MultipleChoiceEvaluator(tokenizer, device, task)
            results[task] = evaluator.evaluate(model) * 100
            print(f"    ✓ Accuracy: {results[task]:.2f}%\n")
        except Exception as e:
            print(f"    ✗ Failed: {e}\n")
            results[task] = 0.0
    
    # 6. Summary
    print("Step 6/6: Results Summary")
    print(f"{'='*70}")
    print(f"{'Task':<20s} {'Accuracy':>10s}")
    print(f"{'-'*35}")
    for task, acc in results.items():
        print(f"{task:<20s} {acc:>9.2f}%")
    
    valid_results = [acc for acc in results.values() if acc > 0]
    if valid_results:
        avg_acc = sum(valid_results) / len(valid_results)
        print(f"{'-'*35}")
        print(f"{'Average':<20s} {avg_acc:>9.2f}%")
    print(f"{'='*70}\n")
    
    del model
    torch.cuda.empty_cache()
    
    return results


# ==================== main ====================

def main():
    configs = [
        # {
        #     "model_name": "meta-llama/Llama-3.2-1B",
        #     "act_scales_path": "../act_scales/llama-3.2-1b.pt",
        #     "alpha": 0.5,
        # },
        # {
        #     "model_name": "NousResearch/Meta-Llama-3-8B",
        #     "act_scales_path": "../act_scales/llama-3-8b.pt",
        #     "alpha": 0.5,
        # },
        # {
        #     "model_name": "meta-llama/Llama-2-7b-hf",
        #     "act_scales_path": "../act_scales/llama-2-7b.pt",
        #     "alpha": 0.5,
        # },
        {
            "model_name": "meta-llama/Llama-2-13b-hf",
            "act_scales_path": "../act_scales/llama-2-13b.pt",
            "alpha": 0.5,
        }
    ]
    
    all_results = {}
    
    for config in configs:
        model_name = config["model_name"]
        model_short = model_name.split('/')[-1]
        
        print("\n" + "#"*70)
        print(f"# {model_name}")
        print("#"*70)
        
        # Uniform
        print("\n" + ">"*70)
        print("> Uniform SmoothQuant (Baseline)")
        print(">"*70)
        uniform_results = evaluate_model(
            model_name=model_name,
            act_scales_path=config["act_scales_path"],
            alpha=config["alpha"],
            use_layerwise=False
        )
        
        # Layer-wise
        print("\n" + ">"*70)
        print("> Layer-wise SmoothQuant")
        print(">"*70)
        layerwise_results = evaluate_model(
            model_name=model_name,
            act_scales_path=config["act_scales_path"],
            use_layerwise=True
        )
        
        all_results[model_short] = {
            "uniform": uniform_results,
            "layerwise": layerwise_results
        }
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80 + "\n")
    
    all_tasks = ["lambada_openai", "hellaswag", "piqa", "winogrande", "openbookqa", "rte", "copa"]
    
    for model_name, results in all_results.items():
        print(f"\n{'─'*80}")
        print(f"Model: {model_name}")
        print(f"{'─'*80}")
        print(f"{'Task':<20s} {'Uniform':>12s} {'Layer-wise':>12s} {'Improvement':>12s}")
        print(f"{'-'*65}")
        
        uniform = results["uniform"]
        layerwise = results["layerwise"]
        
        for task in all_tasks:
            if task in uniform and task in layerwise:
                uni_acc = uniform[task]
                lw_acc = layerwise[task]
                diff = lw_acc - uni_acc
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                
                print(f"{task:<20s} {uni_acc:>11.2f}% {lw_acc:>11.2f}% {diff:>9.2f}% {symbol}")
        
        valid_uni = [v for v in uniform.values() if v > 0]
        valid_lw = [v for v in layerwise.values() if v > 0]
        
        if valid_uni and valid_lw:
            avg_uni = sum(valid_uni) / len(valid_uni)
            avg_lw = sum(valid_lw) / len(valid_lw)
            avg_diff = avg_lw - avg_uni
            symbol = "↑" if avg_diff > 0 else "↓" if avg_diff < 0 else "="
            
            print(f"{'-'*65}")
            print(f"{'Average':<20s} {avg_uni:>11.2f}% {avg_lw:>11.2f}% {avg_diff:>9.2f}% {symbol}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=None)
    args = parser.parse_args()
    
    if args.model_index is not None:
        configs = [
            {
                "model_name": "NousResearch/Meta-Llama-3-8B",
                "act_scales_path": "../act_scales/llama-3-8b.pt",
                "alpha": 0.5,
            },
            {
                "model_name": "meta-llama/Llama-3.2-1B",
                "act_scales_path": "../act_scales/llama-3.2-1b.pt",
                "alpha": 0.5,
            },
            {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "act_scales_path": "../act_scales/llama-2-7b.pt",
                "alpha": 0.5,
            }
        ]
        
        config = configs[args.model_index]
        
        print("\n>>> Uniform SmoothQuant")
        uniform = evaluate_model(
            config["model_name"], 
            config["act_scales_path"], 
            config["alpha"], 
            False
        )
        
        print("\n>>> Layer-wise SmoothQuant")
        layerwise = evaluate_model(
            config["model_name"], 
            config["act_scales_path"], 
            use_layerwise=True
        )
        
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        all_tasks = ["lambada_openai", "hellaswag", "piqa", "winogrande", "openbookqa", "rte", "copa"]
        for task in all_tasks:
            if task in uniform and task in layerwise:
                diff = layerwise[task] - uniform[task]
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"{task:<20s} {uniform[task]:>9.2f}% → {layerwise[task]:>9.2f}% ({diff:+.2f}%) {symbol}")
        print("="*70)
    else:
        main()