import argparse
import json
import time
import os
from pathlib import Path
from typing import Optional

import psutil
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM, pipeline as ort_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Install with: pip install openai")


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return api_key


def load_data(path):
    """Load JSON data (supports {'data': [...]} or top-level list)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict) and 'data' in payload and isinstance(payload['data'], list):
        return payload['data']
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON format in {path}")


def load_quantized_model(quantized_model_dir, batch_size=12):
    """Load quantized ONNX model as a pipeline."""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        quantized_model_dir = os.path.join(BASE_DIR, "..", "models", "quantized_models", "t5-small-finetuned-summarization-xsum")
        quantized_model_dir = os.path.abspath(quantized_model_dir)
        print("Loading quantized model from:", quantized_model_dir)
        model = ORTModelForSeq2SeqLM.from_pretrained(quantized_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
        summarizer = ort_pipeline(
            'summarization',
            model=model,
            tokenizer=tokenizer,
            device_map='auto',
            batch_size=batch_size
        )
        return summarizer, tokenizer
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        raise


def generate_with_local_model(summarizer, tokenizer, texts, do_sample=False):
    """Generate predictions using local quantized model."""
    proc = psutil.Process()
    
    predictions = []
    latencies = []
    mem_before = proc.memory_info().rss / (1024 ** 2)
    
    for idx, text in enumerate(texts):
        input_tokens = len(tokenizer.encode(text, truncation=True))
        
        start = time.perf_counter()
        try:
            params = {'do_sample': do_sample}
            pred_obj = summarizer(text, **params)
            pred = pred_obj[0].get('summary_text') if isinstance(pred_obj, list) and len(pred_obj) > 0 else str(pred_obj)
        except Exception as e:
            print(f"Error generating local prediction at index {idx}: {e}")
            pred = None
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)
        
        output_tokens = len(tokenizer.encode(pred, truncation=True)) if pred else 0
        total_tokens = input_tokens + output_tokens
        throughput = (total_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        
        predictions.append({
            'pred': pred,
            'latency_ms': latency_ms,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'throughput_tokens_per_sec': throughput
        })
    
    mem_after = proc.memory_info().rss / (1024 ** 2)
    mem_delta = mem_after - mem_before
    
    return predictions, latencies, mem_before, mem_after, mem_delta


def generate_with_gpt4_mini(client, texts, system_prompt=None):
    """Generate predictions using GPT-4o-mini API."""
    if system_prompt is None:
        system_prompt = "You are a professional summarizer. Summarize the given text concisely in 100-150 words."
    
    predictions = []
    latencies = []
    
    for text in texts:
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ],
                max_tokens=200,
                temperature=0.0,
            )
            pred = response.choices[0].message.content
            usage = response.usage
        except Exception as e:
            print(f"Error generating GPT-4o-mini prediction: {e}")
            pred = None
            usage = None
        
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)
        
        predictions.append({
            'pred': pred,
            'latency_ms': latency_ms,
            'input_tokens': usage.prompt_tokens if usage else 0,
            'output_tokens': usage.completion_tokens if usage else 0,
            'throughput_tokens_per_sec': (
                (usage.prompt_tokens + usage.completion_tokens) / (latency_ms / 1000.0)
                if usage and latency_ms > 0 else 0
            )
        })
    
    return predictions, latencies


def compute_rouge_metrics(predictions, gold_targets):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    
    for pred, gold in zip(predictions, gold_targets):
        if pred is None or not pred.strip():
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougel_scores.append(0.0)
            continue
        
        scores = scorer.score(gold, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougel_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': {'mean': np.mean(rouge1_scores), 'std': np.std(rouge1_scores), 'scores': rouge1_scores},
        'rouge2': {'mean': np.mean(rouge2_scores), 'std': np.std(rouge2_scores), 'scores': rouge2_scores},
        'rougeL': {'mean': np.mean(rougel_scores), 'std': np.std(rougel_scores), 'scores': rougel_scores}
    }


def compute_bertscore_metrics(predictions, gold_targets):
    """Compute BERTScore."""
    preds_clean = [p if p and p.strip() else "." for p in predictions]
    golds_clean = [g if g and g.strip() else "." for g in gold_targets]
    
    try:
        P, R, F1 = bert_score(preds_clean, golds_clean, lang='en', rescale_with_baseline=True, device='cpu')
        return {
            'precision': {'mean': float(P.mean()), 'std': float(P.std()), 'scores': P.tolist()},
            'recall': {'mean': float(R.mean()), 'std': float(R.std()), 'scores': R.tolist()},
            'f1': {'mean': float(F1.mean()), 'std': float(F1.std()), 'scores': F1.tolist()}
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare quantized ONNX model vs GPT-4o-mini.')
    parser.add_argument('--data', type=str, default='data/data.json', help='Path to data.json')
    parser.add_argument('--model_dir', type=str, default='../models/quantized_models/t5-small-finetuned-summarization-xsum/', help='Path to quantized model')
    parser.add_argument('--out', type=str, default='data/comparison_results.json', help='Output path for results')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    texts = [d.get('source') for d in data]
    gold_targets = [d.get('target') for d in data]
    print(f"Loaded {len(texts)} examples.")

    # ============ LOCAL MODEL ============
    print(f"\n{'='*80}")
    print("EVALUATING LOCAL QUANTIZED MODEL")
    print(f"{'='*80}")
    print(f"Loading model from {args.model_dir}...")
    summarizer, tokenizer = load_quantized_model(args.model_dir)

    print(f"Generating predictions on {len(texts)} examples...")
    local_preds_data, local_latencies, mem_before, mem_after, mem_delta = generate_with_local_model(
        summarizer, tokenizer, texts)
    
    local_preds = [p['pred'] for p in local_preds_data]

    print("Computing ROUGE metrics...")
    local_rouge = compute_rouge_metrics(local_preds, gold_targets)

    print("Computing BERTScore metrics...")
    local_bertscore = compute_bertscore_metrics(local_preds, gold_targets)

    local_throughputs = [p['throughput_tokens_per_sec'] for p in local_preds_data]

    local_results = {
        'model': 'Quantized ONNX (t5-small-finetuned-summarization-xsum)',
        'latency_mean_ms': float(np.mean(local_latencies)),
        'latency_std_ms': float(np.std(local_latencies)),
        'throughput_mean_tokens_per_sec': float(np.mean(local_throughputs)),
        'throughput_std_tokens_per_sec': float(np.std(local_throughputs)),
        'memory_before_mb': float(mem_before),
        'memory_after_mb': float(mem_after),
        'memory_delta_mb': float(mem_delta),
        'rouge': local_rouge,
        'bertscore': local_bertscore,
        'per_example': local_preds_data
    }

    # ============ GPT-4O-MINI ============
    print(f"\n{'='*80}")
    print("EVALUATING GPT-4o-mini")
    print(f"{'='*80}")
    
    if not OPENAI_AVAILABLE:
        print("ERROR: OpenAI package not installed. Install with: pip install openai")
        return

    try:
        api_key = load_env()
        client = OpenAI(api_key=api_key)
        
        print(f"Generating predictions on {len(texts)} examples...")
        gpt_preds_data, gpt_latencies = generate_with_gpt4_mini(client, texts)
        gpt_preds = [p['pred'] for p in gpt_preds_data]

        print("Computing ROUGE metrics...")
        gpt_rouge = compute_rouge_metrics(gpt_preds, gold_targets)

        print("Computing BERTScore metrics...")
        gpt_bertscore = compute_bertscore_metrics(gpt_preds, gold_targets)

        gpt_throughputs = [p['throughput_tokens_per_sec'] for p in gpt_preds_data]

        gpt_results = {
            'model': 'GPT-4o-mini',
            'latency_mean_ms': float(np.mean(gpt_latencies)),
            'latency_std_ms': float(np.std(gpt_latencies)),
            'throughput_mean_tokens_per_sec': float(np.mean(gpt_throughputs)),
            'throughput_std_tokens_per_sec': float(np.std(gpt_throughputs)),
            'memory_before_mb': None,
            'memory_after_mb': None,
            'memory_delta_mb': None,
            'rouge': gpt_rouge,
            'bertscore': gpt_bertscore,
            'per_example': gpt_preds_data
        }

    except Exception as e:
        print(f"Error with GPT-4o-mini: {e}")
        gpt_results = None

    # ============ COMPARISON ============
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    # Create comparison table
    comparison_data = {
        'Metric': [
            'Latency (ms)',
            'Throughput (tokens/sec)',
            'Memory Delta (MB)',
            'ROUGE-1 (F)',
            'ROUGE-2 (F)',
            'ROUGE-L (F)',
            'BERTScore F1'
        ],
        'Local Model': [
            f"{local_results['latency_mean_ms']:.2f} ± {local_results['latency_std_ms']:.2f}",
            f"{local_results['throughput_mean_tokens_per_sec']:.2f} ± {local_results['throughput_std_tokens_per_sec']:.2f}",
            f"{local_results['memory_delta_mb']:.2f}",
            f"{local_rouge['rouge1']['mean']:.4f} ± {local_rouge['rouge1']['std']:.4f}",
            f"{local_rouge['rouge2']['mean']:.4f} ± {local_rouge['rouge2']['std']:.4f}",
            f"{local_rouge['rougeL']['mean']:.4f} ± {local_rouge['rougeL']['std']:.4f}",
            f"{local_bertscore['f1']['mean']:.4f} ± {local_bertscore['f1']['std']:.4f}" if local_bertscore else "N/A"
        ]
    }

    if gpt_results:
        comparison_data['GPT-4o-mini'] = [
            f"{gpt_results['latency_mean_ms']:.2f} ± {gpt_results['latency_std_ms']:.2f}",
            f"{gpt_results['throughput_mean_tokens_per_sec']:.2f} ± {gpt_results['throughput_std_tokens_per_sec']:.2f}",
            "N/A (API)",
            f"{gpt_rouge['rouge1']['mean']:.4f} ± {gpt_rouge['rouge1']['std']:.4f}",
            f"{gpt_rouge['rouge2']['mean']:.4f} ± {gpt_rouge['rouge2']['std']:.4f}",
            f"{gpt_rouge['rougeL']['mean']:.4f} ± {gpt_rouge['rougeL']['std']:.4f}",
            f"{gpt_bertscore['f1']['mean']:.4f} ± {gpt_bertscore['f1']['std']:.4f}" if gpt_bertscore else "N/A"
        ]

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()

    # Save detailed results
    results = {
        'local_model': local_results,
        'gpt4_mini': gpt_results,
        'comparison_table': comparison_data
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {args.out}")


if __name__ == '__main__':
    main()
