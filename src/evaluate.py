import argparse
import json
import os
import time
from pathlib import Path
import psutil
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM, pipeline as ort_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score


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


def load_model(quantized_model_dir, batch_size=12):
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
        print(f"Error loading model: {e}")
        raise


def generate_predictions(summarizer, tokenizer, texts, do_sample=False):
    """Generate predictions and measure latency, memory, throughput."""
    proc = psutil.Process()
    
    predictions = []
    latencies = []
    mem_before = proc.memory_info().rss / (1024 ** 2)  # MB
    
    for idx, text in enumerate(texts):
        # Count input tokens
        input_tokens = len(tokenizer.encode(text, truncation=True))
        
        start = time.perf_counter()
        try:
            params = {'do_sample': do_sample}
            pred_obj = summarizer(text, **params)
            pred = pred_obj[0].get('summary_text') if isinstance(pred_obj, list) and len(pred_obj) > 0 else str(pred_obj)
        except Exception as e:
            print(f"Error generating prediction at index {idx}: {e}")
            pred = None
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)
        
        # Count output tokens
        output_tokens = len(tokenizer.encode(pred, truncation=True)) if pred else 0
        total_tokens = input_tokens + output_tokens
        throughput_tokens_per_sec = (total_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        
        predictions.append({
            'pred': pred,
            'latency_ms': latency_ms,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'throughput_tokens_per_sec': throughput_tokens_per_sec
        })
    
    mem_after = proc.memory_info().rss / (1024 ** 2)  # MB
    mem_delta = mem_after - mem_before
    
    return predictions, latencies, mem_before, mem_after, mem_delta


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
        'rouge1': {
            'mean': np.mean(rouge1_scores),
            'std': np.std(rouge1_scores),
            'scores': rouge1_scores
        },
        'rouge2': {
            'mean': np.mean(rouge2_scores),
            'std': np.std(rouge2_scores),
            'scores': rouge2_scores
        },
        'rougeL': {
            'mean': np.mean(rougel_scores),
            'std': np.std(rougel_scores),
            'scores': rougel_scores
        }
    }


def compute_bertscore_metrics(predictions, gold_targets):
    """Compute BERTScore (F1 metric)."""
    preds_clean = [p if p and p.strip() else "." for p in predictions]
    golds_clean = [g if g and g.strip() else "." for g in gold_targets]
    
    try:
        P, R, F1 = bert_score(preds_clean, golds_clean, lang='en', rescale_with_baseline=True, device='cpu')
        return {
            'precision': {
                'mean': float(P.mean()),
                'std': float(P.std()),
                'scores': P.tolist()
            },
            'recall': {
                'mean': float(R.mean()),
                'std': float(R.std()),
                'scores': R.tolist()
            },
            'f1': {
                'mean': float(F1.mean()),
                'std': float(F1.std()),
                'scores': F1.tolist()
            }
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate the quantized summarization model.')
    parser.add_argument('--data', type=str, default='data/data.json', help='Path to data.json')
    parser.add_argument('--model_dir', type=str, default='../models/quantized_models/t5-small-finetuned-summarization-xsum/', help='Path to quantized model')
    parser.add_argument('--out', type=str, default='data/evaluation_results.json', help='Output path for results')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    texts = [d.get('source') for d in data]
    gold_targets = [d.get('target') for d in data]

    print(f"Loading model from {args.model_dir}...")
    summarizer, tokenizer = load_model(args.model_dir)

    # Generate predictions
    print(f"Generating predictions on {len(texts)} examples...")
    predictions_data, latencies, mem_before, mem_after, mem_delta = generate_predictions(
        summarizer, tokenizer, texts
    )
    predictions = [p['pred'] for p in predictions_data]

    # Compute metrics
    print("Computing ROUGE metrics...")
    rouge_results = compute_rouge_metrics(predictions, gold_targets)

    print("Computing BERTScore metrics...")
    bertscore_results = compute_bertscore_metrics(predictions, gold_targets)

    # Compile results
    latency_stats = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
    }

    throughputs = [p['throughput_tokens_per_sec'] for p in predictions_data]
    throughput_stats = {
        'mean_tokens_per_sec': float(np.mean(throughputs)),
        'std_tokens_per_sec': float(np.std(throughputs)),
        'min_tokens_per_sec': float(np.min(throughputs)),
        'max_tokens_per_sec': float(np.max(throughputs)),
    }

    memory_stats = {
        'memory_before_mb': float(mem_before),
        'memory_after_mb': float(mem_after),
        'memory_delta_mb': float(mem_delta),
    }

    results = {
        'model': args.model_dir,
        'num_examples': len(texts),
        'latency': latency_stats,
        'throughput': throughput_stats,
        'memory': memory_stats,
        'rouge': rouge_results,
        'bertscore': bertscore_results,
        'per_example_results': [
            {
                'source': texts[i],
                'gold': gold_targets[i],
                'pred': predictions[i],
                'latency_ms': predictions_data[i]['latency_ms'],
                'input_tokens': predictions_data[i]['input_tokens'],
                'output_tokens': predictions_data[i]['output_tokens'],
                'throughput_tokens_per_sec': predictions_data[i]['throughput_tokens_per_sec'],
            }
            for i in range(len(texts))
        ]
    }

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.out}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nModel: {args.model_dir}")
    print(f"Examples: {len(texts)}")
    print(f"\nLATENCY:")
    print(f"  Mean: {latency_stats['mean_ms']:.2f} ms")
    print(f"  Std:  {latency_stats['std_ms']:.2f} ms")
    print(f"  Min:  {latency_stats['min_ms']:.2f} ms")
    print(f"  Max:  {latency_stats['max_ms']:.2f} ms")
    print(f"\nTHROUGHPUT:")
    print(f"  Mean: {throughput_stats['mean_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Std:  {throughput_stats['std_tokens_per_sec']:.2f} tokens/sec")
    print(f"\nMEMORY:")
    print(f"  Before: {mem_before:.2f} MB")
    print(f"  After:  {mem_after:.2f} MB")
    print(f"  Delta:  {mem_delta:.2f} MB")
    print(f"\nROUGE SCORES (F-measure):")
    print(f"  ROUGE-1: {rouge_results['rouge1']['mean']:.4f} ± {rouge_results['rouge1']['std']:.4f}")
    print(f"  ROUGE-2: {rouge_results['rouge2']['mean']:.4f} ± {rouge_results['rouge2']['std']:.4f}")
    print(f"  ROUGE-L: {rouge_results['rougeL']['mean']:.4f} ± {rouge_results['rougeL']['std']:.4f}")
    if bertscore_results:
        print(f"\nBERTSCORE (F1):")
        print(f"  Precision: {bertscore_results['precision']['mean']:.4f} ± {bertscore_results['precision']['std']:.4f}")
        print(f"  Recall:    {bertscore_results['recall']['mean']:.4f} ± {bertscore_results['recall']['std']:.4f}")
        print(f"  F1:        {bertscore_results['f1']['mean']:.4f} ± {bertscore_results['f1']['std']:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
