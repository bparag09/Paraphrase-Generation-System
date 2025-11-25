import argparse
import time
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os

def generate_summary(summarizer, text, min_length=100, max_length=150, do_sample=False):
    params = {
        'do_sample': do_sample
    }

    start = time.perf_counter()
    try:
        pred_obj = summarizer(text, **params)
        text_out = pred_obj[0].get('summary_text') if isinstance(pred_obj, list) and len(pred_obj) > 0 else str(pred_obj)
    except Exception as e:
        print(f"Error during generation: {e}")
        text_out = None
    latency_ms = (time.perf_counter() - start) * 1000.0
    return text_out, latency_ms


def load_quantized_model(quantized_model_dir='../models/quantized_models/t5-small-finetuned-summarization-xsum/', batch_size=12):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        quantized_model_dir = os.path.join(BASE_DIR, "..", "models", "quantized_models", "t5-small-finetuned-summarization-xsum")
        quantized_model_dir = os.path.abspath(quantized_model_dir)
        print("Loading quantized model from:", quantized_model_dir)
        quantized_model = ORTModelForSeq2SeqLM.from_pretrained(quantized_model_dir)
        quantized_tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
        summarizer = pipeline(
            'summarization',
            model=quantized_model,
            tokenizer=quantized_tokenizer,
            device_map='auto',
            batch_size=batch_size
        )
        return summarizer
    except Exception as e:
        print(f"Error loading quantized model from {quantized_model_dir}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='../models/quantized_models/t5-small-finetuned-summarization-xsum/')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--batch_size', type=int, default=12)
    args = parser.parse_args()

    summarizer = load_quantized_model(args.model_dir, batch_size=args.batch_size)
    if args.text is None:
        args.text = 'A cover letter is a formal document that accompanies your resume when you apply for a job. It serves as an introduction and provides additional context for your application.'

    out, latency = generate_summary(summarizer, args.text, min_length=args.min_length, max_length=args.max_length, do_sample=args.do_sample)

    print(f'Latency ms: {latency:.1f}')
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
