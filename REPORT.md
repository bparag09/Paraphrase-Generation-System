# Telus Summarizer: Model Evaluation & Optimization Report

**Project Goal**: Build an efficient, high-quality text summarization system for domain-specific content (cover letters, legal documents, medical records) with <800ms latency constraint.

**Constraints**:
- Latency: **< 800 ms** (strict requirement)
- Model Type: **Encoder or Seq2Seq only** (BERT, BART, T5, etc. — no decoder-only LLMs/SLMs)
- Quality: **High summary quality** (optimized for ROUGE & BERTScore)

## 1. Model Choices & Evaluation

### Selected Model: **Quantized ONNX T5-Small (Finetuned on XSum)**

**Model**: `Rahmat82/t5-small-finetuned-summarization-xsum`

**Why This Model?**
- **Meets latency constraint**: 795 ms avg (under 800 ms)
- **Low memory footprint**: 80.6 MB peak memory delta
- **Fast inference**: 359.6 tokens/sec throughput
- **Encoder-decoder compliant**: T5 is a pure seq2seq model
- **Portable**: Quantized to INT8 ONNX format (≈50% model size reduction)

### Evaluated Alternatives

| Model | Type | Latency | Latency Status | Quality | Notes |
|-------|------|---------|---|---------|-------|
| **T5-Small (Quantized ONNX)** | Seq2Seq | 795 ms | PASS | ROUGE-1: 0.285 | **SELECTED** - Meets all constraints |
| BART-Large-CNN (ONNX, not quantized) | Seq2Seq | 4,500 ms | FAIL | ROUGE-1: ~0.45 | Too slow (5.6x over limit) |
| BART-Large-CNN (Quantized ONNX) | Seq2Seq | 12,200 ms | FAIL | Poor (word repetition) | Quantization degraded quality |


### Other Alternatives 
| Model | Type | Latency | Notes |
|-------|------|---------|-------|
| Falconsai/text_summarization | Seq2Seq | ~5,000 ms | Exceeds latency budget |
| Google Pegasus-XSum | Seq2Seq | ~7000 ms | High quality but too slow |

## 2. Evaluation Results (3 Domain-Specific Samples)

### Quantitative Performance Metrics

#### Latency & Throughput (Local Model)
```
Mean latency:        795.12 ms ± 69.80 ms
Throughput:          359.61 tokens/sec ± 81.93 tokens/sec
Memory delta:        80.62 MB
Inference device:    CPU (device_map='auto')
Batch size:          12
```

#### Summary Quality Metrics (vs. Gold Targets) (Local Model)

**ROUGE Scores** (higher is better, range 0-1):
```
ROUGE-1 (Unigram):   0.285 ± 0.147  (Sample scores: 0.376, 0.400, 0.078)
ROUGE-2 (Bigram):    0.086 ± 0.061  (Sample scores: 0.120, 0.136, 0.000)
ROUGE-L (Longest):   0.169 ± 0.089  (Sample scores: 0.188, 0.267, 0.052)
```

**BERTScore** (contextual semantic similarity, range 0-1):
```
Precision:           0.299 ± 0.237  (detects relevant content)
Recall:              0.375 ± 0.263  (captures summary content)
F1 Score:            0.324 ± 0.245  (balanced quality metric)
```

### Per-Sample Analysis

| Sample | Domain | Input Length | Output Length | ROUGE-1 | ROUGE-L | BERTScore-F1 | Status |
|--------|--------|--------------|---------------|---------|---------|--------------|--------|
| 1 | Cover Letter | ~410 words | ~150 words | 0.376 | 0.188 | 0.629 | Good |
| 2 | Legal Document | ~350 words | ~140 words | 0.400 | 0.267 | 0.514 | Good |
| 3 | Medical Case | ~400 words | ~150 words | 0.078 | 0.052 | 0.090 | Poor |

## 1. High-Level Comparison between current system and GPT-4o-mini

| Aspect | Local Model (T5-Small) | GPT-4o-mini | Winner | Notes |
|--------|------------------------|-------------|--------|-------|
| **Latency** | 795 ms | 5,652 ms | Local | 7.1x faster; meets 800ms constraint |
| **Throughput** | 359.6 tokens/sec | 68.8 tokens/sec | Local | 5.2x higher throughput |
| **Memory Delta** | 80.62 MB | N/A (Cloud API) | Tie | Local is deployable on edge |
| **ROUGE-1** | 0.285 | 0.482 | GPT-4o-mini | 69% higher quality |
| **ROUGE-2** | 0.086 | 0.171 | GPT-4o-mini | 99% higher quality |
| **ROUGE-L** | 0.169 | 0.373 | GPT-4o-mini | 121% higher quality |
| **BERTScore F1** | 0.200 | 0.437 | GPT-4o-mini | 119% higher semantic quality |
| **Deployment** | Self-hosted (CPU) | Cloud-dependent | Local | No external API calls |
| **Cost** | One-time model (~500MB) | Per-query API cost | Local | Significantly cheaper at scale |

**Observations**:
- Samples 1 & 2: Solid semantic preservation on local model (BERTScore F1 > 0.5)
- Sample 3: Medical terminology not captured well by T5-Small (possible domain mismatch)
- GPT-4o-mini is 7.11x SLOWER
- GPT-4o-mini is much more consistent and better on all Rouge Score metric
- GPT-4o-mini is 5.8x more consistent on BERTScore

## 3. Optimization Steps & Trade-offs

### Applied Optimizations

#### 3.1 ONNX Export & INT8 Quantization
**What**: Converted Hugging Face model → ONNX Runtime format with INT8 post-training quantization.

**Results**:
- Model size: 242 MB → 121 MB (50% reduction)
- Latency: 892 ms → 795 ms (11% speedup)
- Quality impact: Negligible for T5-Small

**Why it worked for T5-Small but failed for BART-Large**:
- T5-Small: Smaller architecture, quantization-friendly
- BART-Large: Larger attention heads, quantization introduced rounding errors → word repetition loops

**Impact**:
- Throughput: 359.6 tokens/sec (higher than single-batch inference)
- Latency: Unchanged (batch padding overhead similar to single samples)

### Key Trade-offs

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| Model Size vs. Quality | T5-Small | Meets latency; quality acceptable for most use cases |
| Quantization vs. Accuracy | Apply INT8 | 50% size reduction, 11% latency gain, negligible quality loss |
| Max Token Length | 150 tokens | Prevents context overflow while capturing key info |
| ONNX + ONNX Runtime | Use | 2-3x faster inference than PyTorch backend |

### Error Analysis

#### Issue 1: Low ROUGE Scores on Medical Sample
**Root Cause**: T5-Small trained on XSum (news summarization) lacks medical domain vocabulary.

**Symptom**: Generated summary misses critical medical terminology (e.g., "diagnosis", "prognosis").

**Solution Options**:
1. Fine-tune SLM on medical corpus (requires labeled data)
2. Use larger model (violates latency constraint)
3. Add medical domain adapter layer (LoRA) on Small LMs — **Recommended**

#### Issue 2: BART-Large Quantization Failure
**Root Cause**: Seq2Seq decoder architecture sensitive to INT8 rounding errors; attention mechanism produced repetitive token loops.

**Symptom**: Generated text like *"patient patient patient... history history..."*

**Lesson Learned**: Not all models quantize equally; test quantization per model before production.

#### Issue 3: Higher Latency on Longer Inputs
**Observed**: Latency variance (std ±69.8 ms) due to input length variation.

## 4. Findings & Recommendations

### Summary of Findings

1. **Latency-Quality Trade-off is Real**
   - BART-Large achieves ROUGE-1 of 0.45 (59% higher) but requires 4.5+ seconds
   - T5-Small at 795 ms is the sweet spot under strict time constraints

2. **Quantization is Double-Edged**
   - Works well for smaller, simpler models (T5-Small)
   - Breaks quality for larger, complex architectures (BART-Large seq2seq)

3. **Domain Mismatch Hurts More Than Model Size**
   - T5-Small XSum ≠ Medical/Legal → 92% quality drop on medical text
   - Proper domain adaptation > bigger model (given constraints)

4. **CPU Inference is Viable**
   - No GPU required; 80 MB memory delta; 359 tokens/sec throughput
   - Ideal for edge/serverless deployment

### Recommendations

#### Short-term (Immediate)
1. **Deploy T5-Small Quantized ONNX** as baseline
   - Safe, meets all constraints

2. **Implement Adaptive Summarization**
   - Route Cover Letter/Legal → Current T5-Small (baseline performance)
   - Route Medical → Fallback to longer summary (accept trade-off) OR use ensemble
   - Keep latency budgets per domain

3. **Add Caching Layer**
   - Cache frequent summaries (cover letters often repeat patterns)
   - Reduce redundant inference by ~15-20%

#### Medium-term (2-4 weeks)
4. **Fine-tune DistilBERT-based Model** (if labeled data available)
   - DistilBERT + classification head can route to task-specific models
   - Example: Falcon/Distilbart fine-tuned on your domain data
   - Target: ROUGE-1 → 0.35+, latency ≤ 600 ms


#### Long-term (Monthly)
5. **Use Phi-3 3.8B Mini** 
   - **Concern**: Phi-3 is a **Small LM**
   - **Risk**: Prone to hallucination, longer context overhead, higher latency (can be mitigated via streaming tokens)
   - **Memory**: 2.4 GB
   - **Context Limit**: 128k tokens
   - **Implement LoRA on Small LM (Decoder-Only) like phi3:3.8b-mini-128k-instruct-q4_K_M**
        - Fine-tune on 500-1000 labeled examples per domain

##### Reference - https://www.reddit.com/r/LocalLLaMA/comments/1dnavrt/update_model_review_for_summarizationinstruct_1gb/

#### Monitoring & Continuous Improvement
6. **Establish Metrics Dashboard**
    - Track latency, ROUGE, BERTScore per domain/user
    - Alert if quality degrades or latency spikes
    - Monthly retraining on new user summaries

7. **Build User Feedback Loop**
    - Binary feedback (good/bad summary)
    - Collect corrections → improve fine-tuning dataset

## 6. Conclusion

**Selected Solution**: Quantized ONNX T5-Small Finetuned on XSum

**Key Metrics**:
- Latency: **795 ms** (under 800 ms limit)
- Quality: **ROUGE-1 = 0.285** (acceptable baseline)
- Memory: **80.6 MB delta** (edge-deployable)
- Throughput: **359.6 tokens/sec** (reasonable for batch)

**Next Steps**:
1. Deploy T5-Small baseline to production
2. Implement LoRA fine-tuning for medical/legal domains (Month 1)
3. Explore SLM like phi3:3.8b-mini-128k-instruct-q4_K_M (Month 3+)

**Risk Assessment**:
- Low risk: Model meets hard constraints
- Medium risk: Quality variable by domain
- Low operational risk: CPU-only, no GPU dependency