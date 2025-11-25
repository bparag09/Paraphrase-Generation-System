# Text Paraphrasing System

An efficient, production-ready text paraphrasing system optimized for domain-specific content (cover letters, legal documents, medical records) with strict latency constraints.

## Project Overview

This project evaluates and deploys a quantized ONNX T5-Small model for high-performance, CPU-friendly text summarization. The system meets an 800ms latency constraint while maintaining acceptable summary quality across multiple domains.

**Key Features:**
- Quantized ONNX model (50% size reduction)
- CPU-only inference (no GPU required)
- 795ms average latency (meets 800ms constraint)
- RESTful API via FastAPI
- Comprehensive evaluation metrics (ROUGE, BERTScore)
- Comparison with GPT-4o-mini baseline

---

## Directory Structure

```
Telus_Summarizer/
├── README.md                          # This file
├── REPORT.md                          # Detailed evaluation & optimization analysis
├── requirements.txt                   # Python dependencies
├── .env                               # environment file
│
├── data/
│   ├── data.json                      # Sample evaluation dataset (3 domain examples)
│   ├── comparison_results.json        # Local model vs GPT-4o-mini results
│   └── evaluation_results.json        # Detailed evaluation metrics
│
├── models/
│   └── quantized_models/
│       ├── t5-small-finetuned-summarization-xsum/
│       │   ├── encoder_model.onnx     # ONNX encoder (quantized INT8)
│       │   ├── decoder_model.onnx     # ONNX decoder
│       │   ├── decoder_with_past_model.onnx
│       │   ├── config.json            # Model configuration
│       │   ├── tokenizer.json         # BPE tokenizer
│       │   ├── vocab.json
│       │   ├── merges.txt
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer_config.json
│       │   ├── ort_config.json        # ONNX Runtime config
│       │   └── ...
│       └── bart-large-cnn/            # BART alternative (not recommended)
│
└── src/
    ├── __pycache__/
    ├── api.py                         # FastAPI web service
    ├── inference.py                   # Model loading & inference utilities
    ├── evaluate.py                    # Evaluation metrics (ROUGE, BERTScore)
    ├── compare_models.py              # Compare local model vs GPT-4o-mini
    └── experiment.ipynb               # Jupyter notebook for exploration
```

---

## Prerequisites

- **Python**: 3.9 or higher
- **Conda**: For environment management (recommended)
- **OpenAI API Key**: For GPT-4o-mini comparison (optional but recommended)
- **Disk Space**: ~500MB for quantized models

---

## Setup Instructions

### 1. Clone/Navigate to Project Directory

### 2. Create Conda Environment

Create a new Conda environment with Python 3.11:

```powershell
conda create -n summarizer python=3.11
```

Activate the environment:

```powershell
conda activate summarizer
```

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### 4. Setup Environment Variables (Required for compare_models.py)

Create a `.env` file in the project root directory with your OpenAI API key:

```powershell
# Create .env file
@"
OPENAI_API_KEY=sk-your-openai-api-key-here
"@ | Out-File -Encoding UTF8 .env
```

**Example `.env` file:**
```
OPENAI_API_KEY=sk-proj-...
```

### 5. Verify Installation of packages

---

## How to Run Each Python File

### 1. inference.py - Local Model Inference

**Purpose**: Load the quantized ONNX T5-Small model and generate summaries locally.

**Features:**
- Loads pre-trained quantized model (121 MB)
- Generates summaries with latency tracking
- Returns throughput metrics (tokens/sec)

**Usage:**

```powershell
# Single text input
python -m src/inference --text "Your text to summarize here..."
```
---

### 2. evaluate.py - Evaluate Model Quality

**Purpose**: Evaluate the quantized model on a dataset using ROUGE and BERTScore metrics.

**Usage:**

```powershell
# Evaluate on custom dataset
python -m src.evaluate --data_path data/custom_data.json --output_path results/eval.json
```

### 3. compare_models.py - Compare Local Model vs GPT-4o-mini

**Purpose**: Run a side-by-side comparison of the local quantized model against OpenAI's GPT-4o-mini.

**Prerequisites:**
- `.env` file must be configured with `OPENAI_API_KEY`
- Internet connectivity (for GPT-4o-mini API calls)

**Usage:**

```powershell
# Compare with custom dataset
python -m src.compare_models --data_path data/custom_data.json
```
---

### 4. api.py - Launch FastAPI Web Service

**Purpose**: Start a production-ready REST API for real-time summarization.

**Usage:**

```powershell
# Run with custom host/port
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload
```

**API Endpoints:**

#### POST `/summarize` - Generate Summary
```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Your long text here...\", \"do_sample\": false}"
```

**Request Body:**
```json
{
  "text": "Long text to summarize",
  "do_sample": false
}
```

#### GET `/health` - Health Check
```bash
curl http://127.0.0.1:8000/health
```

### 5. experiment.ipynb - Jupyter Notebook

**Purpose**: Interactive exploration, visualization, and experimentation environment.

**Features:**
- Load and inspect models
- Generate predictions with different parameters
- Analyze evaluation results
- Compare outputs across models

## Quick Start Examples

---

## Common Issues & Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'optimum'"
**Solution:**
```powershell
conda activate <your_conda_env>
pip install optimum[onnxruntime]
```

### Performance Metrics (Current System) (t5-small-finetuned-xsum)
| Metric | Value |
|--------|-------|
| Latency (mean) | 795 ms |
| Latency (std) | ±69.8 ms |
| Throughput | 359.6 tokens/sec |
| Memory Delta | 80.6 MB |
| ROUGE-1 | 0.285 |
| ROUGE-2 | 0.086 |
| ROUGE-L | 0.169 |
| BERTScore F1 | 0.200 |

## References

- **REPORT.md**: Detailed model evaluation and optimization analysis
- **Hugging Face Models**: https://huggingface.co/Rahmat82/t5-small-finetuned-summarization-xsum
- **OpenAI API**: https://platform.openai.com/