from fastapi import FastAPI
from pydantic import BaseModel
import time
from fastapi.middleware.cors import CORSMiddleware
from src.inference import load_quantized_model, generate_summary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    text: str
    do_sample: bool = False

SUMMARIZER = None

@app.on_event('startup')
def startup_event():
    global SUMMARIZER
    quantized_model_dir = '../models/quantized_models/t5-small-finetuned-summarization-xsum/'
    SUMMARIZER = load_quantized_model(quantized_model_dir, batch_size=12)

@app.post('/summarize')
def summarize(req: SummaryRequest):
    start = time.perf_counter()
    summary, gen_latency = generate_summary(
        SUMMARIZER,
        req.text,
        do_sample=req.do_sample
    )
    total_latency = (time.perf_counter() - start) * 1000.0
    return {
        'summary': summary,
        'gen_latency_ms': gen_latency,
        'total_latency_ms': total_latency
    }

@app.get('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'model': 't5-small-finetuned-summarization-xsum (quantized ONNX)'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
