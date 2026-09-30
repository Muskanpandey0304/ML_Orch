# Cybersecurity AI Model Orchestrator

An ML-based orchestration layer for cybersecurity AI that routes user queries to the most relevant cybersecurity expert domain and task variant before generating a guided response.

The system combines **semantic embedding-based routing, confidence-aware fallback, LLM semantic routing, and asynchronous model inference** to support structured cybersecurity query processing.

## Key Features

* **Two-stage semantic routing**
  Routes queries first to a cybersecurity expert domain and then to the most relevant task variant.

* **Embedding-based classification**
  Uses `BAAI/bge-large-en-v1.5` as the primary embedding model with `all-MiniLM-L6-v2` as a fallback.

* **Cosine-similarity routing**
  Uses normalized embeddings and similarity scores to determine the most relevant expert and task.

* **Confidence-aware fallback**
  Low-margin routing decisions can be passed to an LLM-based semantic router for additional classification.

* **Guided response generation**
  Builds domain-aware prompts using the selected expert scope, boundaries, and task context.

* **Asynchronous inference**
  Uses `AsyncOpenAI` and `httpx` to communicate with an OpenAI-compatible inference server.

* **Resilient response parsing**
  Handles structured LLM routing outputs and separates reasoning content from final answers.

## Architecture

```text
User Query
    │
    ▼
Embedding Encoder
    │
    ▼
Expert-Level Routing
    │
    ├── High Confidence ──────────────┐
    │                                 │
    └── Low Margin → LLM Router ──────┤
                                      ▼
                             Final Expert Selection
                                      │
                                      ▼
                              Variant-Level Routing
                                      │
                                      ▼
                                  Plan Generation
                                      │
                                      ▼
                             Guided Prompt Creation
                                      │
                                      ▼
                               Final LLM Inference
                                      │
                                      ▼
                                  Final Answer
```

## Routing Model

The router uses a two-stage classification process:

1. **Expert routing** identifies the most relevant cybersecurity domain.
2. **Variant routing** identifies the specific task within that domain.

Similarity scores and confidence margins are used to determine whether a query receives a generic, soft-guided, or strict routing path.

## Expert Taxonomy

The system organizes cybersecurity queries across **14 expert domains**, including areas such as:

* Code & Vulnerability Analysis
* Network & Traffic Analysis
* Cloud & Infrastructure Security
* Threat Intelligence
* Malware & Reverse Engineering
* Digital Forensics & Incident Response
* Detection Engineering & SIEM
* Identity & Access Governance
* Threat Hunting
* Application Security & Supply Chain
* GRC & Risk
* Orchestration & Synthesis
* Data Security & Privacy
* OT / ICS / SCADA Security

## Tech Stack

* **Python**
* **Sentence Transformers**
* **NumPy**
* **OpenAI Python SDK**
* **httpx**
* **AsyncIO**
* **OpenAI-compatible LLM inference**

## Installation

```bash
pip install sentence-transformers numpy openai httpx
```

## Configuration

The inference server and model can be configured through environment variables:

```bash
export SERVER_URL="http://localhost:8000/v1"
export MODEL_NAME="your-model-name"
```

On Windows PowerShell:

```powershell
$env:SERVER_URL="http://localhost:8000/v1"
$env:MODEL_NAME="your-model-name"
```

## Run

```bash
python ml_orch_new1.py
```

Then enter a cybersecurity query:

```text
Prompt > Analyze this suspicious authentication behavior...
```

The CLI displays:

* Selected routing tier
* Routing method
* Expert domain
* Confidence
* Task variant
* Generated plan
* Model reasoning, when available
* Final response

## Project Contribution

The ML/orchestration layer focuses on:

* Semantic embedding-based query routing
* Two-stage expert and task classification
* Confidence and margin-based routing decisions
* LLM-assisted semantic fallback routing
* Asynchronous inference integration
* Guided prompt construction and response orchestration

## Purpose

The project is designed to make cybersecurity AI systems more structured by **routing each request to the most relevant domain and task context before generating the final response**.

## Disclaimer

This project is intended for **authorized cybersecurity research, analysis, and defensive security use**. Use it only with systems, data, and models you are authorized to access.
