# Multi-LLM Ensemble for Medical Question Answering - Finalized Project Plan

## 1. Project Overview

**Title:** Enhancing Medical Question Answering Through Multi-LLM Ensemble Synergy

**Objective:** Develop and evaluate ensemble learning techniques that combine three LLMs (GPT-4, LLaMA-2, and Medical-ChatGPT) to improve performance on medical question answering across three benchmark datasets.

**Expected Impact:** Achieve 2-7% improvement in accuracy over individual models through intelligent ensemble strategies.

---

## 2. LLM APIs and Integration

### 2.1 GPT-4 (OpenAI API)

**API Details:**
- **Provider:** OpenAI
- **Base URL:** `https://api.openai.com/v1/chat/completions`
- **Authentication:** API key required (`OPENAI_API_KEY`)
- **Model Identifier:** `gpt-4` or `gpt-4-turbo-preview`
- **Rate Limits:** Varies by API tier
- **Cost Model:** Per-token pricing ($0.03-0.06 per 1K input tokens, $0.06-0.12 per 1K output tokens)

**Integration Requirements:**
```python
# Install: pip install openai
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Medical question here"}],
    temperature=0.7,
    max_tokens=500
)
```

**Strengths for Medical QA:**
- Strong general reasoning and understanding
- Excellent at multi-hop reasoning
- Good performance on complex clinical scenarios

---

### 2.2 LLaMA-2 (Meta's Open-Source Model)

**API Details:**
- **Provider:** Multiple options (Together AI, Replicate, Hugging Face Inference API)
- **Recommended Provider:** Together AI
- **Base URL:** `https://api.together.xyz/v1/chat/completions`
- **Authentication:** API key required (`TOGETHER_API_KEY`)
- **Model Identifier:** `meta-llama/Llama-2-70b-chat-hf`
- **Rate Limits:** Depends on pricing tier
- **Cost Model:** Lower cost than GPT-4 (~$0.0008 per 1K input tokens)

**Integration Requirements:**
```python
# Install: pip install together
import together

together.api_key = "YOUR_API_KEY"
response = together.Complete.create(
    prompt="Medical question here",
    model="meta-llama/Llama-2-70b-chat-hf",
    max_tokens=500,
    temperature=0.7
)
```

**Strengths for Medical QA:**
- Cost-effective and open-source
- Transparent model weights for research
- Good efficiency-to-performance ratio
- Growing medical fine-tuned variants

---

### 2.3 Medical AI 4o (OpenAI - Oxford Medical Edition)

**API Details:**
- **Provider:** OpenAI
- **Base URL:** `https://api.openai.com/v1/chat/completions`
- **Authentication:** API key required (`OPENAI_API_KEY`)
- **Model Identifier:** `gpt-4o-medical` or `gpt-4-turbo-medical`
- **Rate Limits:** Varies by API tier
- **Cost Model:** Per-token pricing (varies by tier)

**Integration Requirements:**
```python
# Install: pip install openai
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="gpt-4o-medical",
    messages=[{"role": "user", "content": "Medical question here"}],
    temperature=0.7,
    max_tokens=500
)
```

**Strengths for Medical QA:**
- Purpose-built for medical applications by Oxford-trained medical experts
- Superior understanding of clinical terminology and medical concepts
- Optimized for high-accuracy medical reasoning and diagnosis support
- Reduced hallucination on medical facts and procedures
- State-of-the-art performance on medical benchmarks

---

## 3. Datasets

### 3.1 PubMedQA

**Description:** Biomedical research question answering dataset

**Format:** 
- **Task Type:** Yes/No/Maybe multiple choice
- **Domain Focus:** Biomedical research abstracts
- **Size:** ~1 million questions
- **Questions:** Based on PubMed article titles and abstracts

**Access:**
```python
from datasets import load_dataset
dataset = load_dataset("pubmed_qa", "pqa_artificial")
# Or: load_dataset("pubmed_qa", "pqa_labeled")
```

**Evaluation Metrics:**
- Accuracy (primary)
- Per-category performance (Yes/No/Maybe distribution)

**Use Cases in Ensemble:**
- Test general biomedical knowledge
- Evaluate literature comprehension

---

### 3.2 MedQA-USMLE

**Description:** Multiple-choice medical questions from USMLE exams

**Format:**
- **Task Type:** Multiple choice (4-5 options)
- **Domain Focus:** Clinical reasoning and medical diagnosis
- **Size:** ~12,700 questions
- **Questions:** Actual USMLE-style licensing exam questions
- **Languages:** English, Chinese, Japanese, Spanish (use English subset)

**Access:**
```python
from datasets import load_dataset
dataset = load_dataset("medmcqa")  # Note: Check Hugging Face for exact dataset
# Or download from: https://github.com/jind11/MedQA
```

**Evaluation Metrics:**
- Accuracy (primary)
- Per-specialty accuracy (e.g., cardiology, neurology)
- Confidence calibration

**Use Cases in Ensemble:**
- Evaluate clinical diagnostic reasoning
- Test highest-complexity medical scenarios
- Measure clinical reliability

---

### 3.3 MedMCQA

**Description:** Comprehensive medical multiple-choice question dataset

**Format:**
- **Task Type:** Multiple choice (4 options)
- **Domain Focus:** Multi-topic medical knowledge (Indian medical exams)
- **Size:** ~193,000 questions
- **Questions:** Various medical specialties and general medicine
- **Split:** Train (~153K), Valid (~10K), Test (~38K)

**Access:**
```python
from datasets import load_dataset
dataset = load_dataset("medmcqa")
# Includes: question, options (a/b/c/d), correct answer, subject
```

**Evaluation Metrics:**
- Accuracy (primary)
- Per-subject performance (anatomy, pharmacology, physiology, etc.)
- Difficulty-stratified accuracy

**Use Cases in Ensemble:**
- Comprehensive medical knowledge assessment
- Largest dataset for robust statistical analysis
- Diverse specialty coverage

---

## 4. Ensemble Methodology

### 4.1 Boosting-Based Weighted Majority Vote

**Algorithm:**
1. Query all three LLMs for each question
2. Track individual model accuracy on validation set
3. Assign dynamic weights based on:
   - Overall accuracy
   - Confidence scores (if available)
   - Question difficulty
4. Compute weighted majority vote across predictions
5. Output ensemble prediction with confidence

**Implementation Structure:**
```
Step 1: Individual Model Inference
├── GPT-4 inference batch
├── LLaMA-2 inference batch
└── Medical-ChatGPT inference batch

Step 2: Weight Calculation
├── Compute validation accuracy per model
├── Calculate difficulty-adjusted weights
└── Store weight matrix

Step 3: Ensemble Voting
├── Collect predictions from all models
├── Apply weights to vote counts
├── Select majority class with confidence
└── Output final prediction
```

**Hyperparameters to Tune:**
- Weight update frequency
- Difficulty estimation method
- Confidence weighting factor

---

### 4.2 Cluster-Based Dynamic Model Selection

**Algorithm:**
1. Embed each question using ClinicalBERT or similar encoder
2. Cluster questions by semantic similarity and reasoning type
3. For each cluster, determine optimal model(s) via validation performance
4. During inference:
   - Embed new question
   - Assign to closest cluster
   - Select best model(s) for that cluster
   - Output prediction with uncertainty estimate

**Implementation Structure:**
```
Step 1: Question Clustering (Offline)
├── Encode all questions with ClinicalBERT
├── Apply K-means or HDBSCAN clustering
├── Analyze cluster characteristics
└── Generate cluster-to-domain mapping

Step 2: Cluster-Specific Model Selection
├── Evaluate each model per cluster
├── Build cluster → model(s) mapping
├── Compute cluster-level confidence scores
└── Store selection matrix

Step 3: Dynamic Inference
├── Encode test question
├── Identify nearest cluster
├── Select optimal model(s)
└── Query selected model(s)
└── Output final prediction
```

**Hyperparameters to Tune:**
- Number of clusters (K)
- Embedding model (ClinicalBERT vs. alternatives)
- Cluster assignment threshold
- Model selection strategy (single best vs. ensemble within cluster)

---

## 5. Project Timeline and Milestones

### Phase 1: Setup & Baseline (Weeks 1-2)
- [ ] Set up API credentials and client libraries
- [ ] Download and preprocess datasets
- [ ] Implement inference pipelines for each LLM
- [ ] Establish baseline accuracy for each model individually
- [ ] Create evaluation framework and logging system

**Deliverable:** Individual model baseline metrics on all three datasets

### Phase 2: Boosting Ensemble (Weeks 3-4)
- [ ] Implement weighted majority vote logic
- [ ] Calculate validation-based weights
- [ ] Evaluate different weighting strategies
- [ ] Analyze performance gains
- [ ] Identify hard examples and model disagreements

**Deliverable:** Boosting ensemble results with 2-5% improvement

### Phase 3: Clustering & Dynamic Selection (Weeks 5-6)
- [ ] Implement question embedding pipeline (ClinicalBERT)
- [ ] Perform clustering analysis
- [ ] Build cluster-to-model mapping
- [ ] Evaluate dynamic selection strategy
- [ ] Compare against boosting approach

**Deliverable:** Dynamic selection results and comparative analysis

### Phase 4: Analysis & Finalization (Weeks 7-8)
- [ ] Error analysis (misclassifications, disagreement patterns)
- [ ] Calibration analysis (confidence vs. correctness)
- [ ] Cost-benefit analysis (API costs vs. accuracy gains)
- [ ] Generate visualizations and statistical tests
- [ ] Write final report and prepare publication

**Deliverable:** Complete results, visualizations, and research paper draft

---

## 6. Implementation Details

### 6.1 Repository Structure
```
multi-llm-ensemble-medical-qa/
├── config/
│   ├── api_keys.yaml          # API credentials
│   └── experiment_config.yaml  # Hyperparameters
├── data/
│   ├── pubmedqa/
│   ├── medqa_usmle/
│   └── medmcqa/
├── src/
│   ├── llm_clients/
│   │   ├── gpt4_client.py
│   │   ├── llama2_client.py
│   │   └── medical_chatgpt_client.py
│   ├── ensemble/
│   │   ├── boosting_ensemble.py
│   │   └── dynamic_selection.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── analysis.py
│   └── utils/
│       ├── dataset_loader.py
│       └── embedder.py
├── notebooks/
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_ensemble_comparison.ipynb
│   └── 03_error_analysis.ipynb
├── results/
│   ├── baseline_metrics.json
│   ├── ensemble_results.json
│   └── visualizations/
├── requirements.txt
└── README.md
```

### 6.2 API Cost Estimation

**Assumptions:** 
- PubMedQA: 10,000 sample questions
- MedQA-USMLE: 12,700 questions
- MedMCQA: 50,000 sample questions
- Average question length: 200 tokens input, 50 tokens output

**Cost per Dataset:**

| LLM | Provider | Cost/1M Tokens | Est. Cost (PubMedQA) | Est. Cost (MedQA) | Est. Cost (MedMCQA) |
|-----|----------|-----------------|----------------------|-------------------|----------------------|
| GPT-4 | OpenAI | $45 input | ~$45 | ~$57 | ~$225 |
| LLaMA-2 | Together | $0.90 input | <$1 | ~$1 | ~$5 |
| Medical-ChatGPT | HF | $3-8 input | ~$3-8 | ~$4-10 | ~$15-40 |

**Total Estimated Cost:** $150-500 for complete evaluation (one ensemble method)

---

## 7. Evaluation Metrics & Analysis

### 7.1 Primary Metrics

**Accuracy:** Percentage of correct predictions
$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

**Per-Dataset Accuracy:** Separate accuracy for each benchmark
- PubMedQA (Yes/No/Maybe classification)
- MedQA-USMLE (4-5 way classification)
- MedMCQA (4-way classification)

### 7.2 Secondary Metrics

**Confidence Calibration:** Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{m} P(\text{predicted confidence} \in \text{bin } m) \times |\text{accuracy}_m - \text{confidence}_m|$$

**Model Disagreement Rate:** Percentage of questions where ensemble models disagree
- Analyze correlation between disagreement and error rate

**Specialized Performance:** Category-specific accuracy (e.g., cardiology vs. surgery)

### 7.3 Error Analysis

- **Error Type Classification:** 
  - Factual errors (wrong medical facts)
  - Reasoning errors (correct facts, wrong logic)
  - Ambiguity mishandling
  
- **Model Contribution Analysis:**
  - Which model corrects errors from others?
  - Specialization patterns by dataset
  
- **Failure Case Documentation:**
  - Top 20 most difficult questions per dataset
  - Analysis of ensemble failure modes

---

## 8. Expected Outcomes

### 8.1 Quantitative Results

**Baseline Individual Model Performance:**
- GPT-4: 85-92% accuracy (dataset-dependent)
- LLaMA-2: 70-85% accuracy (varies by domain)
- Medical-ChatGPT: 80-90% accuracy (excellent on domain-specific questions)

**Ensemble Improvement Targets:**
- Boosting ensemble: +2-5% over best individual model
- Dynamic selection: +3-7% over best individual model
- Combined approach: Up to +7-10% on difficult domains

### 8.2 Qualitative Insights

1. **Specialization patterns:** Which model excels in which medical domains?
2. **Complementarity:** How do models' errors complement each other?
3. **Reliability:** Which ensemble better handles edge cases?
4. **Interpretability:** Can we explain ensemble decisions?

### 8.3 Deliverables

1. **Code Repository:** Fully functional ensemble implementation
2. **Results Report:** Comprehensive metrics and comparisons
3. **Visualizations:** Performance plots, confusion matrices, t-SNE clusters
4. **Error Analysis:** Detailed breakdown of failure modes
5. **Publication-Ready Paper:** Results and methodology writeup

---

## 9. Resource Requirements

### 9.1 Computing Resources
- GPU: NVIDIA A100 (optional, for local LLaMA-2 deployment)
- RAM: 32GB+ (for embedding computations)
- Storage: 50GB (for downloaded datasets and results)

### 9.2 Software Dependencies
```
python>=3.9
openai>=1.0.0
together>=0.1.0
huggingface-hub>=0.16.0
transformers>=4.30.0
datasets>=2.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0 (for embeddings)
```

### 9.3 API Access & Credentials
- OpenAI API key (GPT-4 access)
- Together AI API key (LLaMA-2)
- Hugging Face API token (Medical-ChatGPT)

---

## 10. Success Criteria

1. ✓ All three LLM APIs successfully integrated and working
2. ✓ Baseline metrics established for all three models on all three datasets
3. ✓ Boosting ensemble implemented with ≥2% improvement
4. ✓ Dynamic selection implemented with ≥3% improvement
5. ✓ Comprehensive error analysis completed
6. ✓ Reproducible results documented with code and scripts
7. ✓ Publication-quality visualizations generated
8. ✓ Statistical significance testing completed

---

## 11. References & Resources

- **LLM-Synergy Framework:** Han et al. (2023) - Ensemble methods for LLMs
- **ClinicalBERT:** Alsentzer et al. (2019) - Medical BERT embeddings
- **Dataset Papers:**
  - PubMedQA: Jin et al. (2019)
  - MedQA-USMLE: Jin et al. (2021)
  - MedMCQA: Pal et al. (2022)

---

**Project Status:** Ready for Implementation

**Last Updated:** December 2025

**Team Lead:** [Your Name]

**Expected Project Completion:** 8 weeks from start
