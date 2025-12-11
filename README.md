# Multi-LLM Ensemble for Medical Question Answering

A comprehensive framework for evaluating ensemble methods combining GPT-4, LLaMA-2, and Medical-ChatGPT on medical question answering tasks.

## Project Overview

This project implements and evaluates two ensemble strategies for improving medical QA performance:

1. **Boosting-Based Weighted Majority Vote**: Dynamic weighting based on model accuracy
2. **Cluster-Based Dynamic Model Selection**: Intelligent model selection based on question embedding clusters

Evaluation is performed on three standard medical QA benchmarks:
- **PubMedQA**: Biomedical research question answering
- **MedQA-USMLE**: USMLE clinical board exam questions
- **MedMCQA**: Comprehensive multi-topic medical exam questions

## Project Structure

```
multi-llm-ensemble-medical-qa/
├── config/
│   ├── api_keys.yaml              # API credentials
│   └── experiment_config.yaml      # Experiment hyperparameters
├── data/
│   ├── pubmedqa/                  # PubMedQA dataset
│   ├── medqa_usmle/               # MedQA-USMLE dataset
│   └── medmcqa/                   # MedMCQA dataset
├── src/
│   ├── llm_clients/
│   │   ├── base_client.py         # Abstract base client
│   │   ├── gpt4_client.py         # GPT-4 API client
│   │   ├── llama2_client.py       # LLaMA-2 API client
│   │   └── medical_chatgpt_client.py  # Medical ChatGPT API client
│   ├── ensemble/
│   │   ├── boosting_ensemble.py   # Weighted majority vote
│   │   └── dynamic_selection.py   # Cluster-based selection
│   ├── evaluation/
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── analysis.py            # Results analysis
│   ├── utils/
│   │   ├── config.py              # Configuration loader
│   │   ├── dataset_loader.py      # Dataset utilities
│   │   └── embedder.py            # Question embeddings
│   └── main.py                    # Main orchestration script
├── notebooks/
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_ensemble_comparison.ipynb
│   └── 03_error_analysis.ipynb
├── results/
│   ├── baseline_metrics.json
│   ├── ensemble_results.json
│   └── visualizations/
├── requirements.txt
├── FINALIZED_PROJECT_PLAN.md
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Multi-LLM-Ensemble-for-Med
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials**
   
   Edit `config/api_keys.yaml` with your API credentials:
   
   ```yaml
   openai:
     api_key: your-openai-api-key
   
   together_ai:
     api_key: your-together-ai-api-key
   
   hugging_face:
     token: your-huggingface-token
   ```

   Or set environment variables:
   ```bash
   export OPENAI_API_KEY=your-key
   export TOGETHER_API_KEY=your-key
   export HF_TOKEN=your-token
   ```

## Usage

### Running Baseline Evaluation Only

```bash
python src/main.py --baseline-only --sample-size 50
```

### Running Full Pipeline

```bash
python src/main.py --sample-size 100
```

### Command-line Options

- `--config-dir`: Path to configuration directory (default: `./config`)
- `--sample-size`: Number of samples to evaluate (default: 100)
- `--baseline-only`: Run only baseline evaluation without ensemble

## LLM API Details

### GPT-4 (OpenAI)
- **Provider**: OpenAI
- **Model**: `gpt-4` or `gpt-4-turbo-preview`
- **Cost**: ~$0.06/1K input tokens
- **Strengths**: Strong general reasoning, multi-hop reasoning

### LLaMA-2 (Together AI)
- **Provider**: Together AI
- **Model**: `meta-llama/Llama-2-70b-chat-hf`
- **Cost**: ~$0.0008/1K input tokens
- **Strengths**: Cost-effective, open-source, efficient

### Medical-ChatGPT (Hugging Face)
- **Provider**: Hugging Face
- **Model**: `OpenBioLLM/OpenBioLLM-70B`
- **Cost**: Variable by provider
- **Strengths**: Domain-specific medical knowledge, reduced hallucination

## Datasets

### PubMedQA
- **Format**: Yes/No/Maybe multiple choice
- **Size**: ~1M biomedical research questions
- **Domain**: Biomedical research abstracts
- **Access**: `datasets.load_dataset("pubmed_qa", "pqa_labeled")`

### MedQA-USMLE
- **Format**: 4-5 way multiple choice
- **Size**: ~12,700 USMLE exam questions
- **Domain**: Clinical reasoning and diagnosis
- **Access**: `datasets.load_dataset("bigbio/medqa", "medqa_en")`

### MedMCQA
- **Format**: 4-way multiple choice
- **Size**: ~193,000 questions
- **Domain**: Multi-specialty medical knowledge
- **Access**: `datasets.load_dataset("medmcqa")`

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Percentage of correct predictions

### Secondary Metrics
- **Calibration (ECE)**: Expected Calibration Error
- **Disagreement Rate**: Percentage of samples where models disagree
- **Per-Category Accuracy**: Accuracy stratified by question category

## Results

Expected performance improvements:
- Ensemble accuracy: +2-7% over best individual model
- Stability: Reduced variance in predictions
- Reliability: Better calibration and confidence estimation

## Example Results

| Model | PubMedQA | MedQA | MedMCQA | Avg |
|-------|----------|-------|---------|-----|
| GPT-4 | 0.88 | 0.82 | 0.75 | 0.82 |
| LLaMA-2 | 0.78 | 0.72 | 0.68 | 0.73 |
| Medical-ChatGPT | 0.85 | 0.80 | 0.79 | 0.81 |
| **Boosting Ensemble** | **0.91** | **0.85** | **0.80** | **0.85** |
| **Dynamic Selection** | **0.92** | **0.87** | **0.82** | **0.87** |

## Configuration

Edit `config/experiment_config.yaml` to customize:
- Dataset settings (sample sizes, splits)
- Ensemble hyperparameters
- Evaluation metrics
- Logging options

## Error Analysis

The framework includes comprehensive error analysis:
- Error type classification (factual, reasoning, ambiguity)
- Model contribution analysis
- Failure case documentation
- Disagreement pattern analysis

## Jupyter Notebooks

Interactive analysis notebooks are available:

1. **01_baseline_evaluation.ipynb**: Evaluate individual models
2. **02_ensemble_comparison.ipynb**: Compare ensemble methods
3. **03_error_analysis.ipynb**: Analyze errors and disagreements

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

- **LLM-Synergy Framework**: Han et al. (2023) - Ensemble methods for LLMs
- **ClinicalBERT**: Alsentzer et al. (2019) - Medical BERT embeddings
- **Dataset Papers**:
  - PubMedQA: Jin et al. (2019)
  - MedQA-USMLE: Jin et al. (2021)
  - MedMCQA: Pal et al. (2022)

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation in `FINALIZED_PROJECT_PLAN.md`
- Review the Jupyter notebooks for examples

## Acknowledgments

- OpenAI for GPT-4 API
- Meta for LLaMA-2
- Hugging Face for model hosting and inference APIs
- Medical QA dataset creators

---

**Project Status**: Ready for Implementation

**Last Updated**: December 2025

**Expected Timeline**: 8 weeks from start
