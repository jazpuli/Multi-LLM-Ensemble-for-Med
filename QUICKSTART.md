# Quick Start Guide

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create environment variables for your API credentials:

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:TOGETHER_API_KEY = "your-together-ai-api-key"
$env:HF_TOKEN = "your-huggingface-token"
```

**Or edit `config/api_keys.yaml` directly:**
```yaml
openai:
  api_key: your-openai-api-key

together_ai:
  api_key: your-together-ai-api-key

hugging_face:
  token: your-huggingface-token
```

### 3. Run Baseline Evaluation

To evaluate individual models only:
```bash
python src/main.py --baseline-only --sample-size 50
```

### 4. Run Full Pipeline

To run baseline + ensemble evaluation:
```bash
python src/main.py --sample-size 100
```

### 5. View Results

Results are saved in the `results/` directory:
- `baseline_results.json` - Individual model performance
- `boosting_ensemble_results.json` - Boosting ensemble results
- `evaluation_report.json` - Summary report

### 6. Interactive Analysis

Open Jupyter notebooks for interactive analysis:
```bash
jupyter notebook notebooks/01_baseline_evaluation.ipynb
jupyter notebook notebooks/02_ensemble_comparison.ipynb
jupyter notebook notebooks/03_error_analysis.ipynb
```

## Project Structure

```
src/
├── llm_clients/         # LLM API clients
├── ensemble/            # Ensemble methods
├── evaluation/          # Metrics and analysis
├── utils/               # Utilities (config, datasets, embeddings)
└── main.py              # Main orchestration script

config/
├── api_keys.yaml        # API credentials
└── experiment_config.yaml # Experiment settings

notebooks/              # Jupyter notebooks for analysis
results/                # Evaluation results
data/                   # Downloaded datasets
```

## Key Features

### LLM Clients
- **GPT-4**: OpenAI API client
- **LLaMA-2**: Together AI client
- **Medical-ChatGPT**: Hugging Face Inference API client

### Ensemble Methods
- **Boosting Ensemble**: Weighted majority vote with adaptive weights
- **Dynamic Selection**: Cluster-based model selection

### Datasets
- **PubMedQA**: Biomedical research QA
- **MedQA-USMLE**: Clinical board exam questions
- **MedMCQA**: Multi-topic medical exam questions

### Evaluation
- Accuracy metrics
- Calibration analysis (ECE, MCE)
- Model disagreement analysis
- Error classification and analysis

## Configuration

Edit `config/experiment_config.yaml` to customize:

```yaml
datasets:
  medmcqa:
    sample_size: 50000  # Number of samples to use
    enabled: true

ensemble:
  boosting:
    enabled: true
    difficulty_adjustment: true
  
  dynamic_selection:
    enabled: true
    n_clusters: 10
```

## Common Issues

### Issue: API Key Not Found
**Solution**: Check environment variables or `config/api_keys.yaml`
```bash
echo $env:OPENAI_API_KEY  # Windows PowerShell
echo $OPENAI_API_KEY      # Linux/Mac
```

### Issue: Dataset Loading Error
**Solution**: Datasets are downloaded on-demand. Ensure internet connection and sufficient disk space (~10GB)

### Issue: Out of Memory
**Solution**: Reduce `sample_size` in `config/experiment_config.yaml`

## Next Steps

1. **Configure API credentials** (see step 2 above)
2. **Run baseline evaluation** to test setup
3. **Review results** in `results/` directory
4. **Open notebooks** for interactive analysis
5. **Adjust configuration** for your needs

## Support

For questions or issues:
1. Check `FINALIZED_PROJECT_PLAN.md` for detailed documentation
2. Review Jupyter notebooks for examples
3. Check error logs in `logs/experiment.log`

## Performance Tips

- Start with small `sample_size` (50-100) to test
- Cache results to avoid re-running expensive queries
- Use GPU if available for embedding computations
- Monitor API costs and usage

## Expected Results

Based on the project plan:
- Individual model accuracy: 70-92%
- Boosting ensemble: +2-5% improvement
- Dynamic selection: +3-7% improvement

See `FINALIZED_PROJECT_PLAN.md` for detailed expected outcomes.
