# Execution Guidelines

## Running the Project

### Prerequisites
- Python 3.9+
- API keys for OpenAI, Together AI, and Hugging Face

### Installation
```bash
pip install -r requirements.txt
```

## Execution Steps

### Phase 1: Setup
1. Install dependencies
2. Configure API keys
3. Verify dataset loading

### Phase 2: Baseline Evaluation
```bash
python src/main.py --baseline-only --sample-size 50
```

**Output:**
- `results/baseline_results.json`
- Individual model accuracy on each dataset

### Phase 3: Ensemble Evaluation
```bash
python src/main.py --sample-size 100
```

**Output:**
- `results/boosting_ensemble_results.json`
- Ensemble accuracy and disagreement metrics
- `results/evaluation_report.json`

### Phase 4: Analysis

Open Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

1. **01_baseline_evaluation.ipynb**: Individual model performance
2. **02_ensemble_comparison.ipynb**: Ensemble comparison
3. **03_error_analysis.ipynb**: Error patterns and specialization

## Performance Expectations

### Baseline (Individual Models)
- GPT-4: 85-92% accuracy
- LLaMA-2: 70-85% accuracy
- Medical-ChatGPT: 80-90% accuracy

### Ensemble
- Boosting: +2-5% improvement
- Dynamic Selection: +3-7% improvement

## API Costs

### Per-Dataset Cost Estimates
| LLM | Cost per Dataset |
|-----|-----------------|
| GPT-4 | $50-200 |
| LLaMA-2 | <$5 |
| Medical-ChatGPT | $5-50 |

**Total estimated cost**: $150-500 for full evaluation

## Optimization Tips

1. **Reduce sample size for testing**: `--sample-size 10`
2. **Cache API responses**: Implement caching in `src/main.py`
3. **Batch processing**: Use parallel API calls where possible
4. **GPU acceleration**: Enable CUDA for embeddings

## Monitoring

Check logs:
```bash
tail -f logs/experiment.log
```

Monitor results in real-time:
```bash
ls -lh results/
cat results/baseline_results.json | python -m json.tool
```

## Troubleshooting

### Rate Limiting
- Implement exponential backoff (already in clients)
- Use async requests for batch operations
- Consider request throttling

### Dataset Issues
- Verify internet connection
- Check disk space (10GB+ recommended)
- Manually download datasets if needed

### API Errors
- Check API keys and permissions
- Verify API rate limits haven't been exceeded
- Review API documentation for service limits

## Advanced Configuration

### Custom Weights for Boosting
Edit `config/experiment_config.yaml`:
```yaml
ensemble:
  boosting:
    weight_update_frequency: 1000
    difficulty_adjustment: true
    confidence_weighting: true
```

### Clustering Parameters
```yaml
ensemble:
  dynamic_selection:
    n_clusters: 10
    clustering_method: kmeans
    cluster_selection_threshold: 0.5
```

## Result Interpretation

### Accuracy Metrics
- Higher = Better
- Compare ensemble to best individual model
- Improvement % = (ensemble - best) / best * 100

### Disagreement Rate
- 0 = All models agree
- 1 = All disagreement
- High disagreement may indicate specialization potential

### Calibration (ECE)
- Lower = Better
- Represents confidence-correctness gap
- Target: < 0.1

## Next Steps After Execution

1. **Analyze results** using Jupyter notebooks
2. **Generate visualizations** (provided in notebooks)
3. **Document findings** in a report
4. **Iterate on hyperparameters** if needed
5. **Publish results** or integrate into production

## Support Resources

- `FINALIZED_PROJECT_PLAN.md`: Detailed project plan
- `README.md`: Full documentation
- `QUICKSTART.md`: Quick setup guide
- `Jupyter notebooks`: Interactive examples
