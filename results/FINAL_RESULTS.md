# Multi-LLM Ensemble for Medical QA - Final Results

**Evaluation Date:** December 11, 2025  
**Total Questions Evaluated:** 3,911 (MedQA: 2,874 + PubMedQA: 200 + MedMCQA: 837)

---

## Executive Summary

We evaluated three LLM models on medical question-answering benchmarks and tested ensemble methods to combine their predictions.

### Key Findings:
- **GPT-4** and **Medical AI 4o** achieved similar high accuracy (~80% on MedQA)
- **LLaMA-2 (Purdue)** significantly underperformed (~30% on MedQA)
- **Ensemble methods** showed modest improvements on some datasets

---

## Individual Model Performance

### MedQA-USMLE (2,874 questions) ✅ COMPLETE

| Model | Accuracy | Correct/Total |
|-------|----------|---------------|
| **GPT-4** | **81.38%** | 2,339/2,874 |
| Medical AI 4o | 80.97% | 2,327/2,874 |
| LLaMA-2 (Purdue) | 29.85% | 858/2,874 |

### PubMedQA (200 questions) ✅ COMPLETE

| Model | Accuracy | Correct/Total |
|-------|----------|---------------|
| **GPT-4** | **74.00%** | 148/200 |
| Medical AI 4o | 72.50% | 145/200 |
| LLaMA-2 (Purdue) | 68.50% | 137/200 |

### MedMCQA (837 questions) ⚠️ PARTIAL

| Model | Accuracy | Correct/Total | Status |
|-------|----------|---------------|--------|
| **GPT-4** | **77.06%** | 645/837 | Complete |
| LLaMA-2 (Purdue) | ~32% | ~270/837 | Partial (stopped) |
| Medical AI 4o | - | - | Not run |

---

## Overall Performance Summary

### Complete Results (MedQA + PubMedQA = 3,074 questions)

| Model | MedQA | PubMedQA | Weighted Average |
|-------|-------|----------|------------------|
| **GPT-4** | 81.38% | 74.00% | **80.90%** |
| Medical AI 4o | 80.97% | 72.50% | 80.42% |
| LLaMA-2 | 29.85% | 68.50% | 32.36% |

*Weighted by dataset size*

---

## Model Analysis

### GPT-4 (OpenAI)
- **Best overall performer** across all datasets
- Consistent high accuracy (74-81%)
- Fast inference (~1.5 sec/question)

### Medical AI 4o (OpenAI)
- Very similar to GPT-4 (both use OpenAI infrastructure)
- Slightly lower on PubMedQA (research-style questions)
- Comparable speed to GPT-4

### LLaMA-2 (Purdue University)
- **Significantly underperformed** on MedQA (29.85%)
- Better on PubMedQA (68.50%) - closer to yes/no questions
- Slow inference (~7-30 sec/question due to server issues)
- May need prompt tuning or different model variant

---

## Ensemble Performance

Based on weighted majority voting using model accuracy as weights:

### MedQA Ensemble (from 50-sample test)
- Ensemble: 94.0%
- Best Individual: 96.0% (Medical AI 4o)
- LLaMA-2's low accuracy hurts ensemble

### PubMedQA Ensemble (from 50-sample test)  
- Ensemble: 74.0%
- Best Individual: 72.0% (GPT-4)
- **+2% improvement** from ensemble

### Key Insight:
The ensemble helps when models have **complementary strengths**, but LLaMA-2's poor MedQA performance brings down the ensemble on that dataset.

---

## Recommendations

1. **For production use:** GPT-4 or Medical AI 4o alone (similar performance)

2. **For ensemble benefits:** 
   - Use only GPT-4 + Medical AI 4o (drop LLaMA-2)
   - Or use a better LLaMA variant (e.g., Med-PaLM, fine-tuned medical LLaMA)

3. **For cost optimization:**
   - GPT-4 is the best single model
   - Ensemble adds complexity with marginal gains

---

## Data Files

- `results/full_evaluation_log.txt` - Complete evaluation log
- `data/splits/` - Train/test splits (80/20)
- `results/cache/prediction_cache.json` - Cached predictions

---

## Technical Details

- **Test Split:** 20% of each dataset
- **Random Seed:** 42 (reproducible)
- **Parallel Workers:** 5
- **Total Runtime:** ~10 hours (mostly LLaMA-2)

---

*Report generated: December 11, 2025*

