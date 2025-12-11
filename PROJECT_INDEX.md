# Project Index

## 📚 Complete File Structure and Description

### Root Level Documentation

| File | Purpose |
|------|---------|
| `FINALIZED_PROJECT_PLAN.md` | Comprehensive 11-section project plan with APIs, datasets, methodology, timeline, and resource requirements |
| `README.md` | Full project documentation, installation, usage, and reference guide |
| `QUICKSTART.md` | 6-step quick start guide for immediate setup and execution |
| `EXECUTION_GUIDE.md` | Detailed execution instructions, troubleshooting, and performance tips |
| `COMPLETION_SUMMARY.md` | Summary of all deliverables and project completion status |
| `PROJECT_PROPOSAL.md` | Original project proposal with objectives and methodology |
| `requirements.txt` | Python package dependencies (17 packages) |
| `.gitignore` | Git ignore rules for credentials and data |

### Source Code (`src/`)

#### Main Orchestration
- **`main.py`** (~400 lines)
  - `EnsembleOrchestrator` class for pipeline management
  - Initialization of LLM clients
  - Dataset loading
  - Baseline evaluation
  - Ensemble evaluation
  - Report generation
  - CLI with argparse

#### LLM Clients (`src/llm_clients/`)

1. **`base_client.py`** (~40 lines)
   - `BaseLLMClient` abstract class
   - Template for all LLM implementations
   - Common methods: `query()`, `batch_query()`, `format_question()`, `extract_answer()`

2. **`gpt4_client.py`** (~100 lines)
   - `GPT4Client` for OpenAI GPT-4 API
   - Implements: query, batch_query, answer extraction
   - Features: Rate limiting, retries, token tracking

3. **`llama2_client.py`** (~100 lines)
   - `LLaMA2Client` for Together AI LLaMA-2 API
   - Same interface as GPT-4 for compatibility
   - Cost-effective inference

4. **`medical_chatgpt_client.py`** (~100 lines)
   - `MedicalChatGPTClient` for Hugging Face Inference API
   - Medical domain-specific implementation
   - Same interface for easy swapping

#### Ensemble Methods (`src/ensemble/`)

1. **`boosting_ensemble.py`** (~170 lines)
   - `BoostingEnsemble` class
   - Methods: `update_weights()`, `predict()`, `batch_predict()`, `analyze_disagreement()`
   - Features: Dynamic weighting, confidence weighting, difficulty adjustment

2. **`dynamic_selection.py`** (~180 lines)
   - `ClusterBasedEnsemble` class
   - Methods: `fit()`, `assign_cluster()`, `select_models()`, `predict()`, `batch_predict()`
   - Features: KMeans clustering, per-cluster model mapping, dynamic selection

#### Evaluation (`src/evaluation/`)

1. **`metrics.py`** (~140 lines)
   - `MetricsCalculator` for all metric calculations
   - Methods: accuracy, per-category accuracy, calibration, disagreement rate, error analysis
   - `EvaluationReport` for report generation and printing

2. **`analysis.py`** (~170 lines)
   - `ResultsAnalyzer` for comprehensive analysis
   - Methods: save/load results, model comparison, improvement analysis, error distribution, confidence gap, summary tables

#### Utilities (`src/utils/`)

1. **`config.py`** (~50 lines)
   - `ConfigLoader` for YAML configuration loading
   - Environment variable expansion
   - Methods: `load_yaml()`, `get_api_keys()`, `get_experiment_config()`

2. **`dataset_loader.py`** (~180 lines)
   - `DatasetLoader` for medical QA datasets
   - Methods: `load_pubmedqa()`, `load_medqa_usmle()`, `load_medmcqa()`, `load_all_datasets()`
   - Features: Caching, error handling, example serialization

3. **`embedder.py`** (~60 lines)
   - `QuestionEmbedder` for question embeddings
   - Uses sentence-transformers library
   - Methods: `embed_questions()`, `embed_single()`, `similarity()`

#### Package Initialization Files
- `src/__init__.py`
- `src/llm_clients/__init__.py`
- `src/ensemble/__init__.py`
- `src/evaluation/__init__.py`
- `src/utils/__init__.py`

### Configuration (`config/`)

1. **`api_keys.yaml`**
   - API credentials for 3 LLM providers
   - Environment variable support (${VAR_NAME})
   - Easy credential management without hardcoding

2. **`experiment_config.yaml`**
   - Dataset configuration (sample sizes, splits)
   - Ensemble hyperparameters
   - Evaluation settings
   - Logging configuration
   - Fully customizable

### Jupyter Notebooks (`notebooks/`)

1. **`01_baseline_evaluation.ipynb`** (~150 lines)
   - Load configuration and datasets
   - Examine dataset structure
   - Load and analyze baseline results
   - Visualizations of individual model performance

2. **`02_ensemble_comparison.ipynb`** (~130 lines)
   - Load all results (baseline, boosting, dynamic)
   - Compare ensemble methods
   - Visualize comparisons by dataset and method
   - Analyze improvements and disagreement rates

3. **`03_error_analysis.ipynb`** (~140 lines)
   - Load results and data
   - Analyze accuracy distributions
   - Model specialization analysis
   - Error type classification
   - Complementarity insights

### Project Directories

- **`data/`** - Downloaded datasets (created on first run)
- **`results/`** - Evaluation results and visualizations (created on first run)
- **`logs/`** - Experiment logs (created on first run)

## 📊 Code Statistics

### By Module
| Module | Lines | Classes | Methods |
|--------|-------|---------|---------|
| LLM Clients | ~300 | 4 | ~20 |
| Ensemble | ~350 | 2 | ~15 |
| Evaluation | ~310 | 3 | ~15 |
| Utilities | ~290 | 3 | ~20 |
| Main | ~400 | 1 | ~6 |
| **Total** | **~1,650** | **13** | **~76** |

### Documentation
| Document | Lines | Sections |
|----------|-------|----------|
| FINALIZED_PROJECT_PLAN.md | ~600 | 11 |
| README.md | ~350 | 15 |
| QUICKSTART.md | ~150 | 10 |
| EXECUTION_GUIDE.md | ~200 | 12 |
| COMPLETION_SUMMARY.md | ~350 | 15 |

## 🔗 Dependencies

### Core Libraries
- `openai` - GPT-4 API
- `together` - LLaMA-2 API
- `huggingface_hub` - Medical ChatGPT API

### ML/NLP
- `transformers` - NLP models
- `torch` - Deep learning
- `sentence-transformers` - Embeddings
- `scikit-learn` - ML utilities

### Data & Processing
- `datasets` - Dataset loading
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Utilities
- `pyyaml` - Configuration
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars
- `scipy` - Scientific computing
- `joblib` - Serialization

### Analysis
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

## 🎯 Key Components

### 1. LLM Integration
- Unified interface for 3 different LLM APIs
- Automatic retry logic and error handling
- Token usage tracking
- Extensible for adding more LLMs

### 2. Ensemble Methods
- Boosting: Weighted voting based on model accuracy
- Dynamic Selection: Cluster-based model routing
- Both use same prediction interface

### 3. Datasets
- PubMedQA: Biomedical research QA
- MedQA-USMLE: Clinical board exams
- MedMCQA: Multi-specialty medical exams

### 4. Evaluation
- Comprehensive metrics (accuracy, calibration, disagreement)
- Error analysis and classification
- Model comparison and contribution analysis

### 5. Configuration
- YAML-based settings
- Environment variable support
- Easy hyperparameter tuning

## 🚀 How to Use This Index

1. **For Setup**: See `QUICKSTART.md`
2. **For Execution**: See `EXECUTION_GUIDE.md`
3. **For Implementation Details**: Read the source files listed above
4. **For Interactive Analysis**: Open Jupyter notebooks
5. **For Complete Overview**: Read `FINALIZED_PROJECT_PLAN.md`

## ✅ All Components

- [x] 3 LLM API clients (GPT-4, LLaMA-2, Medical-ChatGPT)
- [x] 2 Ensemble methods (Boosting, Dynamic Selection)
- [x] 3 Medical QA datasets (PubMedQA, MedQA-USMLE, MedMCQA)
- [x] Comprehensive evaluation framework
- [x] Configuration system with environment variables
- [x] Main orchestration script
- [x] 3 Jupyter notebooks for analysis
- [x] Complete documentation (5 markdown files)
- [x] Requirements file with all dependencies
- [x] .gitignore for version control

---

**Total Files**: 30+ source/documentation files
**Total Lines of Code**: ~1,650 (excluding docs)
**Total Documentation**: ~1,650 lines
**Status**: ✅ COMPLETE AND READY TO USE
