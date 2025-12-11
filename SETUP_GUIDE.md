# Setup Guide: API Access for Multi-LLM Ensemble

## 1. Getting GPT-4 Access

### Option A: Using GPT-4o (Recommended - Easier Access)
GPT-4o is more widely available and has similar capabilities:
1. Go to https://platform.openai.com/account/api-keys
2. Create or use your existing API key
3. Make sure your OpenAI account has payment enabled
4. Update `config/api_keys.yaml`:
   ```yaml
   openai:
     api_key: your-api-key-here
     gpt4_model: gpt-4o    # Uses GPT-4o
     medical_model: gpt-4o
   ```

### Option B: Getting Access to GPT-4 (Original)
If you need the original GPT-4 model:
1. **Sign up for OpenAI**: https://platform.openai.com/signup
2. **Enable billing**: https://platform.openai.com/account/billing/overview
3. **Request GPT-4 access**:
   - Go to https://platform.openai.com/account/api-keys
   - Your account should automatically get GPT-4 access once you have:
     - Active billing
     - Some usage history with GPT-3.5
     - Your account is at least 3 months old (for new accounts)
4. **Wait time**: Usually 1-2 weeks for new accounts to get GPT-4 access
5. Update `config/api_keys.yaml` once you have access:
   ```yaml
   openai:
     api_key: your-api-key-here
     gpt4_model: gpt-4        # Original GPT-4
     medical_model: gpt-4o    # Medical variant
   ```

### Verifying Your API Key Works
```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer your-api-key-here"
```

---

## 2. Using Purdue University LLaMA Access

Purdue provides access to LLaMA models through several methods:

### Option A: Purdue HPC (High Performance Computing)
If you have access to Purdue's HPC cluster:
1. SSH into the HPC cluster (e.g., `bell.rcac.purdue.edu`)
2. Load the LLaMA module:
   ```bash
   module load llama
   # or
   module load pytorch  # If LLaMA is installed
   ```
3. Check available models:
   ```bash
   ls /depot/models/llama/
   ```
4. Update `config/api_keys.yaml`:
   ```yaml
   purdue_llama:
     endpoint: "local"  # Local deployment
     model_path: "/depot/models/llama/llama-2-70b"
     use_hpc: true
   ```

### Option B: Purdue AI Gateway (If Available)
Check if your department/lab has access:
1. Contact your advisor or lab manager
2. Get the endpoint URL and credentials
3. Update `config/api_keys.yaml`:
   ```yaml
   purdue_llama:
     endpoint: "https://your-department-gateway.purdue.edu"
     api_key: "your-department-key"
     model: "llama-2-70b-chat"
   ```

### Option C: Local LLaMA Deployment
Run LLaMA locally using Ollama or similar:
1. Install Ollama: https://ollama.ai
2. Pull LLaMA 2 model:
   ```bash
   ollama pull llama2
   ```
3. Update `config/api_keys.yaml`:
   ```yaml
   purdue_llama:
     endpoint: "http://localhost:11434"
     model: "llama2"
     use_local: true
   ```

### Option D: Use Replicate or Hugging Face
As a backup if Purdue access isn't available:
1. Sign up at https://replicate.com or https://huggingface.co
2. Get API key
3. Update the LLaMA client to use their API

---

## 3. Configuration File Template

Here's the updated `config/api_keys.yaml`:

```yaml
# API Keys Configuration
# DO NOT commit this file to version control

openai:
  api_key: your-openai-api-key-here
  gpt4_model: gpt-4o  # Change to 'gpt-4' if you have access
  medical_model: gpt-4o
  temperature: 0.7
  max_tokens: 500

purdue_llama:
  # Choose ONE of the following configurations:
  
  # Option 1: Purdue HPC Local
  endpoint: "local"
  model_path: "/depot/models/llama/llama-2-70b"
  
  # Option 2: Purdue AI Gateway (if available)
  # endpoint: "https://your-purdue-gateway.edu"
  # api_key: "your-purdue-key"
  
  # Option 3: Local Ollama
  # endpoint: "http://localhost:11434"
  # use_local: true
  
  model: "llama-2-70b-chat"
  temperature: 0.7
  max_tokens: 500
```

---

## 4. Updating the LLaMA Client for Purdue

The LLaMA2Client will be updated to support Purdue's configuration:

```python
# The client will automatically detect:
# - Local HPC deployment
# - Remote API endpoints
# - Local Ollama instance
```

---

## 5. Testing Your Setup

Once configured, test each model:

```bash
# Test all models
python -m src.main

# Or test individual models
python -c "
from src.llm_clients.gpt4_client import GPT4Client
from src.llm_clients.llama2_client import LLaMA2Client

# Test GPT-4o
gpt = GPT4Client(api_key='your-key', config={})
response = gpt.query('What is 2+2?')
print(f'GPT-4o: {response}')

# Test LLaMA
llama = LLaMA2Client(api_key='your-key', config={})
response = llama.query('What is 2+2?')
print(f'LLaMA: {response}')
"
```

---

## 6. Troubleshooting

### GPT-4 Access Error
- **Error**: `model_not_found` for `gpt-4`
- **Solution**: Use `gpt-4o` instead (widely available) or wait for GPT-4 access

### Invalid API Key
- **Error**: `invalid_api_key` or `401 Unauthorized`
- **Solution**: 
  - Double-check your key in `config/api_keys.yaml`
  - Ensure no extra spaces or quotes
  - Regenerate key at https://platform.openai.com/account/api-keys

### Purdue LLaMA Not Found
- **Solution**:
  1. Contact your lab advisor
  2. Check if you have HPC access: `ssh bell.rcac.purdue.edu`
  3. Use local Ollama as fallback

---

## 7. Recommended Setup for Purdue Users

For maximum accessibility:
1. **Use GPT-4o** from OpenAI (easy access once you have API key)
2. **Use Local Ollama** with LLaMA 2 (no credentials needed, runs on your machine)
3. **Medical AI 4o** uses same OpenAI key as GPT-4o

This gives you a fully functional 3-model ensemble without external API dependencies!

---

Need help? Check the project README or contact your lab administrator.
