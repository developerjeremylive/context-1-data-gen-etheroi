# Context-1 Data Generation Web UI

A graphical interface for running the Context-1 synthetic data generation pipeline.

## Features

- **Easy Configuration**: All pipeline settings available through a visual interface
- **Real-time Output**: Watch the pipeline run in real-time with streaming logs
- **Multi-domain Support**: Run pipelines for Web, SEC, Patents, or Email domains
- **Model Selection**: Choose from multiple Claude models for each pipeline component
- **API Key Management**: Securely enter and manage API keys
- **Results Viewer**: Browse and inspect generated JSON files

## Prerequisites

Before running the Web UI, you need:

1. **Python 3.10+** installed
2. **Git** installed
3. **Required API Keys** (see below)

## Required API Keys

| API Key | Purpose | Required For |
|---------|---------|--------------|
| **Anthropic API Key** | Claude model access | All domains |
| **OpenAI API Key** | Embeddings | Web, SEC, Patents, Email |
| **Serper API Key** | Web search | Web domain |
| **Jina API Key** | Page fetching | Web domain |
| **Chroma API Key** | Vector database | Web, SEC, Patents, Email |
| **Chroma Database** | Vector database | Web, SEC, Patents, Email |
| **GitHub Token** | Repository access | All domains |

### Getting API Keys

- **Anthropic**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys
- **Serper**: https://serper.dev/
- **Jina**: https://jina.ai/reader/
- **Chroma**: https://console.chroma.cloud/
- **GitHub**: https://github.com/settings/tokens (create a classic PAT with `repo` scope)

## Running on Windows with uv

### Quick Start

1. **Install uv** (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install dependencies**:
   ```powershell
   cd web_ui
   uv pip install -r requirements.txt
   ```

3. **Run the Web UI**:
   ```powershell
   streamlit run app.py
   ```

4. **Open in browser**: The app will open automatically at `http://localhost:8501`

### Using a Virtual Environment

```powershell
# Create a new virtual environment
uv venv .venv

# Activate it
.venv\Scripts\Activate

# Install dependencies
uv pip install -r requirements.txt

# Run
streamlit run app.py
```

### Alternative: Using pip

If you prefer pip instead of uv:

```powershell
# Install Streamlit
pip install streamlit>=1.28.0

# Run the app
streamlit run app.py
```

## Configuration Guide

### Sidebar Settings

#### Basic Settings
- **GitHub Token**: Your personal access token for cloning the repository
- **Domain**: Select the type of data generation task:
  - Web Search Tasks
  - SEC Filing Tasks
  - Patent Prior-Art Tasks
  - Email Search Tasks
- **Repository URL**: URL of the repository (pre-filled)
- **Seeds File Path**: Path to your seeds file (e.g., `seeds.txt`)
- **Output Directory**: Where generated JSON files will be saved
- **ChromaDB Collection**: Name for the vector database collection

#### Model Selection
Choose which Claude model to use for each pipeline component:
- **Explore Model**: For discovering new information
- **Verify Model**: For fact-checking information
- **Distract Model**: For creating false leads
- **Extend Model**: For expanding the search space

#### Iteration Limits
Control how long the pipeline runs:
- **Explore Max Iterations**: Maximum exploration steps (default: 20)
- **Verify Max Retries**: Verification attempts per fact (default: 3)
- **Distract Max Iterations**: Maximum distractor generation (default: 15)
- **Extend Max Iterations**: Maximum extension steps (default: 20)
- **Extension Rounds**: Additional rounds (default: 0)
- **Max Workers**: Parallel workers (default: 8)

#### API Keys
Enter your API keys securely (not stored, just used during runtime)

### Running the Pipeline

1. Enter all required settings in the sidebar
2. Enter API keys in the sidebar
3. Click **"Run Pipeline"** button
4. Watch real-time output in the main panel
5. View generated files in the **Results** tab

## Troubleshooting

### "Failed to clone repository"
- Verify your GitHub token is correct
- Ensure the token has `repo` scope
- Check your internet connection

### "API Key Validation Failed"
- Make sure all required keys for your domain are entered
- Check the error messages for missing keys
- Refer to the API Keys table above

### "Module not found" errors
- Ensure you're in the correct directory
- Check that all dependencies are installed
- Try running: `pip install -r requirements.txt`

### Streamlit not found
- Install Streamlit: `pip install streamlit`
- Or use uv: `uv pip install streamlit`

### Port 8501 already in use
- Streamlit will automatically try port 8502
- Or stop the other application
- Or run with: `streamlit run app.py --server.port 8503`

### Slow performance
- Reduce `max_workers` to lower parallelism
- Increase iteration limits for more thorough exploration

## Architecture

```
web_ui/
├── app.py          # Main Streamlit application
├── config.py       # Configuration dataclass and validation
├── requirements.txt # Python dependencies
└── README_WEB_UI.md # This file
```

## Security Notes

- **Never commit API keys** to version control
- The Web UI does not store API keys - they're only used during the current session
- Use environment variables for production deployments
- Consider using `.env` files with proper `.gitignore` entries

## License

Same as the parent project - see [LICENSE](../LICENSE)