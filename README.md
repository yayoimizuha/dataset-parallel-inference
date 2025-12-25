# Dataset Parallel Inference

A tool to stream datasets from HuggingFace and perform parallel inference using an OpenAI-compatible API, saving results to a SQLite database.

## Architecture

- **`main.py`**: Core logic for dataset streaming, parallel processing, and database writing.
- **`core.py`**: Defines the `InferenceTask` base class.
- **`projects/`**: Directory containing project-specific implementations.

## Setup

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

2. Create a project directory (e.g., `projects/my_project`) with:
   - `task.py`: Implementation of `InferenceTask`.
   - `.env`: Configuration file.

## Usage

Run the inference for a specific project using `uv run`:

```bash
uv run main.py --project example --limit 10
```

- `--project`: name of the directory in `projects/`
- `--limit`: (Optional) limit the number of records to process

## Example Project

See `projects/example/` for a reference implementation.

### Configuration (`.env`)

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
DATASET_NAME=tatsu-lab/alpaca
DATASET_SPLIT=train
MAX_CONCURRENCY=5
```
