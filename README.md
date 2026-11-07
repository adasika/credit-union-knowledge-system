# Credit Union Knowledge System

A RAG-based knowledge system that reads Dupaco credit union website snapshots and answers member questions with the current, correct answer.

## Quick Start

```bash
# Install dependencies
pip install -r knowledge-system/requirements.txt

# Set your API key
export GEMINI_API_KEY="your-key-here"

# Run the eval
cd knowledge-system
python harness/run_eval.py --submission src --eval eval/public_eval.jsonl
```

## How It Works

See [src/NOTES.md](knowledge-system/src/NOTES.md) for a detailed explanation of the approach, design decisions, and API key setup.

## Project Structure

```
knowledge-system/
  src/               # Submission folder
    solution.py      # Main entry point with build()
    config.py        # Authority tiers, regexes, constants
    indexing.py      # BM25 indexing and chunking
    providers.py     # LLM provider management
    NOTES.md         # Approach and API key documentation
  harness/           # Scorer (provided)
  eval/              # Practice questions (provided)
  snapshots/         # Website snapshots (auto-unpacked on first run)
  requirements.txt   # Python dependencies
```
