# CategoryVector

A Python package for category vector generation and management.

## Features

- Category vector generation using sentence transformers
- Vector storage and retrieval using FAISS
- Category hierarchy management
- Efficient similarity search
- Configurable vector dimensions and models

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/categoryvector.git
cd categoryvector
```

3. Install dependencies:
```bash
poetry install
```

## Usage

### Building Category Index

To build a category index from a JSON file:

```bash
# Windows CMD (完整命令)
poetry run python -m categoryvector.cli build --categories data/sample_categories.json --output data/vectors --vector-dim 384 --model all-MiniLM-L6-v2

# Windows CMD (简化命令)
poetry run v build --categories data/sample_categories.json --output data/vectors --vector-dim 384 --model all-MiniLM-L6-v2

# Windows CMD (直接命令)
poetry run build --categories data/sample_categories.json --output data/vectors --vector-dim 384 --model all-MiniLM-L6-v2

# Windows PowerShell (完整命令)
poetry run python -m categoryvector.cli build `
    --categories data/sample_categories.json `
    --output data/vectors `
    --vector-dim 384 `
    --model all-MiniLM-L6-v2

# Windows PowerShell (简化命令)
poetry run v build `
    --categories data/sample_categories.json `
    --output data/vectors `
    --vector-dim 384 `
    --model all-MiniLM-L6-v2

# Windows PowerShell (直接命令)
poetry run build `
    --categories data/sample_categories.json `
    --output data/vectors `
    --vector-dim 384 `
    --model all-MiniLM-L6-v2
```

### Searching Categories

To search for similar categories:

```bash
# Windows CMD (完整命令)
poetry run python -m categoryvector.cli search --index data/vectors --query "儿童" --top-k 10 --threshold 0.6

# Windows CMD (简化命令)
poetry run v search --index data/vectors --query "儿童" --top-k 10 --threshold 0.6

# Windows CMD (直接命令)
poetry run search --index data/vectors --query "儿童" --top-k 10 --threshold 0.6

# Windows PowerShell (完整命令)
poetry run python -m categoryvector.cli search `
    --index data/vectors `
    --query "儿童" `
    --top-k 10 `
    --threshold 0.6

# Windows PowerShell (简化命令)
poetry run v search `
    --index data/vectors `
    --query "儿童" `
    --top-k 10 `
    --threshold 0.6

# Windows PowerShell (直接命令)
poetry run search `
    --index data/vectors `
    --query "儿童" `
    --top-k 10 `
    --threshold 0.6
```

### Command Parameters

#### Build Command Parameters
- `--categories`: Path to the categories JSON file
- `--output`: Output directory for the vector index
- `--vector-dim`: Vector dimension (default: 384)
- `--model`: Model name for vector generation (default: all-MiniLM-L6-v2)

#### Search Command Parameters
- `--index`: Path to the vector index directory
- `--query`: Search query text
- `--top-k`: Number of results to return (default: 10)
- `--threshold`: Similarity threshold (default: 0.6)

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Style

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
