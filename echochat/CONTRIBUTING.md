# Contributing to EchoChat

Thank you for your interest in contributing to EchoChat! This document provides guidelines and instructions for getting involved.

## Getting Help

- **Questions?** Open a [GitHub Discussion](https://github.com/yourusername/echochat/discussions)
- **Found a bug?** Open a [GitHub Issue](https://github.com/yourusername/echochat/issues)
- **Need details?** Check [README_CONTRIBUTORS.md](README_CONTRIBUTORS.md) for in-depth documentation

## How to Contribute

### 1. Report Bugs

Before reporting, check if the issue already exists. When reporting:

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12]
- Python: [e.g., 3.9]
- Ollama Model: [e.g., mistral]
```

### 2. Suggest Enhancements

Describe the feature and its benefits:

```markdown
## Description
What problem does this solve?

## Suggested Solution
How would this work?

## Alternatives
Any other approaches?

## Additional Context
Any other information
```

### 3. Submit Code Changes

#### Step 1: Fork & Clone

```bash
git clone https://github.com/yourusername/echochat.git
cd echochat
```

#### Step 2: Create a Feature Branch

```bash
# Create a descriptive branch name
git checkout -b feature/adding-telegram-support
git checkout -b fix/memory-retrieval-bug
git checkout -b docs/improve-api-docs
```

#### Step 3: Make Changes

- Keep changes focused on one feature/bug
- Write clear, self-documenting code
- Add comments for complex logic
- Update docstrings

#### Step 4: Test Your Changes

```bash
# Run existing tests
pytest backend/test_main.py -v

# Test your specific feature
python -m backend.main
```

#### Step 5: Commit with Clear Messages

```bash
# Good commit messages
git commit -m "[FEATURE] Add Telegram chat export support"
git commit -m "[FIX] Resolve embeddings loading error"
git commit -m "[DOCS] Add API endpoint examples"

# Bad commit messages
git commit -m "fix stuff"
git commit -m "updated"
```

#### Step 6: Push & Create Pull Request

```bash
git push origin feature/adding-telegram-support
```

Then create a PR on GitHub with:

```markdown
## Description
What does this PR do?

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing
How can reviewers test this?

## Checklist
- [ ] Code follows project style
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] No new warnings
```

## Code Style Guidelines

### Python

```python
# Type hints where helpful
from typing import List, Dict, Optional

def analyze_message(text: str, include_emojis: bool = True) -> Dict[str, any]:
    """
    Analyze a message for patterns.
    
    Args:
        text: The message to analyze
        include_emojis: Whether to extract emojis
    
    Returns:
        Dictionary with analysis results
    """
    pass


# Clear variable names
emoji_count = len(extract_emojis(text))  # Good
ec = len(extract_emojis(text))  # Bad

# Keep functions focused and small
# Max ~50 lines, one responsibility per function

# Add comments for non-obvious logic
# Calculate cosine similarity between embeddings
similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
```

### Standards

- **Line Length:** Max 100 characters (except URLs)
- **Imports:** Group into stdlib, third-party, local (separated by blank lines)
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Format:** Follow PEP 8 standard

### Example File Structure

```python
"""
Module docstring explaining purpose.
"""

# Standard library imports
import json
from pathlib import Path
from typing import List, Dict

# Third-party imports
import numpy as np
from sentence_transformers import SentenceTransformer

# Local imports
from backend.config import EMBEDDING_MODEL
from backend.utils import clean_text


class TextAnalyzer:
    """Analyzes text for linguistic patterns."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the analyzer."""
        self.model = SentenceTransformer(model_name)
    
    def analyze(self, text: str) -> Dict:
        """Analyze text."""
        pass
```

## Project Structure & Key Areas

### For Bug Fixes

Look in:
- `backend/responder.py` - LLM response generation
- `backend/memory_store.py` - Semantic search
- `backend/chat_parser.py` - WhatsApp parsing
- `backend/config.py` - Configuration issues

### For New Features

Common areas:
- **New Chat Export Format**: Modify `chat_parser.py`
- **New Analysis Metrics**: Extend `personality_analyzer.py`
- **API Endpoints**: Add to `api.py`
- **LLM Integration**: Update `responder.py`

### For Documentation

- `README.md` - User-facing docs
- `README_CONTRIBUTORS.md` - Developer docs
- Docstrings in Python files
- Comments in complex logic

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest backend/test_main.py -v

# Run with coverage
pytest --cov=backend backend/test_main.py
```

### Writing Tests

```python
import pytest
from backend.chat_parser import parse_whatsapp_chat

def test_parse_whatsapp_chat():
    """Test basic chat parsing."""
    messages = parse_whatsapp_chat("test_data/sample_chat.txt")
    
    assert len(messages) > 0
    assert all("sender" in msg for msg in messages)
    assert all("message" in msg for msg in messages)

def test_parse_whatsapp_chat_with_emojis():
    """Test parsing messages with emojis."""
    messages = parse_whatsapp_chat("test_data/emojis_chat.txt")
    
    emoji_messages = [msg for msg in messages if msg.get("has_emoji")]
    assert len(emoji_messages) > 0
```

## Documentation

### Docstring Format

```python
def retrieve_memories(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve semantically similar past messages.
    
    Uses cosine similarity on embeddings to find contextually
    relevant memories for the input query.
    
    Args:
        query: User message to find similar memories for
        top_k: Number of memories to retrieve (default: 5)
    
    Returns:
        List of memory strings, ordered by similarity
    
    Raises:
        ValueError: If query is empty
        RuntimeError: If embeddings not loaded
    
    Example:
        >>> memories = retrieve_memories("How's college?", k=3)
        >>> print(memories[0])
        "College has been crazy!"
    """
    pass
```

### Updating Documentation

When adding features:
1. Add docstrings to new functions/classes
2. Update [README_CONTRIBUTORS.md](README_CONTRIBUTORS.md) module reference
3. Add usage examples
4. Update this file if adding new dev workflows

## Performance Considerations

- Memory-intensive operations: Consider batch processing
- Embedding generation: Cache embeddings when possible
- LLM calls: These are slow, avoid unnecessary calls
- File I/O: Use buffering for large files

## Security

- Never commit sensitive data (API keys, secrets)
- Use `.gitignore` for personal data
- Validate user inputs
- Sanitize outputs before display
- Consider privacy implications of new features

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Questions?

- Check [README_CONTRIBUTORS.md](README_CONTRIBUTORS.md) for detailed documentation
- Review existing code for patterns
- Ask in GitHub Discussions
- Open an issue for clarification

## Code of Conduct

Be respectful, inclusive, and constructive. Treat contributors with kindness.

---

**Thank you for contributing to EchoChat!** 🚀
