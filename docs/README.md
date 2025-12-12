# Routing Agent Framework

![Routing Agent Logo](https://via.placeholder.com/150)

**Intelligent Local AI Model Router**

A production-grade framework for intelligently routing tasks to the most appropriate local AI models based on task requirements and model capabilities.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/loladebabalola/routing_agent.git
cd routing_agent

# Install dependencies
pip install -r requirements.txt

# Start the chat interface
python -m routing_agent.cli.chat
```

## ✨ Features

- **Automatic Model Detection**: Scans your system for GGUF and HuggingFace models
- **Intelligent Routing**: Matches tasks to the most capable available models
- **Multi-Backend Support**: Works with llama.cpp and HuggingFace models
- **CLI Chat Interface**: User-friendly command-line interface with rich formatting
- **Task Classification**: Automatically categorizes tasks (coding, reasoning, general, etc.)
- **Graceful Fallbacks**: Automatically falls back to alternative models when needed
- **Cross-Platform**: Works on Linux, macOS, and Windows

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Optional: CUDA for GPU acceleration (HuggingFace models)
- Optional: llama.cpp for GGUF model support

### Install from Source

```bash
# Clone the repository
git clone https://github.com/loladebabalola/routing_agent.git
cd routing_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for specific backends
pip install llama-cpp-python  # For GGUF models
pip install accelerate       # For better HuggingFace performance
```

### Install as Package

```bash
pip install git+https://github.com/loladebabalola/routing_agent.git
```

## 🎯 Usage

### Basic Chat

```bash
python -m routing_agent.cli.chat
```

### Task Override

```
/task coding
Write a Python function to sort a list
```

### Available Commands

- `/help` - Show available commands
- `/exit`, `/quit` - Exit the chat
- `/models` - Show available models
- `/debug` - Toggle debug mode
- `/task <type>` - Override task type
- `/clear` - Clear the screen

### Task Types

- `general` - General purpose tasks
- `coding` - Programming and coding tasks
- `reasoning` - Logical reasoning and analysis
- `advanced` - Complex and sophisticated tasks
- `lightweight` - Simple and quick tasks

## 🔧 Configuration

### Model Detection

The framework automatically scans these directories:

- `~/.cache/huggingface/`
- `~/models/`
- `/models/`
- `/usr/share/models/`

### Manual Configuration

Edit `config/agents.yaml` to manually register models:

```yaml
models:
  - name: "my-model"
    path: "/path/to/model.gguf"
    backend: "llama.cpp"
    capabilities: ["general", "coding"]
    model_type: "gguf"
    size: "7B"
```

### Routing Rules

Edit `config/routing_rules.yaml` to customize task routing:

```yaml
coding:
  priority: 3
  fallback: "general"
  description: "Programming tasks"
```

## 📂 Project Structure

```
routing_agent/
├── core/                  # Core framework components
│   ├── detection.py       # Model auto-detection
│   ├── model_registry.py  # Model management
│   ├── router.py          # Task routing logic
│   └── task_classifier.py # Task classification
├── models/                # Model execution backends
│   ├── llama_cpp_runner.py # GGUF model execution
│   └── hf_runner.py       # HuggingFace model execution
├── cli/                   # Command-line interfaces
│   └── chat.py            # Main chat interface
├── config/                # Configuration files
│   ├── agents.yaml        # Model registry config
│   └── routing_rules.yaml # Routing rules
├── utils/                 # Utilities
│   ├── exceptions.py      # Custom exceptions
│   └── logging.py         # Logging configuration
├── docs/                  # Documentation
└── tests/                 # Test suite
```

## 🤖 Supported Models

### GGUF Models (llama.cpp)

- Llama 2/3
- Mistral
- Qwen
- Phi
- Gemma
- StableLM
- CodeLlama

### HuggingFace Models

- Any transformer-based model on HuggingFace Hub
- Auto-detection of model architecture and capabilities

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
# Documentation is in Markdown format
# Build HTML docs with MkDocs (optional)
pip install mkdocs
mkdocs build
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Support

For issues, questions, or contributions:

- **GitHub Issues**: https://github.com/loladebabalola/routing_agent/issues
- **Discussions**: https://github.com/loladebabalola/routing_agent/discussions

## 🎓 Examples

### Example 1: Coding Task

```
/task coding
Write a Python function to reverse a string
```

### Example 2: Reasoning Task

```
/task reasoning
Explain the difference between quantum computing and classical computing
```

### Example 3: General Chat

```
Hello! Can you tell me about the latest developments in AI?
```

## 🚀 Roadmap

- [x] Core routing framework
- [x] Model auto-detection
- [x] CLI chat interface
- [x] Multi-backend support
- [ ] Web interface
- [ ] API server
- [ ] Model performance benchmarking
- [ ] Automatic model updates

## 📊 Performance

The framework is designed for:

- **Low latency**: Fast model selection and execution
- **High reliability**: Graceful fallbacks and error handling
- **Scalability**: Supports multiple models and backends
- **Extensibility**: Easy to add new model types and backends

## 🔒 Security

- All model execution is local - no data leaves your machine
- Configuration files are human-readable and editable
- No telemetry or data collection

## 📖 Documentation

- [Installation Guide](installation.md)
- [Architecture Overview](architecture.md)
- [Usage Guide](usage.md)
- [API Reference](api.md) (coming soon)

---

**Built with ❤️ for the AI community** | © 2023 Lola Debabalola