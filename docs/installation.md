# Installation Guide

This guide provides detailed instructions for installing and setting up the Routing Agent Framework.

## 📋 System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB+ recommended for larger models)
- **Storage**: 10GB+ (depends on model sizes)
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **Python**: 3.11 or 3.12
- **RAM**: 32GB+ (for running multiple large models)
- **GPU**: NVIDIA CUDA-capable GPU (for HuggingFace models)
- **Storage**: SSD with 50GB+ free space

## 🛠️ Installation Methods

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/loladebabalola/routing_agent.git
cd routing_agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install llama-cpp-python  # For GGUF model support
pip install accelerate        # For better HuggingFace performance
```

### Method 2: Install as Package

```bash
pip install git+https://github.com/loladebabalola/routing_agent.git
```

### Method 3: Development Installation

```bash
git clone https://github.com/loladebabalola/routing_agent.git
cd routing_agent
pip install -e .  # Editable installation for development
```

## 📦 Dependency Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyYAML | >=6.0 | Configuration file parsing |
| click | >=8.0 | CLI interface components |
| rich | >=13.0 | Rich console formatting |
| pytest | >=7.0 | Testing framework |

### AI/ML Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=4.0 | HuggingFace model support |
| torch | >=2.0 | PyTorch for model execution |
| llama-cpp-python | >=0.1.0 | GGUF model support (optional) |
| accelerate | >=0.20.0 | GPU acceleration (optional) |

## 🤖 Model Setup

### GGUF Models (llama.cpp)

1. **Download models** from:
   - [HuggingFace GGUF models](https://huggingface.co/models?search=gguf)
   - [llama.cpp model zoo](https://github.com/ggerganov/llama.cpp)

2. **Place models** in detected directories:
   ```bash
   mkdir -p ~/models/gguf
   # Copy your .gguf files here
   ```

3. **Install llama.cpp** (if not using llama-cpp-python):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   ```

### HuggingFace Models

1. **Download models** using HuggingFace CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ~/models/mistral-7b
   ```

2. **Or download manually** from [HuggingFace Hub](https://huggingface.co/models)

3. **Place models** in detected directories:
   ```bash
   mkdir -p ~/.cache/huggingface/hub
   # Models will be automatically cached here
   ```

## 🔧 Configuration

### Automatic Configuration

The framework automatically:
- Detects models in standard locations
- Creates configuration files
- Sets up routing rules

### Manual Configuration

Edit `config/agents.yaml` to manually register models:

```yaml
models:
  - name: "mistral-7b"
    path: "/home/user/models/mistral-7b"
    backend: "huggingface"
    capabilities: ["general", "coding", "reasoning", "advanced"]
    model_type: "transformers"
    size: "7B"
    family: "mistral"
```

### Custom Search Paths

To add custom model search paths, modify the `ModelDetector` class in `core/detection.py`:

```python
def _get_default_search_paths(self) -> List[str]:
    paths = super()._get_default_search_paths()
    paths.append("/custom/path/to/models")
    return paths
```

## 🚀 First Run

```bash
# Start the chat interface
python -m routing_agent.cli.chat

# The system will:
# 1. Scan for available models
# 2. Register detected models
# 3. Show available models
# 4. Start the chat interface
```

## 🐛 Troubleshooting

### Common Issues

#### No models detected

**Solution:**
1. Ensure models are in detected directories
2. Check file permissions
3. Run with debug mode to see detection details

#### Model loading failed

**Solution:**
1. Verify model files are complete and not corrupted
2. Check for sufficient RAM/GPU memory
3. Install required dependencies for the model backend

#### CUDA errors

**Solution:**
1. Install correct CUDA toolkit version
2. Install cuDNN
3. Set environment variables:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

### Debugging

```bash
# Enable debug mode in chat
/debug

# Check logs
tail -f routing_agent.log

# Run with verbose logging
python -m routing_agent.cli.chat --verbose
```

## 🔒 Security Considerations

### Model Safety

- Only use models from trusted sources
- Verify model checksums when possible
- Be cautious with untrusted model inputs

### Data Privacy

- All processing happens locally
- No data is sent to external servers
- Configuration files contain no sensitive information

## 📈 Performance Optimization

### For GGUF Models

```bash
# Use llama-cpp-python with appropriate parameters
pip install llama-cpp-python --upgrade

# Set environment variables for better performance
export LLAMA_CPP_THREADS=8
```

### For HuggingFace Models

```bash
# Use GPU acceleration
pip install accelerate

# Set PyTorch to use CUDA
import torch
print(torch.cuda.is_available())  # Should return True
```

## 🎯 Next Steps

After installation:

1. **Test the system**: Try different task types
2. **Add more models**: Expand your model collection
3. **Customize routing**: Adjust routing rules for your needs
4. **Explore advanced features**: Debug mode, task overrides

## 📚 Additional Resources

- [Official Documentation](README.md)
- [Architecture Guide](architecture.md)
- [Usage Examples](usage.md)
- [GitHub Issues](https://github.com/loladebabalola/routing_agent/issues)