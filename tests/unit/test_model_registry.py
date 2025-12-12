"""
Unit tests for model registry.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from routing_agent.core.model_registry import ModelRegistry, RegisteredModel
from routing_agent.core.detection import DetectedModel
from routing_agent.utils.exceptions import ModelRegistrationError


def test_registered_model_creation():
    """Test RegisteredModel dataclass creation."""
    model = RegisteredModel(
        id="1",
        name="test-model",
        path="/path/to/model",
        model_type="gguf",
        backend="llama.cpp",
        capabilities=["general", "coding"],
        priority=2,
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    
    assert model.id == "1"
    assert model.name == "test-model"
    assert model.capabilities == ["general", "coding"]
    assert model.priority == 2


def test_model_registry_initialization():
    """Test ModelRegistry initialization."""
    registry = ModelRegistry()
    
    assert registry.models == {}
    assert registry.next_id == 1


def test_register_model():
    """Test registering a detected model."""
    registry = ModelRegistry()
    
    # Create a detected model
    detected_model = DetectedModel(
        name="test-model",
        path="/path/to/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    
    # Register the model
    registered_model = registry.register_model(detected_model)
    
    # Verify registration
    assert registered_model.id == "1"
    assert registered_model.name == "test-model"
    assert registered_model.backend == "llama.cpp"
    assert "general" in registered_model.capabilities
    assert "coding" in registered_model.capabilities
    assert "reasoning" in registered_model.capabilities
    
    # Verify model is in registry
    assert len(registry.models) == 1
    assert "1" in registry.models


def test_infer_capabilities():
    """Test capability inference from detected models."""
    registry = ModelRegistry()
    
    # Test llama model (should get coding and reasoning)
    llama_model = DetectedModel(
        name="llama-model",
        path="/path/to/llama.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    
    registered_llama = registry.register_model(llama_model)
    assert "coding" in registered_llama.capabilities
    assert "reasoning" in registered_llama.capabilities
    
    # Test mistral model (should get advanced capabilities)
    mistral_model = DetectedModel(
        name="mistral-model",
        path="/path/to/mistral.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="mistral"
    )
    
    registered_mistral = registry.register_model(mistral_model)
    assert "advanced" in registered_mistral.capabilities
    
    # Test small model (should get lightweight)
    small_model = DetectedModel(
        name="small-model",
        path="/path/to/small.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="500MB",
        quantization="Q4_K",
        family="phi"
    )
    
    registered_small = registry.register_model(small_model)
    assert "lightweight" in registered_small.capabilities


def test_get_models_by_capability():
    """Test getting models by capability."""
    registry = ModelRegistry()
    
    # Register multiple models with different capabilities
    model1 = DetectedModel(
        name="coding-model",
        path="/path/to/coding.gguf",
        model_type="gguf",
        backend="llama.cpp",
        family="llama"
    )
    
    model2 = DetectedModel(
        name="reasoning-model",
        path="/path/to/reasoning.gguf",
        model_type="gguf",
        backend="llama.cpp",
        family="mistral"
    )
    
    registry.register_model(model1)
    registry.register_model(model2)
    
    # Get coding models
    coding_models = registry.get_models_by_capability("coding")
    assert len(coding_models) == 1
    assert coding_models[0].name == "coding-model"
    
    # Get reasoning models
    reasoning_models = registry.get_models_by_capability("reasoning")
    assert len(reasoning_models) == 1
    assert reasoning_models[0].name == "reasoning-model"
    
    # Get general models (should include both)
    general_models = registry.get_models_by_capability("general")
    assert len(general_models) == 2


def test_get_model_by_id():
    """Test getting model by ID."""
    registry = ModelRegistry()
    
    # Register a model
    detected_model = DetectedModel(
        name="test-model",
        path="/path/to/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        family="llama"
    )
    
    registered_model = registry.register_model(detected_model)
    model_id = registered_model.id
    
    # Get model by ID
    retrieved_model = registry.get_model_by_id(model_id)
    assert retrieved_model is not None
    assert retrieved_model.name == "test-model"
    
    # Test non-existent ID
    assert registry.get_model_by_id("999") is None


def test_get_all_models():
    """Test getting all registered models."""
    registry = ModelRegistry()
    
    # Register multiple models
    for i in range(3):
        detected_model = DetectedModel(
            name=f"model-{i}",
            path=f"/path/to/model-{i}.gguf",
            model_type="gguf",
            backend="llama.cpp",
            family="llama"
        )
        registry.register_model(detected_model)
    
    # Get all models
    all_models = registry.get_all_models()
    assert len(all_models) == 3
    
    # Verify all models are RegisteredModel instances
    for model in all_models:
        assert isinstance(model, RegisteredModel)


def test_clear_registry():
    """Test clearing the registry."""
    registry = ModelRegistry()
    
    # Register some models
    for i in range(3):
        detected_model = DetectedModel(
            name=f"model-{i}",
            path=f"/path/to/model-{i}.gguf",
            model_type="gguf",
            backend="llama.cpp",
            family="llama"
        )
        registry.register_model(detected_model)
    
    # Clear registry
    registry.clear()
    
    # Verify registry is empty
    assert len(registry.models) == 0
    assert registry.next_id == 1


def test_parse_size_to_mb():
    """Test size parsing to megabytes."""
    registry = ModelRegistry()
    
    # Test various size formats
    assert registry._parse_size_to_mb("1GB") == 1024.0
    assert registry._parse_size_to_mb("512MB") == 512.0
    assert registry._parse_size_to_mb("2048KB") == 2.0
    assert registry._parse_size_to_mb("2097152B") == 2.0
    
    # Test invalid formats
    assert registry._parse_size_to_mb("invalid") is None
    assert registry._parse_size_to_mb("") is None
    assert registry._parse_size_to_mb(None) is None


def test_load_from_config():
    """Test loading registry from configuration file."""
    registry = ModelRegistry()
    
    # Create a temporary config file
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_agents.yaml")
        
        config_content = """
models:
  - name: "test-model"
    path: "/path/to/model.gguf"
    backend: "llama.cpp"
    model_type: "gguf"
    capabilities: ["general", "coding"]
    priority: 2
    size: "7B"
    quantization: "Q4_K"
    family: "llama"
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Load from config
        registry.load_from_config(config_path)
        
        # Verify model was loaded
        assert len(registry.models) == 1
        model = registry.get_all_models()[0]
        assert model.name == "test-model"
        assert model.capabilities == ["general", "coding"]


def test_save_to_config():
    """Test saving registry to configuration file."""
    registry = ModelRegistry()
    
    # Register a model
    detected_model = DetectedModel(
        name="test-model",
        path="/path/to/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    
    registry.register_model(detected_model)
    
    # Save to config
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_agents.yaml")
        registry.save_to_config(config_path)
        
        # Verify file was created
        assert os.path.exists(config_path)
        
        # Verify content
        with open(config_path, 'r') as f:
            content = f.read()
            assert "test-model" in content
            assert "llama.cpp" in content
            assert "coding" in content


def test_register_from_detection():
    """Test registering models from detection."""
    registry = ModelRegistry()
    
    # Create a mock detector
    mock_detector = MagicMock()
    
    # Add some detected models
    detected_models = [
        DetectedModel(
            name="model1",
            path="/path/to/model1.gguf",
            model_type="gguf",
            backend="llama.cpp",
            family="llama"
        ),
        DetectedModel(
            name="model2",
            path="/path/to/model2.gguf",
            model_type="gguf",
            backend="llama.cpp",
            family="mistral"
        )
    ]
    
    mock_detector.get_detected_models.return_value = detected_models
    
    # Register from detection
    registered_models = registry.register_from_detection(mock_detector)
    
    # Verify models were registered
    assert len(registered_models) == 2
    assert len(registry.models) == 2
    
    # Verify capabilities were inferred
    for model in registered_models:
        assert len(model.capabilities) > 0


def test_manual_capabilities_override():
    """Test manual capabilities override during registration."""
    registry = ModelRegistry()
    
    detected_model = DetectedModel(
        name="test-model",
        path="/path/to/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        family="llama"
    )
    
    # Register with manual capabilities
    custom_capabilities = ["general", "advanced", "special"]
    registered_model = registry.register_model(detected_model, custom_capabilities)
    
    # Verify custom capabilities were used
    assert registered_model.capabilities == custom_capabilities


def test_error_handling_invalid_model():
    """Test error handling for invalid model registration."""
    registry = ModelRegistry()
    
    # Test with None input
    with pytest.raises(Exception):
        registry.register_model(None)
    
    # Test with invalid detected model
    with pytest.raises(Exception):
        registry.register_model("invalid")