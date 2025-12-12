"""
Unit tests for model detection system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from routing_agent.core.detection import ModelDetector, DetectedModel
from routing_agent.utils.exceptions import ModelDetectionError


def test_detected_model_creation():
    """Test DetectedModel dataclass creation."""
    model = DetectedModel(
        name="test-model",
        path="/path/to/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    
    assert model.name == "test-model"
    assert model.path == "/path/to/model.gguf"
    assert model.model_type == "gguf"
    assert model.backend == "llama.cpp"
    assert model.size == "7B"
    assert model.quantization == "Q4_K"
    assert model.family == "llama"


def test_gguf_file_detection():
    """Test GGUF file detection."""
    detector = ModelDetector()
    
    assert detector._is_gguf_file("model.gguf")
    assert detector._is_gguf_file("model.GGUF")
    assert detector._is_gguf_file("/path/to/model.gguf")
    assert not detector._is_gguf_file("model.ggml")
    assert not detector._is_gguf_file("model.txt")


def test_huggingface_model_detection():
    """Test HuggingFace model detection."""
    detector = ModelDetector()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock HuggingFace model directory
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write('{"model_type": "llama"}')
        
        # Should detect as HuggingFace model
        assert detector._is_huggingface_model(temp_dir)
        
        # Test with different file
        bin_path = os.path.join(temp_dir, "pytorch_model.bin")
        with open(bin_path, 'w') as f:
            f.write('dummy')
        
        assert detector._is_huggingface_model(temp_dir)


def test_model_info_extraction():
    """Test model information extraction from paths."""
    detector = ModelDetector()
    
    # Test GGUF model
    gguf_info = detector._extract_model_info_from_path("/models/llama-7b-Q4_K.gguf")
    assert gguf_info['backend'] == 'llama.cpp'
    assert gguf_info['model_type'] == 'gguf'
    assert gguf_info['quantization'] == 'Q4_K'
    assert gguf_info['family'] == 'llama'
    
    # Test HuggingFace model
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write('{"architecture": "mistral"}')
        
        hf_info = detector._extract_model_info_from_path(temp_dir)
        assert hf_info['backend'] == 'huggingface'
        assert hf_info['model_type'] == 'transformers'
        assert hf_info['family'] == 'mistral'


def test_size_formatting():
    """Test file size formatting."""
    detector = ModelDetector()
    
    assert detector._format_size(1024) == "1.0 KB"
    assert detector._format_size(1024 * 1024) == "1.0 MB"
    assert detector._format_size(1024 * 1024 * 1024) == "1.0 GB"
    assert detector._format_size(512) == "512.0 B"


def test_scan_for_models_empty():
    """Test scanning when no models are present."""
    detector = ModelDetector()
    
    # Mock the search paths to return empty
    with patch.object(detector, 'search_paths', []):
        models = detector.scan_for_models()
        assert len(models) == 0


def test_save_detected_models():
    """Test saving detected models to YAML."""
    detector = ModelDetector()
    
    # Add a test model
    test_model = DetectedModel(
        name="test-model",
        path="/test/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        size="7B",
        quantization="Q4_K",
        family="llama"
    )
    detector.detected_models = [test_model]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_models.yaml")
        detector.save_detected_models(output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "test-model" in content
            assert "llama.cpp" in content


def test_get_default_search_paths():
    """Test getting default search paths."""
    detector = ModelDetector()
    paths = detector._get_default_search_paths()
    
    # Should return some paths
    assert isinstance(paths, list)
    assert len(paths) > 0
    
    # Should only return existing paths
    for path in paths:
        assert os.path.exists(path), f"Path {path} should exist"


@patch('os.path.exists')
def test_scan_with_mock_models(mock_exists):
    """Test scanning with mocked model files."""
    detector = ModelDetector()
    
    # Mock exists to return True for our test paths
    mock_exists.return_value = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake GGUF file
        gguf_path = os.path.join(temp_dir, "test-model.gguf")
        with open(gguf_path, 'w') as f:
            f.write('fake gguf content')
        
        # Mock the search paths to only include our temp dir
        detector.search_paths = [temp_dir]
        
        # Scan for models
        models = detector.scan_for_models()
        
        # Should find our fake model
        assert len(models) == 1
        assert models[0].name == "test-model.gguf"
        assert models[0].backend == "llama.cpp"


def test_no_models_error_handling():
    """Test error handling when no models are found."""
    detector = ModelDetector()
    
    # Test saving when no models
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "empty.yaml")
        
        # Should not raise error, just warn
        detector.save_detected_models(output_path)
        
        # File should not be created
        assert not os.path.exists(output_path)