"""
Integration tests for the full routing workflow.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from routing_agent.core.detection import ModelDetector, DetectedModel
from routing_agent.core.model_registry import ModelRegistry
from routing_agent.core.router import TaskRouter
from routing_agent.core.task_classifier import TaskClassifier


def test_full_workflow_with_mock_models():
    """Test the complete workflow from detection to routing."""
    
    # Create temporary directory for test models
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake GGUF model file
        gguf_path = os.path.join(temp_dir, "test-model-Q4_K.gguf")
        with open(gguf_path, 'w') as f:
            f.write('fake gguf content')
        
        # Create a fake HuggingFace model directory
        hf_dir = os.path.join(temp_dir, "hf-model")
        os.makedirs(hf_dir)
        config_path = os.path.join(hf_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write('{"architecture": "mistral"}')
        
        # Step 1: Model Detection
        detector = ModelDetector()
        detector.search_paths = [temp_dir]
        
        detected_models = detector.scan_for_models()
        
        # Should find both models
        assert len(detected_models) == 2
        
        # Verify model types
        gguf_models = [m for m in detected_models if m.backend == 'llama.cpp']
        hf_models = [m for m in detected_models if m.backend == 'huggingface']
        
        assert len(gguf_models) == 1
        assert len(hf_models) == 1
        
        # Step 2: Model Registration
        registry = ModelRegistry()
        registered_models = registry.register_from_detection(detector)
        
        # Should register both models
        assert len(registered_models) == 2
        
        # Verify capabilities were inferred
        for model in registered_models:
            assert len(model.capabilities) > 0
            assert 'general' in model.capabilities
        
        # Step 3: Task Routing
        router = TaskRouter(registry)
        
        # Test coding task routing
        coding_decision = router.route_task("Write a Python function")
        assert coding_decision.model_id in registry.models
        assert coding_decision.confidence > 0.5
        
        # Test reasoning task routing
        reasoning_decision = router.route_task("Explain quantum computing")
        assert reasoning_decision.model_id in registry.models
        assert reasoning_decision.confidence > 0.5
        
        # Test task override
        override_decision = router.route_task("Some text", override_category="coding")
        assert override_decision.model_id in registry.models
        assert override_decision.confidence == 1.0


def test_workflow_with_no_models():
    """Test workflow behavior when no models are available."""
    
    # Step 1: Empty detection
    detector = ModelDetector()
    with patch.object(detector, 'search_paths', []):
        detected_models = detector.scan_for_models()
        assert len(detected_models) == 0
    
    # Step 2: Empty registration
    registry = ModelRegistry()
    registered_models = registry.register_from_detection(detector)
    assert len(registered_models) == 0
    
    # Step 3: Router should handle empty registry
    router = TaskRouter(registry)
    
    # Should raise exception when trying to route with no models
    with pytest.raises(Exception):
        router.route_task("Test task")


def test_workflow_with_config_persistence():
    """Test workflow with configuration persistence."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test model
        gguf_path = os.path.join(temp_dir, "test-model.gguf")
        with open(gguf_path, 'w') as f:
            f.write('fake gguf content')
        
        # Detection and registration
        detector = ModelDetector()
        detector.search_paths = [temp_dir]
        
        registry = ModelRegistry()
        registry.register_from_detection(detector)
        
        # Save configuration
        config_path = os.path.join(temp_dir, "agents.yaml")
        registry.save_to_config(config_path)
        
        # Verify config file was created
        assert os.path.exists(config_path)
        
        # Create new registry and load from config
        new_registry = ModelRegistry()
        new_registry.load_from_config(config_path)
        
        # Should have same models
        assert len(new_registry.get_all_models()) == 1
        assert new_registry.get_all_models()[0].name == "test-model.gguf"
        
        # Test routing with loaded registry
        router = TaskRouter(new_registry)
        decision = router.route_task("Test task")
        
        assert decision.model_id in new_registry.models


def test_workflow_with_routing_rules():
    """Test workflow with custom routing rules."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test models
        for i in range(3):
            model_path = os.path.join(temp_dir, f"model-{i}.gguf")
            with open(model_path, 'w') as f:
                f.write('fake gguf content')
        
        # Detection and registration
        detector = ModelDetector()
        detector.search_paths = [temp_dir]
        
        registry = ModelRegistry()
        registry.register_from_detection(detector)
        
        # Create router
        router = TaskRouter(registry)
        
        # Test different task types
        task_types = ["coding", "reasoning", "advanced", "lightweight"]
        
        for task_type in task_types:
            decision = router.route_task(f"Test {task_type} task", override_category=task_type)
            assert decision.model_id in registry.models
            assert decision.confidence == 1.0
        
        # Test available categories
        categories = router.get_available_categories()
        for task_type in task_types:
            assert task_type in categories


def test_workflow_error_handling():
    """Test error handling in the workflow."""
    
    # Test with invalid model paths
    detector = ModelDetector()
    
    # Mock a detected model with invalid path
    invalid_model = DetectedModel(
        name="invalid-model",
        path="/nonexistent/path/model.gguf",
        model_type="gguf",
        backend="llama.cpp",
        family="llama"
    )
    
    registry = ModelRegistry()
    
    # Should handle invalid model gracefully
    with patch.object(detector, 'get_detected_models', return_value=[invalid_model]):
        registered_models = registry.register_from_detection(detector)
        # Should still try to register but may fail
        assert len(registered_models) == 0  # Registration should fail for invalid path
    
    # Test routing with empty registry
    router = TaskRouter(registry)
    
    with pytest.raises(Exception):
        router.route_task("Test task")


def test_workflow_with_priority_routing():
    """Test workflow with priority-based routing."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple models
        models_data = [
            ("coding-model.gguf", "llama"),
            ("reasoning-model.gguf", "mistral"),
            ("general-model.gguf", "phi")
        ]
        
        for model_name, family in models_data:
            model_path = os.path.join(temp_dir, model_name)
            with open(model_path, 'w') as f:
                f.write('fake gguf content')
        
        # Detection and registration
        detector = ModelDetector()
        detector.search_paths = [temp_dir]
        
        registry = ModelRegistry()
        registry.register_from_detection(detector)
        
        # Get registered models
        all_models = registry.get_all_models()
        assert len(all_models) == 3
        
        # Test that coding tasks prefer coding-capable models
        router = TaskRouter(registry)
        
        coding_decision = router.route_task("Write Python code")
        coding_model = registry.get_model_by_id(coding_decision.model_id)
        
        # Should prefer the llama model for coding
        assert "coding" in coding_model.capabilities
        
        # Test fallback behavior
        # Remove coding-capable models and test fallback
        coding_models = registry.get_models_by_capability("coding")
        for model in coding_models:
            del registry.models[model.id]
        
        # Should fallback to general models
        fallback_decision = router.route_task("Write Python code")
        fallback_model = registry.get_model_by_id(fallback_decision.model_id)
        
        assert "general" in fallback_model.capabilities