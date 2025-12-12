"""
Model registry for managing available AI models.

This module maintains a registry of all available models and their capabilities.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml
import os

from .detection import DetectedModel, ModelDetector
from ..utils.exceptions import ModelRegistrationError
from ..utils.logging import logger


@dataclass
class RegisteredModel:
    """Represents a registered AI model ready for use."""
    id: str
    name: str
    path: str
    model_type: str
    backend: str
    capabilities: List[str]
    priority: int = 1
    size: Optional[str] = None
    quantization: Optional[str] = None
    family: Optional[str] = None


class ModelRegistry:
    """Registry for managing available AI models."""
    
    def __init__(self):
        self.models: Dict[str, RegisteredModel] = {}
        self.next_id = 1
    
    def register_model(self, model: DetectedModel, capabilities: List[str] = None) -> RegisteredModel:
        """Register a detected model."""
        if capabilities is None:
            capabilities = self._infer_capabilities(model)
        
        model_id = str(self.next_id)
        self.next_id += 1
        
        registered_model = RegisteredModel(
            id=model_id,
            name=model.name,
            path=model.path,
            model_type=model.model_type,
            backend=model.backend,
            capabilities=capabilities,
            size=model.size,
            quantization=model.quantization,
            family=model.family
        )
        
        self.models[model_id] = registered_model
        logger.info(f"Registered model: {model.name} (ID: {model_id})")
        
        return registered_model
    
    def _infer_capabilities(self, model: DetectedModel) -> List[str]:
        """Infer model capabilities based on model type and family."""
        capabilities = ['general']
        
        # Add capabilities based on model family
        if model.family:
            if 'llama' in model.family.lower():
                capabilities.extend(['coding', 'reasoning'])
            elif 'mistral' in model.family.lower():
                capabilities.extend(['coding', 'reasoning', 'advanced'])
            elif 'qwen' in model.family.lower():
                capabilities.extend(['coding', 'reasoning'])
            elif 'phi' in model.family.lower():
                capabilities.extend(['coding', 'lightweight'])
            elif 'code' in model.family.lower():
                capabilities.extend(['coding'])
        
        # Add capabilities based on model size
        if model.size:
            size_mb = self._parse_size_to_mb(model.size)
            if size_mb and size_mb < 1000:  # Less than 1GB
                capabilities.append('lightweight')
            elif size_mb and size_mb > 7000:  # More than 7GB
                capabilities.append('advanced')
        
        return list(set(capabilities))
    
    def _parse_size_to_mb(self, size_str: str) -> Optional[float]:
        """Parse size string to megabytes."""
        if not size_str:
            return None
        
        try:
            size_str = size_str.upper()
            if 'GB' in size_str:
                return float(size_str.replace('GB', '').strip()) * 1024
            elif 'MB' in size_str:
                return float(size_str.replace('MB', '').strip())
            elif 'KB' in size_str:
                return float(size_str.replace('KB', '').strip()) / 1024
            elif 'B' in size_str:
                return float(size_str.replace('B', '').strip()) / (1024 * 1024)
        except Exception:
            return None
        
        return None
    
    def register_from_detection(self, detector: ModelDetector) -> List[RegisteredModel]:
        """Register all models detected by the model detector."""
        detected_models = detector.get_detected_models()
        registered_models = []
        
        for detected_model in detected_models:
            try:
                registered_model = self.register_model(detected_model)
                registered_models.append(registered_model)
            except Exception as e:
                logger.error(f"Failed to register model {detected_model.name}: {e}")
                continue
        
        return registered_models
    
    def get_models_by_capability(self, capability: str) -> List[RegisteredModel]:
        """Get models that have a specific capability."""
        return [model for model in self.models.values() if capability in model.capabilities]
    
    def get_model_by_id(self, model_id: str) -> Optional[RegisteredModel]:
        """Get a model by its ID."""
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[RegisteredModel]:
        """Get all registered models."""
        return list(self.models.values())
    
    def load_from_config(self, config_path: str = "config/agents.yaml") -> None:
        """Load model registry from configuration file."""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config or 'models' not in config:
                logger.warning("No models found in config file")
                return
            
            for model_config in config['models']:
                try:
                    detected_model = DetectedModel(
                        name=model_config['name'],
                        path=model_config['path'],
                        model_type=model_config.get('model_type', 'unknown'),
                        backend=model_config.get('backend', 'unknown'),
                        size=model_config.get('size'),
                        quantization=model_config.get('quantization'),
                        family=model_config.get('family')
                    )
                    
                    capabilities = model_config.get('capabilities', [])
                    self.register_model(detected_model, capabilities)
                    
                except Exception as e:
                    logger.error(f"Failed to load model from config: {e}")
                    continue
                    
        except Exception as e:
            raise ModelRegistrationError(f"Failed to load model registry from config: {e}")
    
    def save_to_config(self, config_path: str = "config/agents.yaml") -> None:
        """Save model registry to configuration file."""
        try:
            config_data = {
                'models': [{
                    'id': model.id,
                    'name': model.name,
                    'path': model.path,
                    'model_type': model.model_type,
                    'backend': model.backend,
                    'capabilities': model.capabilities,
                    'priority': model.priority,
                    'size': model.size,
                    'quantization': model.quantization,
                    'family': model.family
                } for model in self.models.values()]
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f, sort_keys=False)
                
            logger.info(f"Saved model registry to {config_path}")
            
        except Exception as e:
            raise ModelRegistrationError(f"Failed to save model registry to config: {e}")
    
    def clear(self) -> None:
        """Clear all registered models."""
        self.models.clear()
        self.next_id = 1