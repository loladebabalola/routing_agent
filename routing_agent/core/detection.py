"""
Model detection system for identifying available local AI models.

This module scans the system for various types of AI models and
automatically registers them for use with the routing framework.
"""

import os
import re
import yaml
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.exceptions import ModelDetectionError
from ..utils.logging import logger


@dataclass
class DetectedModel:
    """Represents a detected AI model."""
    name: str
    path: str
    model_type: str  # llama, mistral, qwen, phi, etc.
    backend: str  # llama.cpp, huggingface, etc.
    size: Optional[str] = None
    quantization: Optional[str] = None
    family: Optional[str] = None


class ModelDetector:
    """Main class for detecting AI models on the system."""
    
    def __init__(self):
        self.detected_models: List[DetectedModel] = []
        self.search_paths = self._get_default_search_paths()
    
    def _get_default_search_paths(self) -> List[str]:
        """Get default search paths for models based on OS."""
        paths = []
        
        # Common model directories
        home_dir = str(Path.home())
        common_paths = [
            f"{home_dir}/.cache/huggingface",
            f"{home_dir}/models",
            "/models",
            "/usr/share/models",
            f"{home_dir}/.local/share/models",
        ]
        
        # Add OS-specific paths
        if platform.system() == "Windows":
            common_paths.extend([
                f"{home_dir}/AppData/Local/Models",
                "C:/models",
                "C:/Program Files/models",
            ])
        elif platform.system() == "Darwin":  # macOS
            common_paths.extend([
                f"{home_dir}/Library/Application Support/Models",
                "/Library/Models",
            ])
        
        # Filter out non-existent paths and return unique ones
        existing_paths = []
        for path in common_paths:
            if os.path.exists(path):
                existing_paths.append(path)
        
        return list(set(existing_paths))
    
    def _is_gguf_file(self, file_path: str) -> bool:
        """Check if a file is a GGUF model file."""
        return file_path.lower().endswith('.gguf')
    
    def _is_huggingface_model(self, directory_path: str) -> bool:
        """Check if a directory contains HuggingFace model files."""
        required_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        
        for required_file in required_files:
            if os.path.exists(os.path.join(directory_path, required_file)):
                return True
        
        # Also check for common HF model patterns
        if any(f.endswith('.bin') or f.endswith('.safetensors') 
               for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))):
            return True
            
        return False
    
    def _extract_model_info_from_path(self, path: str) -> Dict[str, Any]:
        """Extract model information from file/directory path."""
        info = {
            'name': os.path.basename(path),
            'path': path,
            'model_type': 'unknown',
            'backend': 'unknown',
            'size': None,
            'quantization': None,
            'family': None
        }
        
        # Detect model family from name
        family_patterns = {
            'llama': r'(llama|vicuna|alpaca)',
            'mistral': r'mistral',
            'qwen': r'qwen',
            'phi': r'phi',
            'gemma': r'gemma',
            'stablelm': r'stablelm',
            'code': r'code',
        }
        
        for family, pattern in family_patterns.items():
            if re.search(pattern, path.lower()):
                info['family'] = family
                break
        
        # Detect backend and model type
        if self._is_gguf_file(path):
            info['backend'] = 'llama.cpp'
            info['model_type'] = 'gguf'
            
            # Extract quantization from filename
            quant_patterns = {
                'Q2_K': 'Q2_K',
                'Q3_K': 'Q3_K',
                'Q4_K': 'Q4_K',
                'Q5_K': 'Q5_K',
                'Q6_K': 'Q6_K',
                'Q8_0': 'Q8_0',
            }
            
            for quant, pattern in quant_patterns.items():
                if pattern in path:
                    info['quantization'] = quant
                    break
                    
        elif self._is_huggingface_model(path):
            info['backend'] = 'huggingface'
            info['model_type'] = 'transformers'
            
            # Try to read config.json for more info
            config_path = os.path.join(path, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'model_type' in config:
                            info['model_type'] = config['model_type']
                        if 'architecture' in config:
                            info['family'] = config['architecture']
                except Exception:
                    pass
        
        # Try to estimate size
        try:
            if os.path.isfile(path):
                size_bytes = os.path.getsize(path)
                info['size'] = self._format_size(size_bytes)
            elif os.path.isdir(path):
                total_size = sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file())
                info['size'] = self._format_size(total_size)
        except Exception:
            pass
            
        return info
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def scan_for_models(self) -> List[DetectedModel]:
        """Scan the system for available AI models."""
        logger.info("Starting model detection scan...")
        
        self.detected_models = []
        
        # Scan each search path
        for search_path in self.search_paths:
            logger.debug(f"Scanning path: {search_path}")
            
            try:
                for root, dirs, files in os.walk(search_path):
                    # Check for GGUF files
                    for file in files:
                        if self._is_gguf_file(file):
                            file_path = os.path.join(root, file)
                            model_info = self._extract_model_info_from_path(file_path)
                            self.detected_models.append(DetectedModel(**model_info))
                            logger.info(f"Found GGUF model: {file_path}")
                    
                    # Check for HuggingFace model directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if self._is_huggingface_model(dir_path):
                            model_info = self._extract_model_info_from_path(dir_path)
                            self.detected_models.append(DetectedModel(**model_info))
                            logger.info(f"Found HuggingFace model: {dir_path}")
                            
            except Exception as e:
                logger.warning(f"Error scanning {search_path}: {e}")
                continue
        
        logger.info(f"Model detection completed. Found {len(self.detected_models)} models.")
        return self.detected_models
    
    def save_detected_models(self, output_path: str = "detected_models.yaml") -> None:
        """Save detected models to a YAML file."""
        if not self.detected_models:
            logger.warning("No models detected to save.")
            return
        
        try:
            models_data = []
            for model in self.detected_models:
                models_data.append({
                    'name': model.name,
                    'path': model.path,
                    'model_type': model.model_type,
                    'backend': model.backend,
                    'size': model.size,
                    'quantization': model.quantization,
                    'family': model.family
                })
            
            with open(output_path, 'w') as f:
                yaml.safe_dump({'models': models_data}, f, sort_keys=False)
                
            logger.info(f"Saved detected models to {output_path}")
            
        except Exception as e:
            raise ModelDetectionError(f"Failed to save detected models: {e}")
    
    def get_detected_models(self) -> List[DetectedModel]:
        """Get the list of detected models."""
        return self.detected_models


# Import json at the top level
import json