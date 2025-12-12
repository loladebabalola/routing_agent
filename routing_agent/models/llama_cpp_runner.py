"""
Llama.cpp model runner for executing GGUF models.

This module provides an interface to run inference using llama.cpp models.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import subprocess
import json
import os

from ..core.model_registry import RegisteredModel
from ..utils.exceptions import ModelExecutionError
from ..utils.logging import logger


@dataclass
class LlamaCppResult:
    """Result from llama.cpp model execution."""
    response: str
    tokens: int
    time: float
    model: str


class LlamaCppRunner:
    """Runner for executing llama.cpp models."""
    
    def __init__(self, model: RegisteredModel):
        self.model = model
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate that the model is suitable for llama.cpp."""
        if self.model.backend != 'llama.cpp':
            raise ModelExecutionError(f"Model {self.model.name} is not a llama.cpp model")
        
        if not os.path.exists(self.model.path):
            raise ModelExecutionError(f"Model file not found: {self.model.path}")
    
    def _check_llama_cpp_available(self) -> bool:
        """Check if llama.cpp is available."""
        try:
            # Check for main executable
            result = subprocess.run(
                ['which', 'main'], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                return True
            
            # Check for alternative names
            alternatives = ['llama-cpp', 'llama.cpp', 'llama']
            for alt in alternatives:
                result = subprocess.run(
                    ['which', alt], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def execute(self, prompt: str, **kwargs) -> LlamaCppResult:
        """Execute the model with the given prompt."""
        if not self._check_llama_cpp_available():
            raise ModelExecutionError("llama.cpp not found. Please install llama.cpp first.")
        
        # Set default parameters
        params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 512),
            'top_p': kwargs.get('top_p', 0.9),
            'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
            'threads': kwargs.get('threads', 4),
        }
        
        try:
            # Build the command
            command = [
                'main',
                '-m', self.model.path,
                '-p', prompt,
                '--temp', str(params['temperature']),
                '-n', str(params['max_tokens']),
                '--top-p', str(params['top_p']),
                '--repeat-penalty', str(params['repeat_penalty']),
                '-t', str(params['threads']),
                '--json'
            ]
            
            logger.info(f"Executing llama.cpp model: {' '.join(command)}")
            
            # Run the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=kwargs.get('timeout', 300)
            )
            
            if result.returncode != 0:
                raise ModelExecutionError(f"llama.cpp execution failed: {result.stderr}")
            
            # Parse the output
            try:
                output_data = json.loads(result.stdout)
                response = output_data.get('response', result.stdout)
                tokens = output_data.get('tokens', len(response.split()))
                time = output_data.get('time', 0.0)
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                response = result.stdout.strip()
                tokens = len(response.split())
                time = 0.0
            
            return LlamaCppResult(
                response=response,
                tokens=tokens,
                time=time,
                model=self.model.name
            )
            
        except subprocess.TimeoutExpired:
            raise ModelExecutionError("llama.cpp execution timed out")
        except Exception as e:
            raise ModelExecutionError(f"Failed to execute llama.cpp model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'name': self.model.name,
            'path': self.model.path,
            'type': self.model.model_type,
            'backend': self.model.backend,
            'capabilities': self.model.capabilities,
            'size': self.model.size,
            'quantization': self.model.quantization,
            'family': self.model.family
        }