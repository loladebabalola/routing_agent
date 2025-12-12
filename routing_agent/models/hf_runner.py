"""
HuggingFace model runner for executing transformer models.

This module provides an interface to run inference using HuggingFace models.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

from ..core.model_registry import RegisteredModel
from ..utils.exceptions import ModelExecutionError
from ..utils.logging import logger


@dataclass
class HFResult:
    """Result from HuggingFace model execution."""
    response: str
    tokens: int
    time: float
    model: str


class HFRunner:
    """Runner for executing HuggingFace models."""
    
    def __init__(self, model: RegisteredModel):
        self.model = model
        self._validate_model()
        self.model_instance = None
        self.tokenizer = None
        self.device = None
    
    def _validate_model(self) -> None:
        """Validate that the model is suitable for HuggingFace."""
        if self.model.backend != 'huggingface':
            raise ModelExecutionError(f"Model {self.model.name} is not a HuggingFace model")
        
        if not os.path.exists(self.model.path):
            raise ModelExecutionError(f"Model directory not found: {self.model.path}")
    
    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        if self.model_instance is not None:
            return
        
        try:
            # Determine device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading model on device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.path)
            
            # Load model configuration
            config = AutoConfig.from_pretrained(self.model.path)
            
            # Load model with appropriate settings
            self.model_instance = AutoModelForCausalLM.from_pretrained(
                self.model.path,
                config=config,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device
            )
            
            # Move model to device
            self.model_instance.to(self.device)
            
            logger.info(f"Successfully loaded HuggingFace model: {self.model.name}")
            
        except Exception as e:
            raise ModelExecutionError(f"Failed to load HuggingFace model: {e}")
    
    def execute(self, prompt: str, **kwargs) -> HFResult:
        """Execute the model with the given prompt."""
        self._load_model()
        
        # Set default parameters
        params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_new_tokens': kwargs.get('max_tokens', 512),
            'top_p': kwargs.get('top_p', 0.9),
            'repetition_penalty': kwargs.get('repeat_penalty', 1.1),
            'do_sample': kwargs.get('do_sample', True),
        }
        
        try:
            import time
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            # Generate response
            outputs = self.model_instance.generate(
                **inputs,
                temperature=params['temperature'],
                max_new_tokens=params['max_new_tokens'],
                top_p=params['top_p'],
                repetition_penalty=params['repetition_penalty'],
                do_sample=params['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response (remove prompt if it's included)
            response = response[len(prompt):] if response.startswith(prompt) else response
            response = response.strip()
            
            execution_time = time.time() - start_time
            token_count = len(self.tokenizer.tokenize(response))
            
            return HFResult(
                response=response,
                tokens=token_count,
                time=execution_time,
                model=self.model.name
            )
            
        except Exception as e:
            raise ModelExecutionError(f"Failed to execute HuggingFace model: {e}")
    
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
            'family': self.model.family,
            'device': self.device if self.device else 'not loaded'
        }