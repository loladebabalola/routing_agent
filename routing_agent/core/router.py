"""
Main router for intelligent task distribution.

This module handles the routing of tasks to the most appropriate available models
based on task classification and model capabilities.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
import os

from .model_registry import ModelRegistry, RegisteredModel
from .task_classifier import TaskClassifier, TaskClassification
from ..utils.exceptions import RoutingError
from ..utils.logging import logger


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    model_id: str
    model_name: str
    reason: str
    confidence: float


class TaskRouter:
    """Main router for distributing tasks to appropriate models."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.task_classifier = TaskClassifier()
        self.routing_rules = self._load_routing_rules()
    
    def _load_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load routing rules from configuration."""
        default_rules = {
            'general': {
                'priority': 1,
                'fallback': True,
                'description': 'General purpose tasks'
            },
            'coding': {
                'priority': 2,
                'fallback': 'general',
                'description': 'Programming and coding tasks'
            },
            'reasoning': {
                'priority': 3,
                'fallback': 'general',
                'description': 'Logical reasoning and analysis tasks'
            },
            'advanced': {
                'priority': 4,
                'fallback': 'reasoning',
                'description': 'Advanced and complex tasks'
            },
            'lightweight': {
                'priority': 0,
                'fallback': 'general',
                'description': 'Simple and quick tasks'
            }
        }
        
        try:
            config_path = "config/routing_rules.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    custom_rules = yaml.safe_load(f)
                    if custom_rules:
                        # Merge custom rules with defaults
                        default_rules.update(custom_rules)
        except Exception as e:
            logger.warning(f"Failed to load routing rules: {e}")
        
        return default_rules
    
    def route_task(self, task_text: str, override_category: Optional[str] = None) -> RoutingDecision:
        """Route a task to the most appropriate model."""
        # Classify the task
        if override_category:
            classification = TaskClassification(
                primary_category=override_category,
                secondary_categories=['general'],
                confidence=1.0
            )
            logger.info(f"Task routing override: {override_category}")
        else:
            classification = self.task_classifier.classify_task(task_text)
            logger.info(f"Task classified as: {classification.primary_category} (confidence: {classification.confidence:.2f})")
        
        # Find the best model for this task
        best_model, reason = self._find_best_model(classification)
        
        if not best_model:
            raise RoutingError(f"No suitable model found for task category: {classification.primary_category}")
        
        return RoutingDecision(
            model_id=best_model.id,
            model_name=best_model.name,
            reason=reason,
            confidence=classification.confidence
        )
    
    def _find_best_model(self, classification: TaskClassification) -> Tuple[Optional[RegisteredModel], str]:
        """Find the best model for a given task classification."""
        primary_category = classification.primary_category
        secondary_categories = classification.secondary_categories
        
        # Try to find models for the primary category
        primary_models = self.model_registry.get_models_by_capability(primary_category)
        
        if primary_models:
            # Sort by priority (higher priority first)
            primary_models.sort(key=lambda x: x.priority, reverse=True)
            best_model = primary_models[0]
            return best_model, f"Best match for {primary_category} tasks"
        
        # Try secondary categories
        for secondary_category in secondary_categories:
            secondary_models = self.model_registry.get_models_by_capability(secondary_category)
            if secondary_models:
                secondary_models.sort(key=lambda x: x.priority, reverse=True)
                best_model = secondary_models[0]
                return best_model, f"Fallback to {secondary_category} (no {primary_category} models available)"
        
        # Try general models as ultimate fallback
        general_models = self.model_registry.get_models_by_capability('general')
        if general_models:
            general_models.sort(key=lambda x: x.priority, reverse=True)
            best_model = general_models[0]
            return best_model, f"Fallback to general models (no specialized models available)"
        
        return None, "No suitable models available"
    
    def get_available_categories(self) -> List[str]:
        """Get all available task categories."""
        return self.task_classifier.get_task_categories()
    
    def get_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get the current routing rules."""
        return self.routing_rules
    
    def update_routing_rules(self, new_rules: Dict[str, Dict[str, Any]]) -> None:
        """Update routing rules."""
        self.routing_rules.update(new_rules)
        
        # Save to config
        try:
            config_path = "config/routing_rules.yaml"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(new_rules, f, sort_keys=False)
                
            logger.info("Updated routing rules")
        except Exception as e:
            logger.error(f"Failed to save routing rules: {e}")