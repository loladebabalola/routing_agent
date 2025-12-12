"""
Task classifier for determining the type of AI task.

This module classifies user input to determine the most appropriate
model capabilities needed for the task.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TaskClassification:
    """Represents the classification of a task."""
    primary_category: str
    secondary_categories: List[str]
    confidence: float


class TaskClassifier:
    """Classifies tasks based on user input."""
    
    def __init__(self):
        # Define task patterns and their categories
        self.task_patterns = {
            'coding': {
                'patterns': [
                    r'\b(code|program|script|function|class|algorithm|debug|bug|error|syntax)\b',
                    r'\b(python|java|javascript|c\+\+|rust|go|typescript|php|ruby)\b',
                    r'\b(library|framework|api|sd?k|dependency)\b',
                    r'\b(git|github|version control|repository)\b',
                    r'\b(compile|build|deploy|test|unit test|integration test)\b',
                ],
                'keywords': [
                    'write code', 'fix code', 'review code', 'optimize code',
                    'debug', 'implement', 'algorithm', 'data structure',
                    'software', 'application', 'web app', 'mobile app'
                ]
            },
            'reasoning': {
                'patterns': [
                    r'\b(analyze|evaluate|compare|explain|reason|logic|deduce)\b',
                    r'\b(math|mathematics|calculus|algebra|statistics)\b',
                    r'\b(physics|chemistry|biology|science)\b',
                    r'\b(philosophy|ethics|logic|argument)\b',
                ],
                'keywords': [
                    'analyze this', 'explain why', 'compare and contrast',
                    'what is the logic', 'solve this problem', 'mathematical proof',
                    'scientific explanation', 'philosophical argument'
                ]
            },
            'advanced': {
                'patterns': [
                    r'\b(advanced|complex|complicated|sophisticated|expert)\b',
                    r'\b(research|study|analysis|theory|hypothesis)\b',
                    r'\b(neural network|deep learning|ai|machine learning)\b',
                ],
                'keywords': [
                    'advanced mathematics', 'complex analysis', 'research paper',
                    'scientific study', 'theoretical physics', 'machine learning model',
                    'neural network architecture', 'expert level'
                ]
            },
            'lightweight': {
                'patterns': [
                    r'\b(simple|basic|easy|quick|fast|small)\b',
                    r'\b(chat|conversation|casual|informal)\b',
                ],
                'keywords': [
                    'quick question', 'simple answer', 'casual chat',
                    'basic information', 'easy explanation', 'fast response'
                ]
            }
        }
    
    def classify_task(self, task_text: str) -> TaskClassification:
        """Classify a task based on the input text."""
        task_text_lower = task_text.lower()
        
        # Initialize scores for each category
        category_scores = {category: 0.0 for category in self.task_patterns.keys()}
        
        # Check patterns for each category
        for category, patterns_data in self.task_patterns.items():
            # Check regex patterns
            for pattern in patterns_data['patterns']:
                if re.search(pattern, task_text_lower, re.IGNORECASE):
                    category_scores[category] += 1.0
            
            # Check keywords
            for keyword in patterns_data['keywords']:
                if keyword in task_text_lower:
                    category_scores[category] += 0.5
        
        # Find the primary category (highest score)
        if category_scores:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
            primary_score = category_scores[primary_category]
        else:
            primary_category = 'general'
            primary_score = 0.0
        
        # Determine secondary categories (scores > 0 but less than primary)
        secondary_categories = []
        for category, score in category_scores.items():
            if category != primary_category and score > 0:
                secondary_categories.append(category)
        
        # Calculate confidence (normalized score)
        total_possible = len(self.task_patterns)
        confidence = min(primary_score / (total_possible * 1.5), 1.0)  # Cap at 1.0
        
        # Always include 'general' as a fallback
        if primary_category != 'general':
            secondary_categories.append('general')
        
        return TaskClassification(
            primary_category=primary_category,
            secondary_categories=list(set(secondary_categories)),
            confidence=confidence
        )
    
    def get_task_categories(self) -> List[str]:
        """Get all available task categories."""
        return list(self.task_patterns.keys()) + ['general']