"""
Unit tests for task classifier.
"""

import pytest
from routing_agent.core.task_classifier import TaskClassifier, TaskClassification


def test_task_classifier_initialization():
    """Test TaskClassifier initialization."""
    classifier = TaskClassifier()
    
    # Should have task patterns defined
    assert hasattr(classifier, 'task_patterns')
    assert isinstance(classifier.task_patterns, dict)
    
    # Should have expected categories
    expected_categories = ['coding', 'reasoning', 'advanced', 'lightweight']
    for category in expected_categories:
        assert category in classifier.task_patterns


def test_get_task_categories():
    """Test getting available task categories."""
    classifier = TaskClassifier()
    categories = classifier.get_task_categories()
    
    # Should include all expected categories plus general
    expected = ['coding', 'reasoning', 'advanced', 'lightweight', 'general']
    assert set(categories) == set(expected)


def test_classify_coding_task():
    """Test classification of coding tasks."""
    classifier = TaskClassifier()
    
    # Test various coding-related inputs
    coding_inputs = [
        "Write a Python function to sort a list",
        "Debug this JavaScript code",
        "How do I implement a binary search algorithm?",
        "Fix this syntax error in my Python script",
        "What's the best way to structure a React component?"
    ]
    
    for input_text in coding_inputs:
        result = classifier.classify_task(input_text)
        assert result.primary_category == 'coding'
        assert result.confidence > 0.5
        assert 'general' in result.secondary_categories


def test_classify_reasoning_task():
    """Test classification of reasoning tasks."""
    classifier = TaskClassifier()
    
    reasoning_inputs = [
        "Explain the difference between quantum computing and classical computing",
        "Analyze this mathematical proof",
        "Compare and contrast these two philosophical theories",
        "What is the logical fallacy in this argument?",
        "Solve this complex physics problem"
    ]
    
    for input_text in reasoning_inputs:
        result = classifier.classify_task(input_text)
        assert result.primary_category == 'reasoning'
        assert result.confidence > 0.5


def test_classify_advanced_task():
    """Test classification of advanced tasks."""
    classifier = TaskClassifier()
    
    advanced_inputs = [
        "Explain advanced quantum mechanics concepts",
        "Analyze this complex neural network architecture",
        "What are the latest developments in AI research?",
        "Expert-level mathematical analysis required",
        "Sophisticated machine learning model optimization"
    ]
    
    for input_text in advanced_inputs:
        result = classifier.classify_task(input_text)
        assert result.primary_category == 'advanced'
        assert result.confidence > 0.4  # Advanced tasks might have lower confidence


def test_classify_lightweight_task():
    """Test classification of lightweight tasks."""
    classifier = TaskClassifier()
    
    lightweight_inputs = [
        "Quick question about the weather",
        "Simple explanation needed",
        "Casual chat about movies",
        "Basic information request",
        "Fast response required"
    ]
    
    for input_text in lightweight_inputs:
        result = classifier.classify_task(input_text)
        assert result.primary_category == 'lightweight'
        assert result.confidence > 0.3


def test_classify_general_task():
    """Test classification of general tasks."""
    classifier = TaskClassifier()
    
    general_inputs = [
        "Hello, how are you?",
        "Tell me about yourself",
        "What's the capital of France?",
        "Generic question about life",
        "Random conversation starter"
    ]
    
    for input_text in general_inputs:
        result = classifier.classify_task(input_text)
        # General tasks should fall back to general category
        assert result.primary_category in ['general', 'lightweight']


def test_classification_confidence():
    """Test confidence scoring in classification."""
    classifier = TaskClassifier()
    
    # High confidence task
    high_conf_input = "Write a Python function with proper syntax and error handling"
    high_result = classifier.classify_task(high_conf_input)
    assert high_result.confidence > 0.7
    
    # Low confidence task (ambiguous)
    low_conf_input = "Tell me something interesting"
    low_result = classifier.classify_task(low_conf_input)
    assert low_result.confidence < 0.5


def test_task_classification_structure():
    """Test TaskClassification dataclass structure."""
    classifier = TaskClassifier()
    result = classifier.classify_task("Test input")
    
    assert isinstance(result, TaskClassification)
    assert hasattr(result, 'primary_category')
    assert hasattr(result, 'secondary_categories')
    assert hasattr(result, 'confidence')
    assert isinstance(result.secondary_categories, list)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_empty_input_classification():
    """Test classification of empty input."""
    classifier = TaskClassifier()
    result = classifier.classify_task("")
    
    # Empty input should default to general
    assert result.primary_category == 'general'
    assert result.confidence == 0.0


def test_mixed_category_detection():
    """Test detection of multiple categories in single input."""
    classifier = TaskClassifier()
    
    # Input that could match multiple categories
    mixed_input = "Write a Python function to analyze this complex mathematical problem"
    result = classifier.classify_task(mixed_input)
    
    # Should have coding as primary
    assert result.primary_category == 'coding'
    
    # Should have reasoning as secondary
    assert 'reasoning' in result.secondary_categories or 'advanced' in result.secondary_categories
    
    # Should have reasonable confidence
    assert result.confidence > 0.5


def test_case_insensitive_classification():
    """Test that classification is case insensitive."""
    classifier = TaskClassifier()
    
    # Test with different cases
    inputs = [
        "WRITE A PYTHON FUNCTION",
        "write a python function",
        "Write A Python Function",
        "wRiTe a pYtHoN fUnCtIoN"
    ]
    
    for input_text in inputs:
        result = classifier.classify_task(input_text)
        assert result.primary_category == 'coding'
        assert result.confidence > 0.7


def test_special_characters_handling():
    """Test handling of special characters in input."""
    classifier = TaskClassifier()
    
    # Input with special characters
    special_input = "Write a Python function! Can you help? (urgent)"
    result = classifier.classify_task(special_input)
    
    assert result.primary_category == 'coding'
    assert result.confidence > 0.6