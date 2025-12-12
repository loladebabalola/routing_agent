"""
Custom exceptions for the routing agent framework.
"""


class ModelDetectionError(Exception):
    """Raised when model detection fails."""
    pass


class ModelRegistrationError(Exception):
    """Raised when model registration fails."""
    pass


class RoutingError(Exception):
    """Raised when task routing fails."""
    pass


class ModelExecutionError(Exception):
    """Raised when model execution fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass