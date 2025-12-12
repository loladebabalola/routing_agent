"""
Basic tests for CLI functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from routing_agent.cli.chat import ChatCLI


def test_cli_initialization():
    """Test CLI initialization."""
    cli = ChatCLI()
    
    assert cli.console is not None
    assert cli.detector is not None
    assert cli.registry is not None
    assert cli.router is None  # Not initialized yet
    assert cli.debug_mode is False


def test_command_parsing():
    """Test command parsing functionality."""
    cli = ChatCLI()
    
    # Test valid commands
    command, arg = cli._parse_command("/help")
    assert command == "help"
    assert arg is None
    
    command, arg = cli._parse_command("/task coding")
    assert command == "task"
    assert arg == "coding"
    
    command, arg = cli._parse_command("/debug")
    assert command == "debug"
    assert arg is None
    
    # Test non-command input
    command, arg = cli._parse_command("Hello world")
    assert command is None
    assert arg is None
    
    # Test empty input
    command, arg = cli._parse_command("")
    assert command is None
    assert arg is None
    
    # Test command with spaces
    command, arg = cli._parse_command("/task advanced reasoning")
    assert command == "task"
    assert arg == "advanced reasoning"


def test_debug_mode_toggle():
    """Test debug mode toggling."""
    cli = ChatCLI()
    
    # Initial state
    assert cli.debug_mode is False
    
    # Toggle on
    cli._handle_command("debug", None)
    assert cli.debug_mode is True
    
    # Toggle off
    cli._handle_command("debug", None)
    assert cli.debug_mode is False


def test_help_command():
    """Test help command handling."""
    cli = ChatCLI()
    
    # Should not raise exception
    result = cli._handle_command("help", None)
    assert result is True  # Should continue


def test_exit_command():
    """Test exit command handling."""
    cli = ChatCLI()
    
    # Should return False to exit
    result = cli._handle_command("exit", None)
    assert result is False
    
    result = cli._handle_command("quit", None)
    assert result is False


def test_models_command():
    """Test models command handling."""
    cli = ChatCLI()
    
    # Should not raise exception with empty registry
    result = cli._handle_command("models", None)
    assert result is True


def test_unknown_command():
    """Test handling of unknown commands."""
    cli = ChatCLI()
    
    # Should handle unknown command gracefully
    result = cli._handle_command("unknown", None)
    assert result is True


def test_task_command_validation():
    """Test task command with valid and invalid arguments."""
    cli = ChatCLI()
    
    # Mock router to avoid initialization
    mock_router = MagicMock()
    mock_router.get_available_categories.return_value = ["coding", "reasoning"]
    cli.router = mock_router
    
    # Test with valid task type
    with patch.object(cli, '_handle_chat_with_override', return_value=True):
        result = cli._handle_command("task", "coding")
        assert result is True
    
    # Test with invalid task type
    result = cli._handle_command("task", "invalid")
    assert result is True  # Should show available types and continue


def test_clear_command():
    """Test clear command handling."""
    cli = ChatCLI()
    
    # Should not raise exception
    result = cli._handle_command("clear", None)
    assert result is True