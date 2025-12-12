"""
CLI chat interface for the routing agent framework.

This module provides a user-friendly command-line interface for interacting
with the AI models through the routing framework.
"""

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_agent.core.detection import ModelDetector
from routing_agent.core.model_registry import ModelRegistry
from routing_agent.core.router import TaskRouter
from routing_agent.models.llama_cpp_runner import LlamaCppRunner
from routing_agent.models.hf_runner import HFRunner
from routing_agent.utils.exceptions import ModelExecutionError
from routing_agent.utils.logging import logger


class ChatCLI:
    """Main CLI chat interface."""
    
    def __init__(self):
        self.console = Console()
        self.detector = ModelDetector()
        self.registry = ModelRegistry()
        self.router = None
        self.debug_mode = False
    
    def initialize(self) -> None:
        """Initialize the routing framework."""
        self._print_header()
        
        # Detect models
        self._detect_and_register_models()
        
        # Initialize router
        if self.registry.get_all_models():
            self.router = TaskRouter(self.registry)
        else:
            self._print_no_models_message()
            sys.exit(1)
    
    def _print_header(self) -> None:
        """Print the welcome header."""
        header = Text("🤖 Routing Agent Framework", style="bold cyan")
        subtitle = Text("Intelligent Local AI Model Router", style="italic yellow")
        
        self.console.print(Panel.fit(header))
        self.console.print(Panel.fit(subtitle))
        self.console.print("Type '/help' for available commands\n")
    
    def _detect_and_register_models(self) -> None:
        """Detect and register available models."""
        self.console.print("🔍 Detecting available models...")
        
        try:
            # Scan for models
            detected_models = self.detector.scan_for_models()
            
            if detected_models:
                self.console.print(f"✅ Found {len(detected_models)} models")
                
                # Register models
                registered_models = self.registry.register_from_detection(self.detector)
                
                # Save detection results
                self.detector.save_detected_models("detected_models.yaml")
                self.registry.save_to_config()
                
                # Show registered models
                self._show_registered_models()
            else:
                self.console.print("⚠️  No models detected")
                self._print_no_models_message()
                
        except Exception as e:
            self.console.print(f"❌ Error detecting models: {e}")
            logger.error(f"Model detection failed: {e}")
    
    def _show_registered_models(self) -> None:
        """Show information about registered models."""
        models = self.registry.get_all_models()
        
        if not models:
            return
        
        self.console.print("\n📋 Registered Models:")
        
        for model in models:
            info = f"[bold]{model.name}[/bold] ([{model.backend}]) - {', '.join(model.capabilities)}"
            if model.size:
                info += f" • {model.size}"
            self.console.print(f"  • {info}")
        
        self.console.print()
    
    def _print_no_models_message(self) -> None:
        """Print message when no models are available."""
        message = """
❌ No AI models detected on your system.

To use the routing agent, you need to:

1. Download AI models (GGUF or HuggingFace format)
2. Place them in common model directories:
   - ~/.cache/huggingface/
   - ~/models/
   - /models/

Or specify custom model paths in the configuration.

Common model sources:
- https://huggingface.co/models
- https://github.com/ggerganov/llama.cpp
"""
        self.console.print(Panel.fit(message, title="No Models Found", style="red"))
    
    def _parse_command(self, input_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse user input for commands."""
        input_text = input_text.strip()
        
        if input_text.startswith('/'):
            parts = input_text[1:].split(' ', 1)
            command = parts[0].lower()
            argument = parts[1] if len(parts) > 1 else None
            return command, argument
        
        return None, None
    
    def _handle_command(self, command: str, argument: Optional[str]) -> bool:
        """Handle CLI commands. Returns True if should continue, False if should exit."""
        if command == 'help':
            self._show_help()
        elif command == 'exit' or command == 'quit':
            return False
        elif command == 'models':
            self._show_registered_models()
        elif command == 'debug':
            self.debug_mode = not self.debug_mode
            status = "ON" if self.debug_mode else "OFF"
            self.console.print(f"Debug mode: [bold]{status}[/bold]")
        elif command == 'task':
            if argument and argument in self.router.get_available_categories():
                self.console.print(f"Task override set to: [bold]{argument}[/bold]")
                return self._handle_chat_with_override(argument)
            else:
                available = ", ".join(self.router.get_available_categories())
                self.console.print(f"Available task types: {available}")
        elif command == 'clear':
            self.console.clear()
            self._print_header()
        else:
            self.console.print(f"Unknown command: /{command}")
            self._show_help()
        
        return True
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
📖 Available Commands:

  /help          - Show this help message
  /exit, /quit   - Exit the chat
  /models        - Show available models
  /debug         - Toggle debug mode
  /task <type>   - Override task type (coding, reasoning, etc.)
  /clear         - Clear the screen

Just type your message to chat with the AI!
"""
        self.console.print(help_text)
    
    def _handle_chat_with_override(self, task_type: str) -> bool:
        """Handle chat with task type override."""
        self.console.print(f"🎯 Task type override: [bold]{task_type}[/bold]")
        
        while True:
            try:
                user_input = Prompt.ask("💬 You", console=self.console)
                
                if not user_input.strip():
                    continue
                
                command, argument = self._parse_command(user_input)
                if command:
                    if command in ['exit', 'quit']:
                        return False
                    elif command == 'clear':
                        self.console.clear()
                        self._print_header()
                        self.console.print(f"🎯 Task type override: [bold]{task_type}[/bold]")
                        continue
                    else:
                        self._handle_command(command, argument)
                        continue
                
                # Route and execute the task
                self._execute_task(user_input, task_override=task_type)
                
            except KeyboardInterrupt:
                self.console.print("\n⏹️  Chat ended")
                return True
            except EOFError:
                self.console.print("\n⏹️  Chat ended")
                return True
    
    def _execute_task(self, task_text: str, task_override: Optional[str] = None) -> None:
        """Execute a task through the routing framework."""
        try:
            # Route the task
            routing_decision = self.router.route_task(task_text, task_override)
            
            # Get the model
            model = self.registry.get_model_by_id(routing_decision.model_id)
            if not model:
                raise ModelExecutionError(f"Model {routing_decision.model_id} not found")
            
            if self.debug_mode:
                self.console.print(f"🔍 [Debug] Routing decision: {routing_decision.reason}")
                self.console.print(f"🔍 [Debug] Selected model: {model.name} ({model.backend})")
            
            # Execute the model
            if model.backend == 'llama.cpp':
                runner = LlamaCppRunner(model)
                result = runner.execute(task_text)
            elif model.backend == 'huggingface':
                runner = HFRunner(model)
                result = runner.execute(task_text)
            else:
                raise ModelExecutionError(f"Unsupported backend: {model.backend}")
            
            # Display the result
            self._display_result(result)
            
        except Exception as e:
            self.console.print(f"❌ Error executing task: {e}")
            logger.error(f"Task execution failed: {e}")
    
    def _display_result(self, result: Any) -> None:
        """Display the result from model execution."""
        # Format the response
        response_panel = Panel.fit(
            Markdown(result.response),
            title=f"🤖 {result.model}",
            border_style="green"
        )
        
        self.console.print(response_panel)
        
        if self.debug_mode:
            debug_info = f"Tokens: {result.tokens} | Time: {result.time:.2f}s"
            self.console.print(f"[italic grey]{debug_info}[/italic grey]")
        
        self.console.print("─" * self.console.width)
    
    def start_chat(self) -> None:
        """Start the main chat loop."""
        self.initialize()
        
        if not self.router:
            return
        
        self.console.print("🚀 Ready! Type your message or /help for commands.")
        
        while True:
            try:
                user_input = Prompt.ask("💬 You", console=self.console)
                
                if not user_input.strip():
                    continue
                
                # Check for commands
                command, argument = self._parse_command(user_input)
                if command:
                    should_continue = self._handle_command(command, argument)
                    if not should_continue:
                        break
                    continue
                
                # Regular chat message
                self._execute_task(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n⏹️  Chat ended")
                break
            except EOFError:
                self.console.print("\n⏹️  Chat ended")
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}")
                logger.error(f"Chat error: {e}")
        
        self.console.print("👋 Goodbye!")


def main() -> None:
    """Main entry point for the CLI chat."""
    try:
        chat = ChatCLI()
        chat.start_chat()
    except Exception as e:
        console = Console()
        console.print(f"❌ Fatal error: {e}")
        logger.error(f"CLI failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()