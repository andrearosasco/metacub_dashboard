"""
Keyboard command interface for MetaCub Dashboard with persistent status display.
Uses a simpler approach with clear status area separation.
"""
import sys
import select
import termios
import tty
import os
from typing import Optional
import threading
import time


class KeyboardInterface:
    """Keyboard interface with persistent status display using terminal control."""
    
    def __init__(self, window_name: str = "MetaCub Dashboard Controls"):
        self.window_name = window_name
        self.old_settings = None
        self.last_status = ""
        self.current_episode_state = "STOPPED"
        self.is_active = True
        self.app_output_buffer = []
        self.max_app_lines = 15  # Keep last 15 lines of app output
        self.display_lock = threading.Lock()  # Synchronize display updates
        self._setup_terminal()
        
        # Start status update thread with longer interval
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()
    
    def _setup_terminal(self):
        """Setup terminal for non-blocking input."""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
    
    def _status_update_loop(self):
        """Background thread to periodically refresh the entire display."""
        while self.is_active:
            time.sleep(5.0)  # Update every 5 seconds instead of 1
            if self.is_active:  # Check again after sleep
                with self.display_lock:
                    self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the entire terminal display with status and app output."""
        if not sys.stdin.isatty():
            return
            
        # Clear screen and move to top - use direct sys.stdout.write to avoid recursion
        os.system('clear')
        
        # Print header and status using direct sys.stdout.write
        sys.stdout.write("="*80 + "\n")
        sys.stdout.write("  MetaCub Dashboard - Episode Control\n")
        sys.stdout.write("="*80 + "\n")
        sys.stdout.write("  Commands: 's'=start episode | 'e'=end episode | 'r'=reset | 'q'=quit\n")
        sys.stdout.write("="*80 + "\n")
        sys.stdout.write(f"  Status: {self.current_episode_state} | {self.last_status}\n")
        sys.stdout.write("="*80 + "\n")
        sys.stdout.write("\n")
        
        # Print recent application output
        if self.app_output_buffer:
            sys.stdout.write("Recent Output:\n")
            sys.stdout.write("-" * 40 + "\n")
            for line in self.app_output_buffer[-self.max_app_lines:]:
                sys.stdout.write(f"  {line}\n")
            sys.stdout.write("-" * 40 + "\n")
        else:
            sys.stdout.write("No application output yet...\n")
            sys.stdout.write("-" * 40 + "\n")
        
        sys.stdout.flush()
     
    def get_command(self, blocking: bool = False) -> Optional[str]:
        """
        Get keyboard command, optionally blocking until a command is received.
        
        Args:
            blocking: If True, wait until a command is received
            
        Returns:
            str or None: Command character or None if no key pressed (when non-blocking)
            Commands:
            - 's': start episode
            - 'e': end episode  
            - 'q': quit application
            - 'r': reset episode
        """
        if not sys.stdin.isatty():
            return None
        
        if blocking:
            # Blocking mode: wait until a command is received
            while True:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.01) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    
                    if key == 's':
                        return 'start'
                    elif key == 'e':
                        return 'end'
                    elif key == 'q':
                        return 'quit'
                    elif key == 'r':
                        return 'reset'
                    elif key == 'k':
                        return 'keep'
                    elif key == 'd':
                        return 'discard'
                    elif key == '\x03':  # Ctrl+C
                        return 'quit'
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
        else:
            # Non-blocking mode: check once and return
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                
                if key == 's':
                    return 'start'
                elif key == 'e':
                    return 'end'
                elif key == 'q':
                    return 'quit'
                elif key == 'r':
                    return 'reset'
                elif key == 'k':
                    return 'keep'
                elif key == 'd':
                    return 'discard'
                elif key == '\x03':  # Ctrl+C
                    return 'quit'
            
            return None
    
    def update_status(self, status: str):
        """Update the status message."""
        self.last_status = status
    
    def set_episode_state(self, state: str):
        """Update the episode state display."""
        self.current_episode_state = state
    
    def log_app_output(self, message: str):
        """
        Log application output to be displayed in the app area.
        This maintains a buffer of recent messages.
        """
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        with self.display_lock:
            self.app_output_buffer.append(formatted_message)
            
            # Keep only the most recent messages
            if len(self.app_output_buffer) > self.max_app_lines * 2:
                self.app_output_buffer = self.app_output_buffer[-self.max_app_lines:]
    
    def close(self):
        """Restore terminal settings and clean up."""
        self.is_active = False
        
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=1.0)
        
        if self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        # Use direct sys.stdout.write to avoid recursion
        sys.stdout.write("\nKeyboard interface closed.\n")
        sys.stdout.flush()


class StatusAwarePrinter:
    """
    A print replacement that logs to the keyboard interface instead of directly to stdout.
    This prevents application output from interfering with the status display.
    """
    
    def __init__(self, keyboard_interface: KeyboardInterface):
        self.keyboard_interface = keyboard_interface
    
    def print(self, *args, **kwargs):
        """Print replacement that logs to the keyboard interface."""
        # Convert all args to strings and join them
        message = ' '.join(str(arg) for arg in args)
        if message.strip():  # Only log non-empty messages
            self.keyboard_interface.log_app_output(message.strip())
            # Don't immediately refresh display to avoid recursion


def test_keyboard_interface():
    """Test the keyboard interface with persistent status display."""
    interface = KeyboardInterface()
    printer = StatusAwarePrinter(interface)
    
    # Use our custom printer instead of regular print
    printer.print("Testing keyboard interface with persistent status...")
    printer.print("The status area should remain visible at the top.")
    printer.print("Press 's', 'e', 'q', or 'r' keys. Press 'q' to exit test.")
    
    counter = 0
    try:
        import time
        while True:
            command = interface.get_command()
            if command:
                printer.print(f"Command received: {command}")
                interface.update_status(f"Last command: {command} at {time.strftime('%H:%M:%S')}")
                
                # Update episode state based on command
                if command == 'start':
                    interface.set_episode_state("RECORDING")
                elif command == 'end':
                    interface.set_episode_state("STOPPED")
                elif command == 'reset':
                    interface.set_episode_state("RESET")
                elif command == 'quit':
                    break
            
            # Simulate application output every few seconds
            counter += 1
            if counter % 50 == 0:  # Every ~5 seconds at 0.1s sleep
                printer.print(f"Simulated app output #{counter//50}")
                
            time.sleep(0.1)  # Small delay to prevent busy loop
            
    except KeyboardInterrupt:
        sys.stdout.write("\nTest interrupted by user\n")
    finally:
        interface.close()
        sys.stdout.write("Keyboard interface test complete.\n")


if __name__ == "__main__":
    test_keyboard_interface()
