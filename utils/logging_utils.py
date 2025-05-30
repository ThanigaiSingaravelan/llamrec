# utils/logging_utils.py
import logging
import sys
from datetime import datetime
from pathlib import Path
import json
import time
from functools import wraps
from typing import Any, Dict


class StructuredLogger:
    """Enhanced logging with structured output and experiment tracking."""

    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Experiment tracking
        self.experiment_data = {
            'start_time': datetime.now().isoformat(),
            'events': [],
            'metrics': {},
            'errors': []
        }

    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.experiment_data['config'] = config
        self.logger.info(f"🚀 Experiment started with config: {config}")

    def log_step(self, step_name: str, metrics: Dict[str, Any] = None):
        """Log experiment step with optional metrics."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'metrics': metrics or {}
        }
        self.experiment_data['events'].append(event)

        msg = f"📊 Step: {step_name}"
        if metrics:
            msg += f" | Metrics: {metrics}"
        self.logger.info(msg)

    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log errors with context."""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context or {}
        }
        self.experiment_data['errors'].append(error_data)
        self.logger.error(f"❌ {error} | Context: {context}")

    def save_experiment_log(self, output_path: str):
        """Save complete experiment log."""
        self.experiment_data['end_time'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)

        self.logger.info(f"💾 Experiment log saved to {output_path}")


def log_execution_time(logger: StructuredLogger):
    """Decorator to log function execution time."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log_step(
                    f"{func.__name__}_completed",
                    {'execution_time': execution_time}
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_error(
                    f"Function {func.__name__} failed",
                    {'execution_time': execution_time, 'error': str(e)}
                )
                raise

        return wrapper

    return decorator

# Usage example:
# logger = StructuredLogger("llamarec_experiment")
#
# @log_execution_time(logger)
# def run_evaluation():
#     # Your evaluation code
#     pass