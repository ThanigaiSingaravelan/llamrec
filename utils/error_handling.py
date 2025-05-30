# utils/error_handling.py
import logging
import traceback
from functools import wraps
from typing import Optional, Dict, Any, Callable
from enum import Enum


class ErrorLevel(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LlamarecException(Exception):
    """Base exception for LLAMAREC system"""

    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


class DataValidationError(LlamarecException):
    """Raised when data validation fails"""
    pass


class ModelConnectionError(LlamarecException):
    """Raised when model connection fails"""
    pass


class InsufficientDataError(LlamarecException):
    """Raised when insufficient data for processing"""
    pass


def handle_errors(
        fallback_return=None,
        error_level: ErrorLevel = ErrorLevel.ERROR,
        reraise: bool = False
):
    """Decorator for consistent error handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LlamarecException as e:
                # Our custom exceptions - handle gracefully
                logger = logging.getLogger(func.__module__)

                if error_level == ErrorLevel.CRITICAL:
                    logger.critical(f"{func.__name__} failed: {e}")
                    if reraise:
                        raise
                elif error_level == ErrorLevel.ERROR:
                    logger.error(f"{func.__name__} failed: {e}")
                else:
                    logger.warning(f"{func.__name__} warning: {e}")

                return fallback_return

            except Exception as e:
                # Unexpected exceptions - log full traceback
                logger = logging.getLogger(func.__module__)
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())

                if reraise:
                    raise LlamarecException(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        context={"function": func.__name__, "args": str(args)[:100]}
                    )

                return fallback_return

        return wrapper

    return decorator


# Enhanced validation utilities
class DataValidator:
    """Centralized data validation"""

    @staticmethod
    def validate_user_data(user_history: Dict) -> None:
        """Validate user history structure"""
        if not user_history:
            raise DataValidationError("User history is empty")

        required_fields = ['liked', 'count']
        for user_id, domains in user_history.items():
            for domain, data in domains.items():
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    raise DataValidationError(
                        f"Missing fields {missing_fields} for user {user_id}, domain {domain}"
                    )

    @staticmethod
    def validate_dataframe(df, required_columns: list, min_rows: int = 1) -> None:
        """Validate DataFrame structure"""
        if df.empty:
            raise DataValidationError("DataFrame is empty")

        if len(df) < min_rows:
            raise InsufficientDataError(f"DataFrame has {len(df)} rows, need at least {min_rows}")

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")


# Usage in your existing code:
class EnhancedCrossDomainPreprocessor:

    @handle_errors(fallback_return={}, error_level=ErrorLevel.CRITICAL, reraise=True)
    def load_data_with_validation(self) -> Dict[str, pd.DataFrame]:
        """Load data with proper validation"""
        data = {}

        for domain in self.domains:
            file_path = os.path.join(self.base_path, self.domain_file_mapping[domain])

            if not os.path.exists(file_path):
                raise DataValidationError(f"Required file not found: {file_path}")

            df = pd.read_csv(file_path)

            # Validate DataFrame
            DataValidator.validate_dataframe(
                df,
                required_columns=['reviewerID', 'asin', 'overall'],
                min_rows=100  # Minimum viable dataset size
            )

            data[domain] = df

        return data

    @handle_errors(fallback_return=set())
    def find_overlapping_users_safe(self, data: Dict[str, pd.DataFrame]) -> Set[str]:
        """Find overlapping users with error handling"""
        if len(data) < self.min_domains:
            raise InsufficientDataError(
                f"Need {self.min_domains} domains, got {len(data)}"
            )

        # Your existing logic here...
        return self.find_overlapping_users(data)