import traceback
from .logger import get_logger

logger = get_logger(__name__)

# Define custom exceptions for different parts of the application
class ModelLoadingError(Exception):
    """Raised when a model fails to load properly."""

class PreprocessingError(Exception):
    """Raised when an error occurs during data preprocessing."""

class PredictionError(Exception):
    """Raised when there is an issue with generating predictions from the models."""

class APIError(Exception):
    """Raised when there is an issue with the API call to the LLM or any external service."""


def log_exception(e: Exception, custom_message: str = ""):
    """
    Logs detailed information about an exception, including traceback.

    Args:
        e (Exception): The exception to log.
        custom_message (str): Optional custom message to provide additional context in the log.
    """
    exc_type, exc_value, exc_traceback = e.__class__, e, e.__traceback__
    trace_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"{custom_message}\nException type: {exc_type}\nTraceback:\n{trace_details}")
