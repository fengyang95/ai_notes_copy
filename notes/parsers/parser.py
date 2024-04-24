"""Abstract base class for dependency file core."""
import logging
from abc import ABC, abstractmethod
from typing import List


class BaseFileParser(ABC):
    """Abstract base class for dependency file core."""

    def __init__(self) -> None:
        """Initializes the handler with given configuration."""
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def parse(self, content: str) -> List[str]:
        """Parses content of dependency file and returns list of dependencies."""
        ...

    def log_error(self, message: str):
        """Logs error message when parsing fails."""
        self._logger.error(f"Error parsing dependency file {message}")

    def handle_parsing_error(self, error: Exception) -> List[str]:
        """Standardized error handling for parsing exceptions."""
        self.log_error(str(error))
        return []
