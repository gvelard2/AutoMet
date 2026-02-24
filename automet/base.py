from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    """Abstract base class for all AutoMet analyzer modules.

    All analyzers must implement load_data(), run(), and save_output().
    """

    @abstractmethod
    def load_data(self):
        """Load raw data from the source file or directory.

        Must return self to support method chaining.
        """

    @abstractmethod
    def run(self):
        """Execute the full analysis pipeline end-to-end.

        Must return self to support method chaining.
        """

    @abstractmethod
    def save_output(self, filename, dpi=150):
        """Save the primary output figure to a PNG file.

        Args:
            filename: output file name (e.g. 'result.png').
            dpi: image resolution (default: 150).
        """
