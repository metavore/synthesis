import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for basic signal generator.

    Core settings that any signal generator needs.
    """
    frequency: float = 440.0
    phase: float = 0.0
    sample_rate: int = 44100


class Generator:
    """Base class for all signal generators.

    Provides core functionality for sample generation timing,
    amplitude control, and thread safety.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._lock = threading.Lock()
        self._phase = 0.0
        self._amplitude = 0.0

    def get_time_array(self, num_samples: int) -> np.ndarray:
        """Generate time points with phase tracking.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Time points array scaled by frequency and adjusted for phase.
        """
        t = 2 * np.pi * self.config.frequency * (
                np.arange(num_samples) / self.config.sample_rate + self._phase
        )
        self._phase += num_samples / self.config.sample_rate
        return t

    def set_amplitude(self, value: float) -> None:
        """Thread-safe amplitude adjustment.

        Args:
            value: New amplitude value (0.0 to 1.0).
        """
        with self._lock:
            self._amplitude = np.clip(value, 0.0, 1.0)

    @property
    def amplitude(self) -> float:
        """Thread-safe amplitude access."""
        with self._lock:
            return self._amplitude

    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate audio samples.

        This method should be overridden by subclasses.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Array of audio samples.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement generate_samples")