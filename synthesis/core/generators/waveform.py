import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

from synthesis.core.generators.base import Generator, GeneratorConfig


@dataclass
class WaveformConfig(GeneratorConfig):
    """Configuration for waveform generation.

    Extends base config with waveform-specific settings.
    """
    waveform_type: Literal['sine', 'square', 'sawtooth', 'triangle'] = 'sine'
    duty_cycle: float = 0.5  # For pulse/square wave variation


class WaveformGenerator(Generator):
    """Generates basic periodic waveforms.

    Provides pure waveform generation with precise phase tracking
    and amplitude control. Each waveform type is optimized for
    minimal aliasing and consistent amplitude.
    """

    def __init__(self, config: Optional[WaveformConfig] = None):
        super().__init__(config or WaveformConfig())
        self.config: WaveformConfig  # Type hint for clarity

        # Map waveform type to generation method
        self._waveform_map = {
            'sine': self._generate_sine,
            'square': self._generate_square,
            'sawtooth': self._generate_sawtooth,
            'triangle': self._generate_triangle
        }

    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate samples of the configured waveform type.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Array of waveform samples.

        Raises:
            ValueError: If waveform_type is not recognized.
        """
        if self.config.waveform_type not in self._waveform_map:
            raise ValueError(f"Unknown waveform type: {self.config.waveform_type}")

        return self._waveform_map[self.config.waveform_type](num_samples)

    def _generate_sine(self, num_samples: int) -> np.ndarray:
        """Generate sine wave samples."""
        t = self.get_time_array(num_samples)
        return self.amplitude * np.sin(t)

    def _generate_square(self, num_samples: int) -> np.ndarray:
        """Generate square wave samples with variable duty cycle."""
        t = self.get_time_array(num_samples)
        cycle_position = np.mod(t / (2 * np.pi), 1.0)
        return self.amplitude * np.where(cycle_position < self.config.duty_cycle, 1.0, -1.0)

    def _generate_sawtooth(self, num_samples: int) -> np.ndarray:
        """Generate sawtooth wave samples."""
        t = self.get_time_array(num_samples)
        return self.amplitude * (2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi))))

    def _generate_triangle(self, num_samples: int) -> np.ndarray:
        """Generate triangle wave samples."""
        t = self.get_time_array(num_samples)
        return self.amplitude * (2 * np.abs(2 * (t / (2 * np.pi) -
                                                 np.floor(0.5 + t / (2 * np.pi)))) - 1)

    def set_waveform(self, waveform_type: str) -> None:
        """Change the waveform type.

        Args:
            waveform_type: One of 'sine', 'square', 'sawtooth', 'triangle'

        Raises:
            ValueError: If waveform_type is not recognized.
        """
        if waveform_type not in self._waveform_map:
            raise ValueError(f"Unknown waveform type: {waveform_type}")
        self.config.waveform_type = waveform_type
