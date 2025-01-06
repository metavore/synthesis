import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
import logging

from synthesis.core.generators.base import Generator, GeneratorConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class NoiseConfig(GeneratorConfig):
    """Configuration for noise generation.

    Note: frequency from base config is used for modulation/filtering.
    """
    noise_type: Literal['white', 'pink', 'brown'] = 'white'
    num_octaves: int = 8  # For pink noise approximation
    seed: Optional[int] = None


class NoiseGenerator(Generator):
    """Generates various types of noise with consistent characteristics.

    Implements several noise colors through different algorithms:
    - White noise: Uniform random distribution
    - Pink noise: 1/f spectrum using Voss-McCartney algorithm
    - Brown noise: Integrated white noise
    - Filtered noise: Band-limited white noise
    """

    def __init__(self, config: Optional[NoiseConfig] = None):
        super().__init__(config or NoiseConfig())
        self.config: NoiseConfig  # Type hint for clarity
        self._rng = np.random.RandomState(self.config.seed)

        # State for brown noise integration
        self._last_value = 0.0


    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate noise samples of the configured type.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Array of noise samples.
        """
        generators = {
            'white': self._generate_white,
            'pink': self._generate_pink,
            'brown': self._generate_brown,
        }

        if self.config.noise_type not in generators:
            valid_types = ", ".join(generators.keys())
            raise ValueError(f"Unknown noise type: {self.config.noise_type}. Valid types: {valid_types}")

        noise = generators[self.config.noise_type](num_samples)
        return self.amplitude * noise

    def _generate_white(self, num_samples: int) -> np.ndarray:
        """Generate white noise samples."""
        return self._rng.uniform(-1.0, 1.0, size=num_samples)

    def _generate_pink(self, num_samples: int) -> np.ndarray:
        """Generate pink noise using Voss-McCartney algorithm."""
        # Initialize array for each octave
        octaves = [self._generate_white(num_samples)]

        # Generate and sum octaves
        for i in range(1, self.config.num_octaves):
            # Each octave has half as many new values
            octave_size = num_samples // (2 ** i)
            values = self._rng.randn(octave_size)

            # Stretch values to full length
            repeated = np.repeat(values, 2 ** i)
            # Pad if necessary
            if len(repeated) < num_samples:
                repeated = np.pad(repeated, (0, num_samples - len(repeated)), 'edge')
            else:
                repeated = repeated[:num_samples]

            octaves.append(repeated)

        # Sum all octaves and normalize
        pink = np.sum(octaves, axis=0)
        epsilon = 1e-10
        pink /= (np.max(np.abs(pink)) + epsilon)
        return pink

    def _generate_brown(self, num_samples: int) -> np.ndarray:
        """Generate brown noise through integration."""
        white = self._generate_white(num_samples)

        # Integrate white noise with decay
        brown = np.zeros(num_samples)
        decay = 0.99  # Prevents drift

        brown[0] = self._last_value * decay + white[0]
        for i in range(1, num_samples):
            brown[i] = brown[i - 1] * decay + white[i]

        self._last_value = brown[-1]

        # Normalize
        return brown / np.max(np.abs(brown))

    def set_seed(self, seed: Optional[int]) -> None:
        """Set or reset the random seed."""
        self.config.seed = seed
        self._rng = np.random.RandomState(seed)
