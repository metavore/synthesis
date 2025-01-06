import pytest
import logging
import numpy as np
from synthesis.core.generators.noise import NoiseGenerator, NoiseConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_noise_config():
    """Test noise configuration and validation."""
    config = NoiseConfig(
        frequency=1000.0,  # For filtered noise
        noise_type='pink',
        seed=42
    )
    assert config.noise_type == 'pink'
    assert config.seed == 42


def test_white_noise():
    """Test white noise characteristics."""
    gen = NoiseGenerator(NoiseConfig(noise_type='white', seed=42))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(10000)

    assert len(samples) == 10000
    assert -1.0 <= samples.min() <= -0.9  # Should hit near extremes
    assert 0.9 <= samples.max() <= 1.0
    assert -0.1 < np.mean(samples) < 0.1  # Should average near zero

    # Test reproducibility with seed
    gen2 = NoiseGenerator(NoiseConfig(noise_type='white', seed=42))
    gen2.set_amplitude(1.0)
    samples2 = gen2.generate_samples(10000)
    assert np.array_equal(samples, samples2)


def test_pink_noise():
    """Test pink noise spectral characteristics."""
    gen = NoiseGenerator(NoiseConfig(
        noise_type='pink',
        seed=42,
        num_octaves=8
    ))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(16384)  # Power of 2 for FFT

    # Compute power spectrum
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples))

    # Exclude DC and very high frequencies
    mask = (freqs > 0.01) & (freqs < 0.49)

    # Fit slope in log-log space (should be close to -1 for pink noise)
    poly = np.polyfit(
        np.log10(freqs[mask]),
        np.log10(spectrum[mask]),
        deg=1
    )
    logger.debug(f"Pink noise slope: {poly[0]}")
    assert -1.4 < poly[0] < -0.8, f"Pink noise slope {poly[0]} is out of range."


def test_brown_noise():
    """Test brown noise characteristics."""
    gen = NoiseGenerator(NoiseConfig(noise_type='brown'))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(10000)

    # Check for continuity
    derivatives = np.diff(samples)
    assert np.std(derivatives) < np.std(samples)  # Less variation in changes

    # Should have more low frequency content than pink noise
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    low_power = np.sum(spectrum[:len(spectrum) // 10])
    high_power = np.sum(spectrum[len(spectrum) // 10:])
    assert low_power > high_power


def test_seed_control():
    """Test seed setting and resetting for reproducibility."""
    gen = NoiseGenerator(NoiseConfig(noise_type='white'))
    gen.set_amplitude(1)

    # Generate sequences with different seeds
    gen.set_seed(42)
    samples1 = gen.generate_samples(1000)
    gen.set_seed(99)
    samples2 = gen.generate_samples(1000)

    # Verify that sequences with different seeds are distinct
    assert not np.array_equal(samples1, samples2), "Different seeds produced identical sequences."

    # Reset to the first seed and verify reproducibility
    gen.set_seed(42)
    samples3 = gen.generate_samples(1000)
    assert np.array_equal(samples1, samples3), "Reusing the same seed did not reproduce the sequence."

    # Verify repeated calls with the same seed produce different sequences
    gen.set_seed(42)
    samples4 = gen.generate_samples(1000)
    samples5 = gen.generate_samples(1000)
    assert not np.array_equal(samples4, samples5), "Repeated calls with the same seed produced identical sequences."

    logger.debug("Seed control tests passed successfully.")



if __name__ == '__main__':
    pytest.main(['-v'])
