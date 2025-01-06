import pytest
import numpy as np
from synthesis.core.generators.waveform import WaveformGenerator, WaveformConfig


def test_waveform_config():
    """Test waveform configuration."""
    config = WaveformConfig(frequency=440.0, waveform_type='sine')
    assert config.frequency == 440.0
    assert config.waveform_type == 'sine'
    assert config.duty_cycle == 0.5


def test_sine_wave():
    """Test sine wave generation."""
    gen = WaveformGenerator(WaveformConfig(
        frequency=1.0,  # 1 Hz for easy testing
        sample_rate=1000,
        waveform_type='sine'
    ))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(1000)  # One complete cycle

    assert len(samples) == 1000
    assert np.isclose(samples[0], 0, atol=1e-10)  # Starts at zero
    assert np.isclose(samples[250], 1, atol=1e-10)  # Peak at quarter cycle
    assert np.isclose(samples[750], -1, atol=1e-10)  # Trough at three quarters


def test_square_wave():
    """Test square wave generation and duty cycle."""
    gen = WaveformGenerator(WaveformConfig(
        frequency=1.0,
        sample_rate=1000,
        waveform_type='square',
        duty_cycle=0.3
    ))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(1000)

    # Count high and low samples
    high_samples = np.sum(samples > 0)
    assert np.isclose(high_samples / 1000, 0.3, rtol=1e-2)
    assert np.all(np.abs(samples) == 1.0)


def test_sawtooth_wave():
    """Test sawtooth wave generation."""
    gen = WaveformGenerator(WaveformConfig(
        frequency=1.0,
        sample_rate=1000,
        waveform_type='sawtooth'
    ))
    gen.set_amplitude(1.0)

    samples = gen.generate_samples(10000)

    assert len(samples) == 10000
    assert np.isclose(np.min(samples), -1.0)
    assert np.isclose(np.max(samples), 1.0)

    # Test for monotonic increase within cycle
    half_cycle = samples[:500]
    assert np.all(np.diff(half_cycle) > 0)


def test_triangle_wave():
    """Test triangle wave generation."""
    gen = WaveformGenerator(WaveformConfig(
        frequency=1.0,
        sample_rate=1000,
        waveform_type='triangle'
    ))
    gen.set_amplitude(1.0)  # Explicitly set amplitude

    # Generate a single cycle of the triangle wave
    samples_per_cycle = gen.config.sample_rate // gen.config.frequency
    wave = gen.generate_samples(samples_per_cycle)

    # Check basic properties
    assert len(wave) == samples_per_cycle
    assert -1.0 <= wave.min() <= -0.99  # Check minimum value
    assert 0.99 <= wave.max() <= 1.0  # Check maximum value

    # Symmetry test: Compare values on either side of the midpoint
    midpoint = int(samples_per_cycle // 2)
    for i in range(midpoint):
        assert np.isclose(wave[i], -wave[midpoint + i], rtol=1e-2)

    # Ensure all extrema are hit
    assert any(np.isclose(wave, 1.0, rtol=1e-2))
    assert any(np.isclose(wave, -1.0, rtol=1e-2))



def test_invalid_waveform():
    """Test error handling for invalid waveform type."""
    gen = WaveformGenerator()
    with pytest.raises(ValueError):
        gen.set_waveform('invalid_type')


if __name__ == '__main__':
    pytest.main(['-v'])
