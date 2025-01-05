import pytest
import numpy as np
from synthesis.core.generators.base import Generator, GeneratorConfig


def test_generator_config():
    """Test generator configuration initialization and validation."""
    config = GeneratorConfig(frequency=440.0, phase=0.0, sample_rate=44100)
    assert config.frequency == 440.0
    assert config.phase == 0.0
    assert config.sample_rate == 44100


def test_generator_time_array():
    """Test time array generation and phase tracking."""
    gen = Generator(GeneratorConfig(frequency=1.0, sample_rate=1000))

    # Generate two consecutive time arrays
    t1 = gen.get_time_array(500)
    t2 = gen.get_time_array(500)

    # Check array properties
    assert len(t1) == 500
    assert len(t2) == 500

    # Check phase continuity
    combined = np.concatenate([t1, t2])
    assert np.allclose(np.diff(combined), 2 * np.pi / 1000)


def test_amplitude_control():
    """Test thread-safe amplitude handling."""
    gen = Generator()

    # Test amplitude setting and limits
    gen.set_amplitude(0.5)
    assert gen.amplitude == 0.5

    gen.set_amplitude(1.5)  # Should clip to 1.0
    assert gen.amplitude == 1.0

    gen.set_amplitude(-0.5)  # Should clip to 0.0
    assert gen.amplitude == 0.0


def test_base_generate_samples():
    """Test that base generator raises NotImplementedError."""
    gen = Generator()
    with pytest.raises(NotImplementedError):
        gen.generate_samples(100)


if __name__ == '__main__':
    pytest.main(['-v'])
