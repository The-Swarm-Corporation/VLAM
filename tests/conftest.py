"""
Pytest configuration and shared fixtures for VLAM tests.
"""

import pytest
import torch
import numpy as np
from vlam.main import VisionLanguageActionModel, ActionConfig, RobotType


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def arm_config():
    """Standard arm configuration for testing."""
    return ActionConfig.get_arm_config()


@pytest.fixture
def humanoid_config():
    """Standard humanoid configuration for testing."""
    return ActionConfig.get_humanoid_config()


@pytest.fixture
def custom_config():
    """Custom robot configuration for testing."""
    return ActionConfig(
        robot_type=RobotType.MOBILE_MANIPULATOR,
        action_dim=10,
        action_bounds=(-1.5, 1.5),
        joint_names=[f"joint_{i}" for i in range(10)],
    )


@pytest.fixture
def small_model(arm_config):
    """Small model for quick testing."""
    return VisionLanguageActionModel(
        action_config=arm_config, d_model=128, n_layers=2, vocab_size=1000
    )


@pytest.fixture
def medium_model(arm_config):
    """Medium model for more comprehensive testing."""
    return VisionLanguageActionModel(
        action_config=arm_config, d_model=256, n_layers=4, vocab_size=5000
    )


@pytest.fixture
def sample_images():
    """Sample image tensors for testing."""
    return torch.randn(2, 5, 3, 224, 224)


@pytest.fixture
def sample_single_image():
    """Single image tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_actions(arm_config):
    """Sample action tensors for testing."""
    return torch.randn(2, 5, arm_config.action_dim)


@pytest.fixture
def sample_language_tokens():
    """Sample language tokens for testing."""
    return torch.randint(0, 1000, (2, 5))


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_training_data(sample_images, sample_actions, sample_language_tokens):
    """Mock training data batch."""
    return {
        "images": sample_images,
        "actions": sample_actions,
        "language": sample_language_tokens,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark tests as slow (may take a few seconds)"
    )
    config.addinivalue_line("markers", "gpu: mark tests that require GPU")
    config.addinivalue_line(
        "markers", "memory_intensive: mark tests that use significant memory"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if any(
            keyword in item.name.lower()
            for keyword in ["performance", "benchmark", "scaling"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark memory intensive tests
        if any(
            keyword in item.name.lower() for keyword in ["memory", "large", "batch"]
        ):
            item.add_marker(pytest.mark.memory_intensive)


@pytest.fixture
def suppress_logs():
    """Suppress loguru logs during testing."""
    import loguru

    logger = loguru.logger
    logger.remove()
    yield
    # Re-add default handler after test
    logger.add(lambda x: None)  # Null handler for tests
