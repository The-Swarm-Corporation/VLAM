"""
Test the example scripts and usage patterns.
"""

import pytest
import torch
import sys
import os
from pathlib import Path


class TestExamples:
    """Test example scripts and usage patterns."""

    def test_example_script_imports(self):
        """Test that example.py can be imported without errors."""
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        try:
            # Import the example module
            import example

            assert hasattr(example, "__file__")
        except ImportError as e:
            pytest.fail(f"Could not import example.py: {e}")
        finally:
            # Clean up
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))

    def test_basic_usage_pattern(self):
        """Test the basic usage pattern shown in documentation."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        # Basic usage as shown in README
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=256, n_layers=4)

        # Test forward pass
        images = torch.randn(1, 5, 3, 224, 224)
        outputs = model(images)

        # Verify outputs match documentation
        assert "actions" in outputs
        assert "language_logits" in outputs
        assert "features" in outputs

        actions = outputs["actions"]
        language_logits = outputs["language_logits"]
        features = outputs["features"]

        assert actions.shape == (1, 5, 7)  # batch=1, seq=5, action_dim=7
        assert language_logits.shape[0] == 1  # batch size
        assert language_logits.shape[1] == 5  # sequence length
        assert features.shape == (1, 5, 256)  # d_model=256

    def test_single_step_prediction_pattern(self):
        """Test single-step prediction pattern."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Single-step prediction as shown in README
        current_observation = torch.randn(1, 3, 224, 224)
        next_action, description = model.predict_next_action(current_observation)

        assert next_action.shape == (1, 7)  # batch=1, action_dim=7
        assert isinstance(description, str)
        assert len(description) > 0

    def test_robot_configuration_examples(self):
        """Test robot configuration examples from documentation."""
        from vlam.main import VisionLanguageActionModel, ActionConfig, RobotType

        # Test arm configuration
        arm_config = ActionConfig.get_arm_config()
        arm_model = VisionLanguageActionModel(action_config=arm_config)

        assert arm_config.robot_type == RobotType.ARM
        assert arm_config.action_dim == 7

        # Test humanoid configuration
        humanoid_config = ActionConfig.get_humanoid_config()
        humanoid_model = VisionLanguageActionModel(action_config=humanoid_config)

        assert humanoid_config.robot_type == RobotType.HUMANOID
        assert humanoid_config.action_dim == 25

        # Test custom configuration
        custom_config = ActionConfig(
            robot_type=RobotType.MOBILE_MANIPULATOR,
            action_dim=12,
            action_bounds=(-2.0, 2.0),
            joint_names=[f"joint_{i}" for i in range(12)],
        )
        custom_model = VisionLanguageActionModel(action_config=custom_config)

        assert custom_config.robot_type == RobotType.MOBILE_MANIPULATOR
        assert custom_config.action_dim == 12
        assert custom_config.action_bounds == (-2.0, 2.0)

    def test_model_configurations(self):
        """Test different model size configurations."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()

        # Test small configuration
        small_model = VisionLanguageActionModel(
            action_config=config, d_model=512, n_layers=4
        )

        # Test medium configuration
        medium_model = VisionLanguageActionModel(
            action_config=config, d_model=768, n_layers=6
        )

        # Test large configuration
        large_model = VisionLanguageActionModel(
            action_config=config, d_model=1024, n_layers=8
        )

        # All should work with same input
        images = torch.randn(1, 3, 3, 224, 224)

        small_outputs = small_model(images)
        medium_outputs = medium_model(images)
        large_outputs = large_model(images)

        # All should produce correct output shapes
        for outputs in [small_outputs, medium_outputs, large_outputs]:
            assert outputs["actions"].shape == (1, 3, 7)
            assert outputs["language_logits"].shape[0] == 1
            assert outputs["language_logits"].shape[1] == 3

    def test_training_loop_example(self):
        """Test training loop example pattern."""
        from vlam.main import VisionLanguageActionModel, ActionConfig
        import torch.optim as optim
        from torch.nn import CrossEntropyLoss, MSELoss

        # Initialize model as in documentation
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(config)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Loss functions
        action_loss_fn = MSELoss()
        language_loss_fn = CrossEntropyLoss()

        # Mock training step
        batch_size = 2
        seq_len = 5
        vocab_size = 32000

        images = torch.randn(batch_size, seq_len, 3, 224, 224)
        actions_gt = torch.randn(batch_size, seq_len, config.action_dim)
        language_gt = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Training step as shown in documentation
        outputs = model(images)

        action_loss = action_loss_fn(outputs["actions"], actions_gt)
        language_loss = language_loss_fn(
            outputs["language_logits"].view(-1, vocab_size), language_gt.view(-1)
        )

        total_loss = action_loss + language_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check that losses are finite
        assert torch.isfinite(action_loss)
        assert torch.isfinite(language_loss)
        assert torch.isfinite(total_loss)

    def test_model_description_api(self):
        """Test model description API."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_humanoid_config()
        model = VisionLanguageActionModel(action_config=config)

        # Test get_action_description as shown in documentation
        description = model.get_action_description()

        # Check required keys
        required_keys = [
            "robot_type",
            "action_dimension",
            "action_bounds",
            "joint_names",
            "model_parameters",
            "trainable_parameters",
        ]

        for key in required_keys:
            assert key in description

        assert description["robot_type"] == "humanoid"
        assert description["action_dimension"] == 25
        assert isinstance(description["model_parameters"], int)
        assert isinstance(description["trainable_parameters"], int)
        assert description["trainable_parameters"] <= description["model_parameters"]

    def test_inference_modes(self):
        """Test different inference modes."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=256, n_layers=3)

        images = torch.randn(1, 5, 3, 224, 224)

        # Test training mode
        model.train()
        train_outputs = model(images)

        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            eval_outputs = model(images)

        # Outputs should have same shape
        assert train_outputs["actions"].shape == eval_outputs["actions"].shape
        assert (
            train_outputs["language_logits"].shape
            == eval_outputs["language_logits"].shape
        )

        # Test that eval mode is deterministic
        with torch.no_grad():
            eval_outputs2 = model(images)

        assert torch.allclose(eval_outputs["actions"], eval_outputs2["actions"])
        assert torch.allclose(
            eval_outputs["language_logits"], eval_outputs2["language_logits"]
        )

    def test_device_usage_patterns(self):
        """Test device usage patterns."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Test CPU usage
        images_cpu = torch.randn(1, 3, 3, 224, 224)
        outputs_cpu = model(images_cpu)

        assert outputs_cpu["actions"].device.type == "cpu"

        # Test GPU usage if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            images_gpu = images_cpu.cuda()
            outputs_gpu = model_gpu(images_gpu)

            assert outputs_gpu["actions"].device.type == "cuda"

    def test_sequence_length_handling(self):
        """Test sequence length truncation feature."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Create longer sequence
        images = torch.randn(1, 10, 3, 224, 224)

        # Test truncation
        truncate_len = 5
        outputs = model(images, sequence_length=truncate_len)

        assert outputs["actions"].shape[1] == truncate_len
        assert outputs["language_logits"].shape[1] == truncate_len
        assert outputs["features"].shape[1] == truncate_len

    def test_error_patterns(self):
        """Test common error patterns and edge cases."""
        from vlam.main import VisionLanguageActionModel, ActionConfig

        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Test wrong input dimensions
        with pytest.raises(RuntimeError):
            wrong_images = torch.randn(1, 1, 4, 224, 224)  # 4 channels instead of 3
            model(wrong_images)

        # Test empty sequence
        with pytest.raises((RuntimeError, ValueError)):
            empty_images = torch.randn(1, 0, 3, 224, 224)
            model(empty_images)


class TestPlottingExamples:
    """Test plotting and visualization examples."""

    def test_plotting_script_syntax(self):
        """Test that test_plots.py has valid syntax."""
        project_root = Path(__file__).parent.parent
        plot_script = project_root / "test_plots.py"

        # Check that file exists
        assert plot_script.exists(), "test_plots.py not found"

        # Try to compile the script
        with open(plot_script, "r") as f:
            code = f.read()

        try:
            compile(code, str(plot_script), "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in test_plots.py: {e}")

    def test_plotting_imports(self):
        """Test that plotting script imports work."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # These should work for the plotting examples
            assert hasattr(plt, "figure")
            assert hasattr(np, "linspace")
        except ImportError as e:
            pytest.skip(f"Plotting dependencies not available: {e}")

    def test_plot_directory_creation(self):
        """Test plot directory creation logic."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_dir = os.path.join(temp_dir, "plots")

            # This should work without errors
            os.makedirs(plot_dir, exist_ok=True)
            assert os.path.exists(plot_dir)
            assert os.path.isdir(plot_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
