"""
Comprehensive unit tests for VLAM Vision-Language-Action Model.
Tests all major components and functionality.
"""

import pytest
import torch
from vlam.main import (
    VisionLanguageActionModel,
    ActionConfig,
    RobotType,
    VisionEncoder,
    MambaBlock,
    MambaSSM,
    ActionHead,
    LanguageHead,
)


class TestActionConfig:
    """Test ActionConfig class and robot configurations."""

    def test_arm_config_creation(self):
        """Test arm configuration creation."""
        config = ActionConfig.get_arm_config()

        assert config.robot_type == RobotType.ARM
        assert config.action_dim == 7
        assert config.action_bounds == (-1.0, 1.0)
        assert len(config.joint_names) == 7
        assert "shoulder_pan" in config.joint_names
        assert "gripper" in config.joint_names

    def test_humanoid_config_creation(self):
        """Test humanoid configuration creation."""
        config = ActionConfig.get_humanoid_config()

        assert config.robot_type == RobotType.HUMANOID
        assert config.action_dim == 25
        assert config.action_bounds == (-1.0, 1.0)
        assert len(config.joint_names) == 25
        assert "head_yaw" in config.joint_names
        assert "right_ankle_pitch" in config.joint_names

    def test_custom_config_creation(self):
        """Test custom configuration creation."""
        config = ActionConfig(
            robot_type=RobotType.MOBILE_MANIPULATOR,
            action_dim=10,
            action_bounds=(-2.0, 2.0),
            joint_names=[f"joint_{i}" for i in range(10)],
        )

        assert config.robot_type == RobotType.MOBILE_MANIPULATOR
        assert config.action_dim == 10
        assert config.action_bounds == (-2.0, 2.0)
        assert len(config.joint_names) == 10


class TestVisionEncoder:
    """Test VisionEncoder component."""

    def test_vision_encoder_initialization(self):
        """Test vision encoder initialization."""
        encoder = VisionEncoder(model_name="test_model", freeze_backbone=True)

        assert encoder.model_name == "test_model"
        assert encoder.freeze_backbone is True
        assert encoder.feature_dim == 1024
        assert encoder.backbone is not None
        assert encoder.projection is not None

    def test_vision_encoder_forward_pass(self):
        """Test vision encoder forward pass."""
        encoder = VisionEncoder()
        batch_size, channels, height, width = 2, 3, 224, 224

        # Test input
        images = torch.randn(batch_size, channels, height, width)

        # Forward pass
        features = encoder(images)

        # Check output shape
        assert features.shape == (batch_size, 1, 1024)
        assert features.dtype == torch.float32

    def test_vision_encoder_freeze_backbone(self):
        """Test backbone freezing functionality."""
        encoder = VisionEncoder(freeze_backbone=True)

        # Check that backbone parameters are frozen
        for param in encoder.backbone.parameters():
            assert param.requires_grad is False

        # Projection layer should remain trainable
        for param in encoder.projection.parameters():
            assert param.requires_grad is True

    def test_vision_encoder_unfreeze_backbone(self):
        """Test backbone unfreezing functionality."""
        encoder = VisionEncoder(freeze_backbone=False)

        # Check that backbone parameters are trainable
        for param in encoder.backbone.parameters():
            assert param.requires_grad is True


class TestMambaBlock:
    """Test MambaBlock component."""

    def test_mamba_block_initialization(self):
        """Test Mamba block initialization."""
        d_model, d_state, d_conv, expand = 512, 16, 4, 2
        block = MambaBlock(d_model, d_state, d_conv, expand)

        assert block.d_model == d_model
        assert block.d_state == d_state
        assert block.d_conv == d_conv
        assert block.d_inner == int(expand * d_model)

        # Check layer existence
        assert block.in_proj is not None
        assert block.conv1d is not None
        assert block.x_proj is not None
        assert block.dt_proj is not None
        assert block.out_proj is not None

    def test_mamba_block_forward_pass(self):
        """Test Mamba block forward pass."""
        d_model = 512
        block = MambaBlock(d_model)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        output = block(x)

        # Check output shape and type
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32

    def test_mamba_block_different_sizes(self):
        """Test Mamba block with different input sizes."""
        d_model = 256
        block = MambaBlock(d_model)

        # Test different sequence lengths
        for seq_len in [1, 5, 50, 100]:
            x = torch.randn(1, seq_len, d_model)
            output = block(x)
            assert output.shape == (1, seq_len, d_model)


class TestMambaSSM:
    """Test MambaSSM component."""

    def test_mamba_ssm_initialization(self):
        """Test Mamba SSM initialization."""
        d_model, n_layers = 512, 6
        ssm = MambaSSM(d_model, n_layers)

        assert ssm.d_model == d_model
        assert ssm.n_layers == n_layers
        assert len(ssm.layers) == n_layers
        assert ssm.norm is not None

    def test_mamba_ssm_forward_pass(self):
        """Test Mamba SSM forward pass."""
        d_model, n_layers = 256, 4
        ssm = MambaSSM(d_model, n_layers)

        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        output = ssm(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32

    def test_mamba_ssm_residual_connections(self):
        """Test that residual connections preserve gradients."""
        d_model, n_layers = 128, 2
        ssm = MambaSSM(d_model, n_layers)

        x = torch.randn(1, 10, d_model, requires_grad=True)
        output = ssm(x)

        # Compute dummy loss and check gradients
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestActionHead:
    """Test ActionHead component."""

    def test_action_head_initialization(self):
        """Test action head initialization."""
        d_model = 512
        config = ActionConfig.get_arm_config()
        head = ActionHead(d_model, config)

        assert head.d_model == d_model
        assert head.action_config == config
        assert head.action_layers is not None

    def test_action_head_forward_pass(self):
        """Test action head forward pass."""
        d_model = 512
        config = ActionConfig.get_arm_config()
        head = ActionHead(d_model, config)

        batch_size, seq_len = 2, 10
        features = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        actions = head(features)

        # Check output shape and bounds
        assert actions.shape == (batch_size, seq_len, config.action_dim)
        assert torch.all(actions >= config.action_bounds[0])
        assert torch.all(actions <= config.action_bounds[1])

    def test_action_head_different_robots(self):
        """Test action head with different robot configurations."""
        d_model = 256

        # Test arm configuration
        arm_config = ActionConfig.get_arm_config()
        arm_head = ActionHead(d_model, arm_config)

        # Test humanoid configuration
        humanoid_config = ActionConfig.get_humanoid_config()
        humanoid_head = ActionHead(d_model, humanoid_config)

        features = torch.randn(1, 5, d_model)

        arm_actions = arm_head(features)
        humanoid_actions = humanoid_head(features)

        assert arm_actions.shape[2] == arm_config.action_dim
        assert humanoid_actions.shape[2] == humanoid_config.action_dim


class TestLanguageHead:
    """Test LanguageHead component."""

    def test_language_head_initialization(self):
        """Test language head initialization."""
        d_model, vocab_size = 512, 32000
        head = LanguageHead(d_model, vocab_size)

        assert head.d_model == d_model
        assert head.vocab_size == vocab_size
        assert head.language_layers is not None

    def test_language_head_forward_pass(self):
        """Test language head forward pass."""
        d_model, vocab_size = 256, 1000
        head = LanguageHead(d_model, vocab_size)

        batch_size, seq_len = 2, 15
        features = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        logits = head(features)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert logits.dtype == torch.float32

    def test_language_head_different_vocab_sizes(self):
        """Test language head with different vocabulary sizes."""
        d_model = 128
        features = torch.randn(1, 10, d_model)

        for vocab_size in [100, 1000, 50000]:
            head = LanguageHead(d_model, vocab_size)
            logits = head(features)
            assert logits.shape[2] == vocab_size


class TestVisionLanguageActionModel:
    """Test complete VisionLanguageActionModel."""

    def test_vla_model_initialization(self):
        """Test VLA model initialization."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=256, n_layers=4)

        assert model.action_config == config
        assert model.d_model == 256
        assert model.vision_encoder is not None
        assert model.vision_proj is not None
        assert model.mamba is not None
        assert model.action_head is not None
        assert model.language_head is not None

    def test_vla_model_forward_pass(self):
        """Test VLA model forward pass."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(
            action_config=config, d_model=256, n_layers=2, vocab_size=1000
        )

        # Test with sequence input
        batch_size, seq_len = 2, 5
        images = torch.randn(batch_size, seq_len, 3, 224, 224)

        outputs = model(images)

        # Check output keys and shapes
        assert "actions" in outputs
        assert "language_logits" in outputs
        assert "features" in outputs

        assert outputs["actions"].shape == (batch_size, seq_len, config.action_dim)
        assert outputs["language_logits"].shape == (batch_size, seq_len, 1000)
        assert outputs["features"].shape == (batch_size, seq_len, 256)

    def test_vla_model_single_image_input(self):
        """Test VLA model with single image input."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Single image input
        image = torch.randn(1, 3, 224, 224)
        outputs = model(image)

        # Should automatically add sequence dimension
        assert outputs["actions"].shape == (1, 1, config.action_dim)

    def test_vla_model_predict_next_action(self):
        """Test next action prediction."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            action, description = model.predict_next_action(image)

        assert action.shape == (1, config.action_dim)
        assert isinstance(description, str)
        assert description.startswith("Token_")

    def test_vla_model_get_action_description(self):
        """Test action description method."""
        config = ActionConfig.get_humanoid_config()
        model = VisionLanguageActionModel(action_config=config, d_model=256)

        description = model.get_action_description()

        assert "robot_type" in description
        assert "action_dimension" in description
        assert "action_bounds" in description
        assert "joint_names" in description
        assert "model_parameters" in description
        assert "trainable_parameters" in description

        assert description["robot_type"] == "humanoid"
        assert description["action_dimension"] == 25

    def test_vla_model_sequence_length_truncation(self):
        """Test sequence length truncation."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Create longer sequence
        images = torch.randn(1, 10, 3, 224, 224)

        # Truncate to shorter length
        truncate_len = 5
        outputs = model(images, sequence_length=truncate_len)

        assert outputs["actions"].shape[1] == truncate_len
        assert outputs["language_logits"].shape[1] == truncate_len
        assert outputs["features"].shape[1] == truncate_len

    def test_vla_model_different_robot_types(self):
        """Test VLA model with different robot configurations."""
        # Test arm model
        arm_config = ActionConfig.get_arm_config()
        arm_model = VisionLanguageActionModel(
            action_config=arm_config, d_model=128, n_layers=2
        )

        # Test humanoid model
        humanoid_config = ActionConfig.get_humanoid_config()
        humanoid_model = VisionLanguageActionModel(
            action_config=humanoid_config, d_model=128, n_layers=2
        )

        images = torch.randn(1, 3, 3, 224, 224)

        arm_outputs = arm_model(images)
        humanoid_outputs = humanoid_model(images)

        assert arm_outputs["actions"].shape[2] == 7  # 7-DOF arm
        assert humanoid_outputs["actions"].shape[2] == 25  # 25-DOF humanoid

    def test_vla_model_gradient_flow(self):
        """Test gradient flow through the model."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(
            action_config=config,
            d_model=128,
            n_layers=2,
            freeze_vision=False,  # Allow gradients through vision encoder
        )

        images = torch.randn(1, 2, 3, 224, 224, requires_grad=True)
        outputs = model(images)

        # Compute dummy loss
        action_loss = outputs["actions"].sum()
        language_loss = outputs["language_logits"].sum()
        total_loss = action_loss + language_loss

        total_loss.backward()

        # Check that gradients exist
        assert images.grad is not None

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize model
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(
            action_config=config, d_model=256, n_layers=3, vocab_size=1000
        )

        # Simulate training data
        batch_size = 4
        seq_len = 8
        images = torch.randn(batch_size, seq_len, 3, 224, 224)
        target_actions = torch.randn(batch_size, seq_len, config.action_dim)
        target_language = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        outputs = model(images)

        # Compute losses
        action_loss = torch.nn.functional.mse_loss(outputs["actions"], target_actions)
        language_loss = torch.nn.functional.cross_entropy(
            outputs["language_logits"].view(-1, 1000), target_language.view(-1)
        )

        total_loss = action_loss + language_loss

        # Backward pass
        total_loss.backward()

        # Check that loss is finite
        assert torch.isfinite(total_loss)
        assert action_loss.item() > 0
        assert language_loss.item() > 0

    def test_model_save_load(self):
        """Test model saving and loading."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Save model state
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = VisionLanguageActionModel(
            action_config=config, d_model=128, n_layers=2
        )
        new_model.load_state_dict(state_dict)

        # Test that outputs are identical
        images = torch.randn(1, 3, 3, 224, 224)

        with torch.no_grad():
            outputs1 = model(images)
            outputs2 = new_model(images)

        assert torch.allclose(outputs1["actions"], outputs2["actions"])
        assert torch.allclose(outputs1["language_logits"], outputs2["language_logits"])

    def test_model_eval_mode(self):
        """Test model behavior in evaluation mode."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        images = torch.randn(1, 3, 3, 224, 224)

        # Test in training mode
        model.train()
        outputs_train = model(images)

        # Test in evaluation mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(images)

        # Outputs should be deterministic in eval mode
        with torch.no_grad():
            outputs_eval2 = model(images)

        assert torch.allclose(outputs_eval["actions"], outputs_eval2["actions"])
        assert torch.allclose(
            outputs_eval["language_logits"], outputs_eval2["language_logits"]
        )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Test with wrong number of channels
        with pytest.raises(RuntimeError):
            invalid_images = torch.randn(1, 1, 4, 224, 224)  # 4 channels instead of 3
            model(invalid_images)

    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Test with zero sequence length
        with pytest.raises((RuntimeError, ValueError)):
            empty_images = torch.randn(1, 0, 3, 224, 224)
            model(empty_images)

    def test_device_compatibility(self):
        """Test device compatibility."""
        config = ActionConfig.get_arm_config()
        model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

        # Test CPU
        images_cpu = torch.randn(1, 2, 3, 224, 224)
        outputs_cpu = model(images_cpu)

        assert outputs_cpu["actions"].device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            images_gpu = images_cpu.cuda()
            outputs_gpu = model_gpu(images_gpu)

            assert outputs_gpu["actions"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
