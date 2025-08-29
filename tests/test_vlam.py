#!/usr/bin/env python3
"""
Simple tests for VLAM Vision-Language-Action Model.
Pure function tests without any testing framework dependencies.
"""

import torch
import traceback
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


def test_action_config():
    """Test ActionConfig class and robot configurations."""
    print("Testing ActionConfig...")

    # Test arm configuration
    arm_config = ActionConfig.get_arm_config()
    assert arm_config.robot_type == RobotType.ARM
    assert arm_config.action_dim == 7
    assert arm_config.action_bounds == (-1.0, 1.0)
    assert len(arm_config.joint_names) == 7
    assert "shoulder_pan" in arm_config.joint_names
    assert "gripper" in arm_config.joint_names

    # Test humanoid configuration
    humanoid_config = ActionConfig.get_humanoid_config()
    assert humanoid_config.robot_type == RobotType.HUMANOID
    assert humanoid_config.action_dim == 25
    assert len(humanoid_config.joint_names) == 25

    # Test custom configuration
    custom_config = ActionConfig(
        robot_type=RobotType.MOBILE_MANIPULATOR,
        action_dim=10,
        action_bounds=(-2.0, 2.0),
        joint_names=[f"joint_{i}" for i in range(10)],
    )
    assert custom_config.robot_type == RobotType.MOBILE_MANIPULATOR
    assert custom_config.action_dim == 10
    assert custom_config.action_bounds == (-2.0, 2.0)

    print("✅ ActionConfig tests passed!")


def test_vision_encoder():
    """Test VisionEncoder component."""
    print("Testing VisionEncoder...")

    # Test initialization
    encoder = VisionEncoder(model_name="test_model", freeze_backbone=True)
    assert encoder.model_name == "test_model"
    assert encoder.freeze_backbone is True
    assert encoder.feature_dim == 1024

    # Test forward pass
    batch_size, channels, height, width = 2, 3, 224, 224
    images = torch.randn(batch_size, channels, height, width)
    features = encoder(images)

    # Check output shape
    assert features.shape == (batch_size, 1, 1024)
    assert features.dtype == torch.float32

    # Test backbone freezing
    for param in encoder.backbone.parameters():
        assert param.requires_grad is False

    print("✅ VisionEncoder tests passed!")


def test_mamba_block():
    """Test MambaBlock component."""
    print("Testing MambaBlock...")

    # Test initialization
    d_model = 512
    block = MambaBlock(d_model)
    assert block.d_model == d_model
    assert block.d_state == 16  # default
    assert block.d_conv == 4  # default

    # Test forward pass
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    assert output.dtype == torch.float32

    # Test different sequence lengths
    for seq_len in [1, 5, 50]:
        x = torch.randn(1, seq_len, d_model)
        output = block(x)
        assert output.shape == (1, seq_len, d_model)

    print("✅ MambaBlock tests passed!")


def test_mamba_ssm():
    """Test MambaSSM component."""
    print("Testing MambaSSM...")

    # Test initialization
    d_model, n_layers = 256, 4
    ssm = MambaSSM(d_model, n_layers)
    assert ssm.d_model == d_model
    assert ssm.n_layers == n_layers
    assert len(ssm.layers) == n_layers

    # Test forward pass
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, d_model)
    output = ssm(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    assert output.dtype == torch.float32

    print("✅ MambaSSM tests passed!")


def test_action_head():
    """Test ActionHead component."""
    print("Testing ActionHead...")

    # Test with arm configuration
    d_model = 512
    config = ActionConfig.get_arm_config()
    head = ActionHead(d_model, config)
    assert head.d_model == d_model
    assert head.action_config == config

    # Test forward pass
    batch_size, seq_len = 2, 10
    features = torch.randn(batch_size, seq_len, d_model)
    actions = head(features)

    # Check output shape and bounds
    assert actions.shape == (batch_size, seq_len, config.action_dim)
    assert torch.all(actions >= config.action_bounds[0])
    assert torch.all(actions <= config.action_bounds[1])

    # Test with humanoid configuration
    humanoid_config = ActionConfig.get_humanoid_config()
    humanoid_head = ActionHead(d_model, humanoid_config)
    humanoid_actions = humanoid_head(features)
    assert humanoid_actions.shape[2] == humanoid_config.action_dim

    print("✅ ActionHead tests passed!")


def test_language_head():
    """Test LanguageHead component."""
    print("Testing LanguageHead...")

    # Test initialization
    d_model, vocab_size = 256, 1000
    head = LanguageHead(d_model, vocab_size)
    assert head.d_model == d_model
    assert head.vocab_size == vocab_size

    # Test forward pass
    batch_size, seq_len = 2, 15
    features = torch.randn(batch_size, seq_len, d_model)
    logits = head(features)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert logits.dtype == torch.float32

    # Test different vocab sizes
    for vocab_size in [100, 5000]:
        head = LanguageHead(d_model, vocab_size)
        logits = head(features)
        assert logits.shape[2] == vocab_size

    print("✅ LanguageHead tests passed!")


def test_vla_model_basic():
    """Test basic VLA model functionality."""
    print("Testing VLA Model Basic Functionality...")

    # Test initialization
    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(
        action_config=config, d_model=256, n_layers=2, vocab_size=1000
    )
    assert model.action_config == config
    assert model.d_model == 256

    # Test forward pass with sequence input
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

    # Test with single image input
    image = torch.randn(1, 3, 224, 224)
    outputs = model(image)
    assert outputs["actions"].shape == (1, 1, config.action_dim)

    print("✅ VLA Model Basic tests passed!")


def test_vla_model_prediction():
    """Test VLA model prediction methods."""
    print("Testing VLA Model Predictions...")

    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

    # Test next action prediction
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        action, description = model.predict_next_action(image)

    assert action.shape == (1, config.action_dim)
    assert isinstance(description, str)
    assert description.startswith("Token_")

    # Test action description
    description = model.get_action_description()
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

    assert description["robot_type"] == "arm"
    assert description["action_dimension"] == 7
    assert isinstance(description["model_parameters"], int)
    assert isinstance(description["trainable_parameters"], int)

    print("✅ VLA Model Prediction tests passed!")


def test_vla_model_different_robots():
    """Test VLA model with different robot configurations."""
    print("Testing VLA Model with Different Robots...")

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

    print("✅ VLA Model Different Robots tests passed!")


def test_vla_model_sequence_features():
    """Test VLA model sequence-related features."""
    print("Testing VLA Model Sequence Features...")

    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

    # Test sequence length truncation
    images = torch.randn(1, 10, 3, 224, 224)
    truncate_len = 5
    outputs = model(images, sequence_length=truncate_len)

    assert outputs["actions"].shape[1] == truncate_len
    assert outputs["language_logits"].shape[1] == truncate_len
    assert outputs["features"].shape[1] == truncate_len

    # Test different sequence lengths
    for seq_len in [1, 5, 20]:
        images = torch.randn(1, seq_len, 3, 224, 224)
        outputs = model(images)
        assert outputs["actions"].shape[1] == seq_len

    print("✅ VLA Model Sequence Features tests passed!")


def test_vla_model_training_mode():
    """Test VLA model in training vs evaluation mode."""
    print("Testing VLA Model Training/Eval Modes...")

    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(
        action_config=config,
        d_model=128,
        n_layers=2,
        freeze_vision=False,  # Allow gradients
    )

    images = torch.randn(1, 2, 3, 224, 224, requires_grad=True)

    # Test training mode
    model.train()
    outputs = model(images)

    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(images)

    # Outputs should have same shape
    assert outputs["actions"].shape == eval_outputs["actions"].shape
    assert outputs["language_logits"].shape == eval_outputs["language_logits"].shape

    print("✅ VLA Model Training/Eval tests passed!")


def test_vla_model_gradient_flow():
    """Test gradient flow through the model."""
    print("Testing VLA Model Gradient Flow...")

    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(
        action_config=config, d_model=128, n_layers=2, freeze_vision=False
    )

    images = torch.randn(1, 2, 3, 224, 224, requires_grad=True)
    target_actions = torch.randn(1, 2, 7)
    target_language = torch.randint(0, 32000, (1, 2))

    # Forward pass
    outputs = model(images)

    # Compute losses
    action_loss = torch.nn.functional.mse_loss(outputs["actions"], target_actions)
    language_loss = torch.nn.functional.cross_entropy(
        outputs["language_logits"].view(-1, 32000), target_language.view(-1)
    )
    total_loss = action_loss + language_loss

    # Backward pass
    total_loss.backward()

    # Check that gradients exist
    assert images.grad is not None

    # Check that some model parameters have gradients
    grad_count = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1

    assert grad_count > 0, "No gradients found in model parameters"

    # Check that losses are finite
    assert torch.isfinite(total_loss)
    assert torch.isfinite(action_loss)
    assert torch.isfinite(language_loss)

    print("✅ VLA Model Gradient Flow tests passed!")


def test_model_sizes():
    """Test different model size configurations."""
    print("Testing Different Model Sizes...")

    config = ActionConfig.get_arm_config()

    model_configs = [
        {"d_model": 128, "n_layers": 2},
        {"d_model": 256, "n_layers": 4},
        {"d_model": 512, "n_layers": 6},
    ]

    for model_config in model_configs:
        model = VisionLanguageActionModel(action_config=config, **model_config)

        images = torch.randn(1, 3, 3, 224, 224)
        outputs = model(images)

        # All models should produce valid outputs
        assert outputs["actions"].shape == (1, 3, 7)
        assert torch.isfinite(outputs["actions"]).all()
        assert torch.isfinite(outputs["language_logits"]).all()

    print("✅ Different Model Sizes tests passed!")


def test_error_handling():
    """Test error handling and edge cases."""
    print("Testing Error Handling...")

    config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel(action_config=config, d_model=128, n_layers=2)

    # Test wrong input dimensions (should raise error)
    try:
        wrong_images = torch.randn(1, 1, 4, 224, 224)  # 4 channels instead of 3
        model(wrong_images)
        assert False, "Should have raised an error for wrong input channels"
    except RuntimeError:
        pass  # Expected error

    # Test empty sequence (should raise error)
    try:
        empty_images = torch.randn(1, 0, 3, 224, 224)
        model(empty_images)
        assert False, "Should have raised an error for empty sequence"
    except (RuntimeError, ValueError):
        pass  # Expected error

    print("✅ Error Handling tests passed!")


def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("Testing Device Compatibility...")

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
        print("✅ GPU compatibility confirmed!")
    else:
        print("⚠️ GPU not available, skipping GPU tests")

    print("✅ Device Compatibility tests passed!")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("RUNNING VLAM MODEL TESTS")
    print("=" * 60)

    test_functions = [
        test_action_config,
        test_vision_encoder,
        test_mamba_block,
        test_mamba_ssm,
        test_action_head,
        test_language_head,
        test_vla_model_basic,
        test_vla_model_prediction,
        test_vla_model_different_robots,
        test_vla_model_sequence_features,
        test_vla_model_training_mode,
        test_vla_model_gradient_flow,
        test_model_sizes,
        test_error_handling,
        test_device_compatibility,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED:")
            print(f"   Error: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            failed += 1
        print()

    print("=" * 60)
    print("TEST RESULTS:")
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"📊 TOTAL:  {passed + failed}")

    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"💥 {failed} TEST(S) FAILED!")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
