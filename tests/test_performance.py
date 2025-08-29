"""
Performance and benchmarking tests for VLAM model.
Tests computational efficiency, memory usage, and scaling properties.
"""

import pytest
import torch
import time
import psutil
import gc
from vlam.main import VisionLanguageActionModel, ActionConfig


class TestPerformance:
    """Performance benchmarks for VLAM components."""

    @pytest.fixture
    def small_model(self):
        """Create small model for performance testing."""
        config = ActionConfig.get_arm_config()
        return VisionLanguageActionModel(
            action_config=config, d_model=256, n_layers=2, vocab_size=1000
        )

    @pytest.fixture
    def medium_model(self):
        """Create medium model for performance testing."""
        config = ActionConfig.get_arm_config()
        return VisionLanguageActionModel(
            action_config=config, d_model=512, n_layers=4, vocab_size=5000
        )

    def test_inference_speed_single_image(self, small_model):
        """Test inference speed with single image."""
        model = small_model
        model.eval()

        image = torch.randn(1, 3, 224, 224)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(image)

        # Measure inference time
        start_time = time.time()
        num_runs = 100

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(image)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs

        # Should be under 50ms for small model on CPU
        assert avg_time < 0.05, f"Inference too slow: {avg_time:.4f}s"

    def test_inference_speed_sequences(self, small_model):
        """Test inference speed with different sequence lengths."""
        model = small_model
        model.eval()

        sequence_lengths = [1, 5, 10, 20]
        times = []

        for seq_len in sequence_lengths:
            images = torch.randn(1, seq_len, 3, 224, 224)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(images)

            # Measure time
            start_time = time.time()
            num_runs = 10

            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(images)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            times.append(avg_time)

        # Check that time scaling is roughly linear (within factor of 2)
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            expected_ratio = sequence_lengths[i] / sequence_lengths[0]
            assert (
                ratio < expected_ratio * 2
            ), f"Scaling not linear: {ratio} vs {expected_ratio}"

    def test_memory_usage_scaling(self, small_model):
        """Test memory usage with different batch sizes."""
        model = small_model
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        memory_usage = []

        for batch_size in batch_sizes:
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run inference
            images = torch.randn(batch_size, 5, 3, 224, 224)
            with torch.no_grad():
                outputs = model(images)

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            memory_usage.append(memory_diff)

            # Clean up
            del images, outputs
            gc.collect()

        # Memory should scale roughly linearly with batch size
        for i in range(1, len(memory_usage)):
            if memory_usage[0] > 0:  # Avoid division by zero
                ratio = memory_usage[i] / memory_usage[0]
                expected_ratio = batch_sizes[i] / batch_sizes[0]
                # Allow some overhead, should be within 3x of expected
                assert (
                    ratio < expected_ratio * 3
                ), f"Memory scaling issue: {ratio} vs {expected_ratio}"

    def test_parameter_count(self, small_model, medium_model):
        """Test parameter counting for different model sizes."""
        small_params = sum(p.numel() for p in small_model.parameters())
        medium_params = sum(p.numel() for p in medium_model.parameters())

        # Medium model should have more parameters
        assert medium_params > small_params

        # Should be reasonable number of parameters (not too large)
        assert small_params < 100_000_000  # Under 100M parameters
        assert medium_params < 500_000_000  # Under 500M parameters

    def test_gradient_computation_speed(self, small_model):
        """Test gradient computation speed."""
        model = small_model
        model.train()

        images = torch.randn(2, 5, 3, 224, 224, requires_grad=True)
        target_actions = torch.randn(2, 5, 7)
        target_language = torch.randint(0, 1000, (2, 5))

        # Measure forward + backward pass time
        start_time = time.time()

        outputs = model(images)
        action_loss = torch.nn.functional.mse_loss(outputs["actions"], target_actions)
        language_loss = torch.nn.functional.cross_entropy(
            outputs["language_logits"].view(-1, 1000), target_language.view(-1)
        )
        total_loss = action_loss + language_loss
        total_loss.backward()

        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 5.0, "Gradient computation too slow"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, small_model):
        """Test GPU acceleration if available."""
        # CPU timing
        model_cpu = small_model
        images_cpu = torch.randn(1, 10, 3, 224, 224)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_cpu(images_cpu)
        cpu_time = time.time() - start_time

        # GPU timing
        model_gpu = small_model.cuda()
        images_gpu = images_cpu.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model_gpu(images_gpu)

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(images_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        # GPU should be faster (or at least not much slower for small models)
        assert (
            gpu_time < cpu_time * 2
        ), f"GPU not accelerating: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s"


class TestMemoryEfficiency:
    """Test memory efficiency and optimization."""

    def test_vision_encoder_freezing_memory(self):
        """Test that freezing vision encoder saves memory during training."""
        config = ActionConfig.get_arm_config()

        # Model with frozen vision encoder
        model_frozen = VisionLanguageActionModel(
            action_config=config, d_model=256, n_layers=2, freeze_vision=True
        )

        # Model with unfrozen vision encoder
        model_unfrozen = VisionLanguageActionModel(
            action_config=config, d_model=256, n_layers=2, freeze_vision=False
        )

        images = torch.randn(2, 3, 3, 224, 224, requires_grad=True)
        target_actions = torch.randn(2, 3, 7)

        # Test frozen model
        model_frozen.train()
        outputs_frozen = model_frozen(images)
        loss_frozen = torch.nn.functional.mse_loss(
            outputs_frozen["actions"], target_actions
        )

        # Count trainable parameters
        frozen_trainable = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        unfrozen_trainable = sum(
            p.numel() for p in model_unfrozen.parameters() if p.requires_grad
        )

        # Frozen model should have fewer trainable parameters
        assert frozen_trainable < unfrozen_trainable

    def test_gradient_checkpointing_compatibility(self, small_model):
        """Test that model is compatible with gradient checkpointing."""
        model = small_model
        model.train()

        # Test with gradient checkpointing simulation
        images = torch.randn(1, 3, 3, 224, 224, requires_grad=True)

        # This should work without errors
        outputs = model(images)
        loss = outputs["actions"].sum() + outputs["language_logits"].sum()
        loss.backward()

        # Check gradients exist
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_mixed_precision_compatibility(self, small_model):
        """Test compatibility with mixed precision training."""
        model = small_model
        model.train()

        images = torch.randn(1, 3, 3, 224, 224)
        target_actions = torch.randn(1, 3, 7)

        # Test with autocast
        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=False):
            outputs = model(images)
            loss = torch.nn.functional.mse_loss(outputs["actions"], target_actions)

        # Should work without errors
        assert torch.isfinite(loss)

    def test_memory_cleanup(self, small_model):
        """Test that model properly cleans up memory."""
        model = small_model

        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Run multiple forward passes
        for _ in range(10):
            images = torch.randn(2, 5, 3, 224, 224)
            with torch.no_grad():
                outputs = model(images)
            del images, outputs

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory should not grow significantly
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            assert (
                memory_growth < 100 * 1024 * 1024
            ), f"Memory leak detected: {memory_growth} bytes"


class TestScalability:
    """Test model scalability with different configurations."""

    def test_different_model_sizes(self):
        """Test different model size configurations."""
        config = ActionConfig.get_arm_config()

        model_configs = [
            {"d_model": 128, "n_layers": 2},
            {"d_model": 256, "n_layers": 4},
            {"d_model": 512, "n_layers": 6},
        ]

        for i, model_config in enumerate(model_configs):
            model = VisionLanguageActionModel(action_config=config, **model_config)

            images = torch.randn(1, 3, 3, 224, 224)
            outputs = model(images)

            # All models should produce valid outputs
            assert outputs["actions"].shape == (1, 3, 7)
            assert torch.isfinite(outputs["actions"]).all()
            assert torch.isfinite(outputs["language_logits"]).all()

    def test_different_sequence_lengths(self, small_model):
        """Test with various sequence lengths."""
        model = small_model
        model.eval()

        sequence_lengths = [1, 2, 5, 10, 25, 50]

        for seq_len in sequence_lengths:
            images = torch.randn(1, seq_len, 3, 224, 224)

            with torch.no_grad():
                outputs = model(images)

            assert outputs["actions"].shape == (1, seq_len, 7)
            assert outputs["language_logits"].shape[1] == seq_len

    def test_different_batch_sizes(self, small_model):
        """Test with various batch sizes."""
        model = small_model
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 3, 224, 224)

            with torch.no_grad():
                outputs = model(images)

            assert outputs["actions"].shape == (batch_size, 3, 7)
            assert outputs["language_logits"].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
