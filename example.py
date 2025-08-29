import torch
from vlam.main import VisionLanguageActionModel, ActionConfig, MAMBA_AVAILABLE
from loguru import logger


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logger.info("Initializing Vision-Language-Action Model")

    # Show which Mamba implementation is being used
    if MAMBA_AVAILABLE:
        logger.info("✅ Using official mamba-ssm library for optimal performance")
    else:
        logger.info("⚠️  Using fallback Mamba implementation")
        logger.info(
            "💡 Install mamba-ssm for better performance: pip install mamba-ssm"
        )

    # Create model for robotic arm
    arm_config = ActionConfig.get_arm_config()
    arm_model = VisionLanguageActionModel(
        action_config=arm_config, d_model=512, n_layers=4
    )

    # Create model for humanoid
    humanoid_config = ActionConfig.get_humanoid_config()
    humanoid_model = VisionLanguageActionModel(
        action_config=humanoid_config, d_model=512, n_layers=6
    )

    # Test with sample data
    batch_size, seq_len = 2, 5
    images = torch.randn(batch_size, seq_len, 3, 224, 224)

    logger.info("Testing arm model...")
    arm_outputs = arm_model(images)
    logger.info(f"Arm model description: {arm_model.get_action_description()}")

    logger.info("Testing humanoid model...")
    humanoid_outputs = humanoid_model(images)
    logger.info(
        f"Humanoid model description: {humanoid_model.get_action_description()}"
    )

    # Test single-step prediction
    single_image = torch.randn(1, 3, 224, 224)
    action, description = arm_model.predict_next_action(single_image)

    logger.info("VLA Model architecture complete!")
