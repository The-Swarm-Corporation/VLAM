import torch
from vlam.main import VisionLanguageActionModel, ActionConfig

if __name__ == "__main__":
    # Test: create a model and run a forward pass with dummy data
    arm_config = ActionConfig.get_arm_config()
    model = VisionLanguageActionModel.create_with_recommended_vision(
        action_config=arm_config,
        performance_tier="fast",
        d_model=512,
        n_layers=4,
        verbose=True,
    )
    batch_size, seq_len, input_size = 2, 3, 224
    images = torch.randn(batch_size, seq_len, 3, input_size, input_size)
    outputs = model(images)
    print(outputs)
    
    desc = model.get_action_description()
    print(desc)