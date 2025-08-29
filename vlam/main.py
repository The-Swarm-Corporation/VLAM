"""
Vision-Language-Action (VLA) Model for Robotics
Theoretical architecture combining vision, language, and robotic actions using Mamba SSM.

This model takes visual input, processes it through a pretrained vision encoder,
and uses Mamba state space model to generate both language descriptions and
robot actions simultaneously.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Try to import the official mamba-ssm library
try:
    from mamba_ssm import Mamba  # type: ignore

    MAMBA_AVAILABLE = True
    logger.info("Using official mamba-ssm library")
except ImportError:
    MAMBA_AVAILABLE = False
    logger.info("mamba-ssm library not available, using fallback implementation")

import timm


class RobotType(Enum):
    """Supported robot configurations."""

    ARM = "arm"
    HUMANOID = "humanoid"
    MOBILE_MANIPULATOR = "mobile_manipulator"


@dataclass
class ActionConfig:
    """Configuration for different robot action spaces."""

    robot_type: RobotType
    action_dim: int
    action_bounds: Tuple[float, float]
    joint_names: List[str]

    @classmethod
    def get_arm_config(cls) -> "ActionConfig":
        """7-DOF robotic arm configuration."""
        return cls(
            robot_type=RobotType.ARM,
            action_dim=7,
            action_bounds=(-1.0, 1.0),
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow",
                "wrist1",
                "wrist2",
                "wrist3",
                "gripper",
            ],
        )

    @classmethod
    def get_humanoid_config(cls) -> "ActionConfig":
        """Humanoid robot configuration."""
        return cls(
            robot_type=RobotType.HUMANOID,
            action_dim=26,
            action_bounds=(-1.0, 1.0),
            joint_names=[
                "head_yaw",
                "head_pitch",
                "torso_yaw",
                "torso_pitch",
                "torso_roll",
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_elbow",
                "left_wrist_yaw",
                "left_hand",
                "right_shoulder_pitch",
                "right_shoulder_roll",
                "right_elbow",
                "right_wrist_yaw",
                "right_hand",
                "left_hip_yaw",
                "left_hip_roll",
                "left_hip_pitch",
                "left_knee",
                "left_ankle_pitch",
                "left_ankle_roll",
                "right_hip_yaw",
                "right_hip_roll",
                "right_hip_pitch",
                "right_knee",
                "right_ankle_pitch",
            ],
        )


class VisionEncoder(nn.Module):
    """
    Advanced pretrained vision encoder using timm library.
    Supports state-of-the-art vision models including ConvNeXt, ViT, EfficientNet, and more.
    """

    # Best performing models categorized by type
    BEST_MODELS = {
        "convnext": {
            "convnext_base": {"feature_dim": 1024, "input_size": 224},
            "convnext_large": {"feature_dim": 1536, "input_size": 224},
            "convnext_xlarge": {"feature_dim": 2048, "input_size": 224},
        },
        "vit": {
            "vit_base_patch16_224": {"feature_dim": 768, "input_size": 224},
            "vit_large_patch16_224": {"feature_dim": 1024, "input_size": 224},
            "vit_huge_patch14_224": {"feature_dim": 1280, "input_size": 224},
        },
        "efficientnet": {
            "efficientnet_b0": {"feature_dim": 1280, "input_size": 224},
            "efficientnet_b4": {"feature_dim": 1792, "input_size": 380},
            "efficientnet_b7": {"feature_dim": 2560, "input_size": 600},
            "efficientnetv2_s": {"feature_dim": 1280, "input_size": 384},
            "efficientnetv2_m": {"feature_dim": 1280, "input_size": 480},
            "efficientnetv2_l": {"feature_dim": 1280, "input_size": 480},
        },
        "swin": {
            "swin_base_patch4_window7_224": {"feature_dim": 1024, "input_size": 224},
            "swin_large_patch4_window7_224": {"feature_dim": 1536, "input_size": 224},
        },
        "resnet": {
            "resnet50": {"feature_dim": 2048, "input_size": 224},
            "resnet101": {"feature_dim": 2048, "input_size": 224},
            "resnet152": {"feature_dim": 2048, "input_size": 224},
        },
        "deit": {
            "deit3_base_patch16_224": {"feature_dim": 768, "input_size": 224},
            "deit3_large_patch16_224": {"feature_dim": 1024, "input_size": 224},
        },
    }

    def __init__(self, model_name: str = "convnext_base", freeze_backbone: bool = True):
        """
        Initialize vision encoder with timm pretrained models.

        Args:
            model_name: Name of pretrained vision model from timm
            freeze_backbone: Whether to freeze pretrained weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Get model info
        self.model_info = self._get_model_info(model_name)
        self.feature_dim = self.model_info["feature_dim"]
        self.input_size = self.model_info["input_size"]

        # Create backbone
        self.backbone = self._create_timm_backbone()
        self.projection = None  # Not needed for timm models
        logger.info(f"Using timm model: {model_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        logger.info("Vision encoder initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Feature dimension: {self.feature_dim}")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Frozen: {freeze_backbone}")

    def _get_model_info(self, model_name: str) -> Dict[str, int]:
        """Get model information from predefined best models or fallback."""
        for model_type, models in self.BEST_MODELS.items():
            if model_name in models:
                return models[model_name]

        # Fallback for unknown models
        logger.warning(f"Unknown model {model_name}, using default feature_dim=1024")
        return {"feature_dim": 1024, "input_size": 224}

    def _create_timm_backbone(self) -> nn.Module:
        """Create vision backbone using timm library."""
        try:
            # Create model without classifier head
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool="",  # Remove global pooling to get spatial features
            )

            # Get the feature extractor
            return model

        except Exception as e:
            logger.error(f"Failed to load timm model {self.model_name}: {e}")
            logger.info("Falling back to basic implementation")
            return self._create_fallback_backbone()

    def _create_fallback_backbone(self) -> nn.Module:
        """Create fallback vision backbone when timm is not available."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process images through vision encoder.

        Args:
            images: Input images tensor [batch_size, channels, height, width]

        Returns:
            Visual features tensor [batch_size, num_patches, feature_dim]
        """
        logger.debug(f"Vision input shape: {images.shape}")

        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self._forward_timm(images)

        logger.debug(f"Vision features shape: {features.shape}")
        return features

    def _forward_timm(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass using timm model."""
        # Resize images to model's expected input size if needed
        if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            logger.debug(f"Resized images to: {images.shape}")

        # Get features from timm model
        features = self.backbone(images)  # Shape depends on model architecture

        # Handle different output formats from different timm models
        if len(features.shape) == 4:  # CNN-style output [B, C, H, W]
            batch_size, channels, height, width = features.shape
            # Global average pooling to get fixed-size features
            features = F.adaptive_avg_pool2d(features, (1, 1))  # [B, C, 1, 1]
            features = features.flatten(1)  # [B, C]
            features = features.unsqueeze(1)  # [B, 1, C] for sequence compatibility
            logger.debug(f"CNN features processed to: {features.shape}")

        elif len(features.shape) == 3:  # Transformer-style output [B, seq_len, C]
            # Features are already in the right format for transformers
            logger.debug(f"Transformer features: {features.shape}")

        elif len(features.shape) == 2:  # Global pooled features [B, C]
            features = features.unsqueeze(1)  # [B, 1, C] for sequence compatibility
            logger.debug(f"Global features processed to: {features.shape}")

        return features

    def _forward_fallback(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass using fallback implementation."""
        features = self.backbone(images)  # [B, 256, 14, 14]
        logger.debug(f"Fallback backbone output: {features.shape}")

        # Flatten spatial dimensions
        features = features.flatten(1, -1)  # [B, 256*14*14]

        # Project to target feature dimension
        features = self.projection(features)  # [B, feature_dim]

        # Add sequence dimension for compatibility
        features = features.unsqueeze(1)  # [B, 1, feature_dim]

        return features

    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """List all available best-performing models by category."""
        return {
            category: list(models.keys())
            for category, models in cls.BEST_MODELS.items()
        }

    @classmethod
    def get_recommended_model(cls, performance_tier: str = "balanced") -> str:
        """
        Get recommended model based on performance tier.

        Args:
            performance_tier: "fast", "balanced", or "best"

        Returns:
            Recommended model name
        """
        recommendations = {
            "fast": "efficientnet_b0",
            "balanced": "convnext_base",
            "best": "convnext_large",
        }
        return recommendations.get(performance_tier, "convnext_base")


class MambaBlock(nn.Module):
    """
    Simplified Mamba SSM block implementation.
    This is a theoretical implementation of the core Mamba architecture.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        """
        Initialize Mamba block.

        Args:
            d_model: Model dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        logger.debug(
            f"Mamba block: d_model={d_model}, d_state={d_state}, d_inner={self.d_inner}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch, seqlen, dim = x.shape
        logger.debug(f"Mamba input shape: {x.shape}")

        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each: [B, L, d_inner]

        # Convolution for local context
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seqlen].transpose(
            1, 2
        )  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # SSM computation (simplified)
        x_ssm = self._selective_scan(x_conv)  # [B, L, d_inner]

        # Gate and output projection
        y = x_ssm * F.silu(z)  # [B, L, d_inner]
        output = self.out_proj(y)  # [B, L, d_model]

        logger.debug(f"Mamba output shape: {output.shape}")
        return output

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan operation.
        In practice, this would use efficient CUDA kernels.
        """
        # Simplified implementation - in practice this would be much more sophisticated
        batch, seqlen, d_inner = x.shape

        # Project to get selection parameters (simplified for fallback)
        _ = self.x_proj(x)  # [B, L, d_state*2]
        _ = F.softplus(self.dt_proj(x))  # [B, L, d_inner]

        # Simple recurrent computation (placeholder for actual selective scan)
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seqlen):
            # Simplified state update
            h = (
                h * 0.9
                + x[:, t] @ torch.randn(d_inner, self.d_state, device=x.device) * 0.1
            )
            y_t = h @ torch.randn(self.d_state, d_inner, device=x.device)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, L, d_inner]


class MambaSSM(nn.Module):
    """
    Multi-layer Mamba SSM for processing sequential data.
    Uses official mamba-ssm library when available, otherwise fallback implementation.
    """

    def __init__(self, d_model: int, n_layers: int = 6, d_state: int = 16):
        """
        Initialize Mamba SSM.

        Args:
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: State dimension for SSM
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_official = MAMBA_AVAILABLE

        if MAMBA_AVAILABLE:
            # Use official mamba-ssm implementation
            self.layers = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=d_state) for _ in range(n_layers)]
            )
            logger.info(f"Official Mamba SSM: {n_layers} layers, d_model={d_model}")
        else:
            # Use fallback implementation
            self.layers = nn.ModuleList(
                [MambaBlock(d_model, d_state) for _ in range(n_layers)]
            )
            logger.info(f"Fallback Mamba SSM: {n_layers} layers, d_model={d_model}")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba SSM.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        logger.debug(f"Mamba SSM input shape: {x.shape}")

        for i, layer in enumerate(self.layers):
            if self.use_official:
                # Official Mamba layers don't need residual connection as it's built-in
                x = layer(x)
            else:
                # Fallback implementation uses residual connection
                x = x + layer(x)
            logger.debug(f"After Mamba layer {i}: {x.shape}")

        x = self.norm(x)
        logger.debug(f"Mamba SSM output shape: {x.shape}")
        return x


class ActionHead(nn.Module):
    """
    Configurable action prediction head for different robot types.
    """

    def __init__(self, d_model: int, action_config: ActionConfig):
        """
        Initialize action head.

        Args:
            d_model: Input feature dimension
            action_config: Robot action configuration
        """
        super().__init__()
        self.action_config = action_config
        self.d_model = d_model

        # Action prediction layers
        self.action_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, action_config.action_dim),
            nn.Tanh(),  # Normalize to [-1, 1]
        )

        logger.info(
            f"Action head for {action_config.robot_type.value}: {action_config.action_dim} actions"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict actions from features.

        Args:
            features: Input features [batch_size, seq_len, d_model]

        Returns:
            Actions tensor [batch_size, seq_len, action_dim]
        """
        logger.debug(f"Action head input shape: {features.shape}")

        actions = self.action_layers(features)  # [B, L, action_dim]

        # Scale to action bounds
        low, high = self.action_config.action_bounds
        actions = actions * (high - low) / 2.0 + (high + low) / 2.0

        logger.debug(f"Action head output shape: {actions.shape}")
        return actions


class LanguageHead(nn.Module):
    """
    Language generation head for describing robot actions and observations.
    """

    def __init__(self, d_model: int, vocab_size: int = 32000):
        """
        Initialize language head.

        Args:
            d_model: Input feature dimension
            vocab_size: Vocabulary size for language generation
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.language_layers = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, vocab_size)
        )

        logger.info(f"Language head: vocab_size={vocab_size}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate language logits from features.

        Args:
            features: Input features [batch_size, seq_len, d_model]

        Returns:
            Language logits [batch_size, seq_len, vocab_size]
        """
        logger.debug(f"Language head input shape: {features.shape}")

        logits = self.language_layers(features)  # [B, L, vocab_size]

        logger.debug(f"Language head output shape: {logits.shape}")
        return logits


class VisionLanguageActionModel(nn.Module):
    """
    Main Vision-Language-Action model combining all components.

    Architecture:
    1. Vision Encoder: Processes input images
    2. Mamba SSM: Processes sequential visual features
    3. Dual Heads: Generates language and actions simultaneously
    """

    def __init__(
        self,
        action_config: ActionConfig,
        vision_model: str = "convnext_base",
        d_model: int = 1024,
        n_layers: int = 6,
        d_state: int = 16,
        vocab_size: int = 32000,
        freeze_vision: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize VLA model.

        Args:
            action_config: Robot action configuration
            vision_model: Pretrained vision model name from timm (e.g., 'convnext_base', 'vit_base_patch16_224', 'efficientnet_b0')
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: SSM state dimension
            vocab_size: Language vocabulary size
            freeze_vision: Whether to freeze vision encoder
            verbose: Enable verbose logging
        """
        super().__init__()
        self.verbose = verbose
        self.action_config = action_config
        self.d_model = d_model

        # Vision encoder
        self.vision_encoder = VisionEncoder(vision_model, freeze_vision)

        # Project vision features to model dimension
        self.vision_proj = nn.Linear(self.vision_encoder.feature_dim, d_model)

        # Mamba SSM for sequential processing
        self.mamba = MambaSSM(d_model, n_layers, d_state)

        # Dual output heads
        self.action_head = ActionHead(d_model, action_config)
        self.language_head = LanguageHead(d_model, vocab_size)

        # Positional encoding for sequences
        self.pos_encoding = self._create_positional_encoding(d_model)

        logger.info("VLA Model initialized:")
        logger.info(f"  Robot type: {action_config.robot_type.value}")
        logger.info(f"  Action dim: {action_config.action_dim}")
        logger.info(f"  Model dim: {d_model}")
        logger.info(f"  Mamba layers: {n_layers}")

        # if self.verbose is False:
        #     logger.disable("vlam")

    def _create_positional_encoding(
        self, d_model: int, max_len: int = 5000
    ) -> torch.Tensor:
        """Create positional encoding for sequence modeling."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(
        self, images: torch.Tensor, sequence_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VLA model.

        Args:
            images: Input images [batch_size, seq_len, channels, height, width]
                   or [batch_size, channels, height, width] for single timestep
            sequence_length: Optional sequence length for truncation

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, seq_len, action_dim]
                - 'language_logits': Language logits [batch_size, seq_len, vocab_size]
                - 'features': Intermediate features [batch_size, seq_len, d_model]
        """
        # Handle single image input
        if len(images.shape) == 4:
            images = images.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, channels, height, width = images.shape
        logger.info(
            f"Processing sequence: batch={batch_size}, seq_len={seq_len}, img_shape=({channels},{height},{width})"
        )

        # Process images through vision encoder
        # Flatten batch and sequence dimensions for vision processing
        images_flat = images.view(-1, channels, height, width)  # [B*L, C, H, W]
        visual_features = self.vision_encoder(
            images_flat
        )  # [B*L, num_patches, feature_dim]

        # Reshape back to sequence format and pool spatial features
        num_patches, feature_dim = visual_features.shape[1], visual_features.shape[2]
        visual_features = visual_features.view(
            batch_size, seq_len, num_patches, feature_dim
        )

        # Global average pooling over spatial patches
        visual_features = visual_features.mean(dim=2)  # [B, L, feature_dim]
        logger.debug(f"Pooled visual features shape: {visual_features.shape}")

        # Project to model dimension
        features = self.vision_proj(visual_features)  # [B, L, d_model]
        logger.debug(f"Projected features shape: {features.shape}")

        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(features.device)
        features = features + pos_enc

        # Process through Mamba SSM
        mamba_features = self.mamba(features)  # [B, L, d_model]

        # Generate outputs through dual heads
        actions = self.action_head(mamba_features)  # [B, L, action_dim]
        language_logits = self.language_head(mamba_features)  # [B, L, vocab_size]

        # Truncate if sequence_length specified
        if sequence_length is not None:
            actions = actions[:, :sequence_length]
            language_logits = language_logits[:, :sequence_length]
            mamba_features = mamba_features[:, :sequence_length]

        logger.info(
            f"Output shapes - Actions: {actions.shape}, Language: {language_logits.shape}"
        )

        return {
            "actions": actions,
            "language_logits": language_logits,
            "features": mamba_features,
        }

    def predict_next_action(self, images: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Predict next action and description from current observation.

        Args:
            images: Current observation [batch_size, channels, height, width]

        Returns:
            Tuple of (predicted_action, description_text)
        """
        with torch.no_grad():
            outputs = self.forward(images)

            # Get last timestep predictions
            next_action = outputs["actions"][:, -1]  # [B, action_dim]
            language_logits = outputs["language_logits"][:, -1]  # [B, vocab_size]

            # Convert language logits to text (simplified)
            predicted_tokens = torch.argmax(language_logits, dim=-1)  # [B]
            description = f"Token_{predicted_tokens[0].item()}"  # Placeholder for actual tokenizer

        logger.info(f"Predicted action: {next_action[0].tolist()}")
        logger.info(f"Description: {description}")

        return next_action, description

    def get_action_description(self) -> Dict[str, any]:
        """Get human-readable action configuration."""
        return {
            "robot_type": self.action_config.robot_type.value,
            "action_dimension": self.action_config.action_dim,
            "action_bounds": self.action_config.action_bounds,
            "joint_names": self.action_config.joint_names,
            "vision_model": self.vision_encoder.model_name,
            "vision_feature_dim": self.vision_encoder.feature_dim,
            "vision_input_size": self.vision_encoder.input_size,
            "model_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    @classmethod
    def list_supported_vision_models(cls) -> Dict[str, List[str]]:
        """List all supported vision models by category."""
        return VisionEncoder.list_available_models()

    @classmethod
    def get_recommended_vision_model(cls, performance_tier: str = "balanced") -> str:
        """
        Get recommended vision model based on performance requirements.

        Args:
            performance_tier: "fast", "balanced", or "best"

        Returns:
            Recommended model name
        """
        return VisionEncoder.get_recommended_model(performance_tier)

    @classmethod
    def create_with_recommended_vision(
        cls, action_config: ActionConfig, performance_tier: str = "balanced", **kwargs
    ) -> "VisionLanguageActionModel":
        """
        Create VLA model with recommended vision model for given performance tier.

        Args:
            action_config: Robot action configuration
            performance_tier: "fast", "balanced", or "best"
            **kwargs: Additional arguments passed to constructor

        Returns:
            VLA model instance
        """
        vision_model = cls.get_recommended_vision_model(performance_tier)
        return cls(action_config=action_config, vision_model=vision_model, **kwargs)
