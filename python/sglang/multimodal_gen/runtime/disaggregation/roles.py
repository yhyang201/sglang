# SPDX-License-Identifier: Apache-2.0
"""
Role definitions for diffusion pipeline disaggregation.

Each diffusion pipeline can be decomposed into three roles:
- ENCODER: Text/image encoding, latent preparation, timestep preparation
- DENOISING: Iterative denoising loop (the compute-heavy part)
- DECODER: VAE decode from latent space to pixel space
"""

from enum import Enum


class RoleType(str, Enum):
    """Role type for disaggregated diffusion pipelines."""

    MONOLITHIC = "monolithic"  # Default: load everything, run all stages
    ENCODER = "encoder"  # Text/image encoding + latent/timestep prep
    DENOISING = "denoising"  # Denoising loop only
    DECODER = "decoder"  # VAE decode only
    SERVER = "server"  # DiffusionServer head node (no GPU, routes requests)

    @classmethod
    def from_string(cls, value: str) -> "RoleType":
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid role: {value}. Must be one of: {', '.join([r.value for r in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        return [role.value for role in cls]


def get_module_role(module_name: str) -> "RoleType | None":
    """Classify a module name to its primary role.

    Returns None for shared/lightweight modules (e.g., scheduler) that should
    always be loaded regardless of role.
    """
    # Encoder-specific modules (text encoders, tokenizers, image encoders)
    encoder_prefixes = (
        "text_encoder",
        "tokenizer",
        "image_encoder",
        "image_processor",
        "processor",
        "connectors",
    )
    if any(
        module_name == p or module_name.startswith(p + "_") for p in encoder_prefixes
    ):
        return RoleType.ENCODER

    # Denoising-specific modules (transformer / DiT)
    denoising_prefixes = ("transformer",)
    if any(
        module_name == p or module_name.startswith(p + "_") for p in denoising_prefixes
    ):
        return RoleType.DENOISING

    # Decoder-specific modules (VAE, vocoder)
    decoder_prefixes = ("vae", "audio_vae", "video_vae", "vocoder")
    if any(
        module_name == p or module_name.startswith(p + "_") for p in decoder_prefixes
    ):
        return RoleType.DECODER

    # Shared modules (scheduler, etc.) - always loaded
    return None


def filter_modules_for_role(module_names: list[str], role: "RoleType") -> list[str]:
    """Filter module names to only those needed by the given role.

    For MONOLITHIC role, returns all modules unchanged.

    Module loading rules per role:
    - ENCODER: encoder modules + decoder modules (VAE needed for ImageVAEEncoding) + shared
    - DENOISING: denoising modules + shared (no VAE, no text encoders)
    - DECODER: decoder modules + shared
    """
    if role in (RoleType.MONOLITHIC, RoleType.SERVER):
        return module_names

    filtered = []
    for name in module_names:
        module_role = get_module_role(name)

        if module_role is None:
            # Shared module (scheduler, etc.) - always include
            filtered.append(name)
        elif module_role == role:
            # Module belongs to this role
            filtered.append(name)
        elif role == RoleType.ENCODER and module_role == RoleType.DECODER:
            # Encoder also needs VAE for ImageVAEEncoding stages
            filtered.append(name)

    return filtered
