# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.models.vision_utils import resize
from sglang.multimodal_gen.utils import calculate_dimensions


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def qwen_image_preprocess_text(prompt):
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    template = prompt_template_encode
    txt = template.format(prompt)
    return txt


def qwen_image_postprocess_text(outputs, _text_inputs, drop_idx=34):
    # squeeze the batch dim
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(
        hidden_states, _text_inputs.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ]
    )
    return prompt_embeds


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


@dataclass
class QwenImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the QwenImage pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=QwenImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=QwenImageVAEConfig)

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5VLConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_postprocess_text,)
    )
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=True,
                truncation=True,
            ),
            None,
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def prepare_image_processor_kwargs(self, batch):
        if batch.prompt:
            prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            txt = prompt_template_encode.format(batch.prompt)
            return dict(text=[txt], padding=True)
        else:
            return {}

    def get_vae_scale_factor(self):
        return self.vae_config.arch_config.vae_scale_factor

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (batch.height // (vae_scale_factor * 2))
        width = 2 * (batch.width // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, 1, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch_config = self.vae_config.arch_config
        scaling_factor = 1.0 / torch.tensor(
            vae_arch_config.latents_std, device=device
        ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(device, dtype)
        shift_factor = (
            torch.tensor(vae_arch_config.latents_mean)
            .view(1, vae_arch_config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        return scaling_factor, shift_factor

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
        # img_shapes: for global entire image
        img_freqs, txt_freqs = rotary_emb(img_shapes, txt_seq_lens, device=device)

        img_cos, img_sin = (
            img_freqs.real.to(dtype=dtype),
            img_freqs.imag.to(dtype=dtype),
        )
        txt_cos, txt_sin = (
            txt_freqs.real.to(dtype=dtype),
            txt_freqs.imag.to(dtype=dtype),
        )

        return (img_cos, img_sin), (txt_cos, txt_sin)

    def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
        batch_size = prompt_embeds[0].shape[0]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        (img_cos, img_sin), (txt_cos, txt_sin) = self.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def post_denoising_loop(self, latents, batch):
        # unpack latents for qwen-image
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents


@dataclass
class QwenImageEditPipelineConfig(QwenImagePipelineConfig):
    """Configuration for the QwenImageEdit pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I

    def _prepare_edit_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype
    ):
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width
        image_size = batch.original_condition_image_size
        edit_width, edit_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                ),
                (
                    1,
                    edit_height // vae_scale_factor // 2,
                    edit_width // vae_scale_factor // 2,
                ),
            ],
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]
        (img_cos, img_sin), (txt_cos, txt_sin) = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        # perform sp shard on noisy image tokens
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )

        noisy_img_cos = shard_rotary_emb_for_sp(img_cos[:noisy_img_seq_len, :])
        noisy_img_sin = shard_rotary_emb_for_sp(img_sin[:noisy_img_seq_len, :])

        # concat back the img_cos for input image (since it is not sp-shared later)
        img_cos = torch.cat([noisy_img_cos, img_cos[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )
        img_sin = torch.cat([noisy_img_sin, img_sin[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )

        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def preprocess_condition_image(
        self, image, target_width, target_height, _vae_image_processor
    ):
        return resize(image, target_height, target_width, resize_mode="default"), (
            target_width,
            target_height,
        )

    def postprocess_image_latent(self, latent_condition, batch):
        batch_size = batch.batch_size
        if batch_size > latent_condition.shape[0]:
            if batch_size % latent_condition.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // latent_condition.shape[0]
                image_latents = latent_condition.repeat(
                    additional_image_per_prompt, 1, 1, 1
                )
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latent_condition.shape[0]} to {batch_size} text prompts."
                )
        else:
            image_latents = latent_condition
        image_latent_height, image_latent_width = image_latents.shape[3:]
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        image_latents = _pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

        return image_latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def calculate_condition_image_size(self, image, width, height) -> tuple[int, int]:
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, width / height
        )
        return calculated_width, calculated_height

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1)]
        return noise


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
from typing import List, Optional, Union


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    import inspect

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenImageEditPlusPipelineConfig(QwenImageEditPipelineConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from diffusers.image_processor import VaeImageProcessor

        self.image_processor = VaeImageProcessor(vae_scale_factor=8 * 2)

    # from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    # from sglang.multimodal_gen.runtime.server_args import ServerArgs
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        processor=None,
        text_encoder=None,
    ):

        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

        device = device or self._execution_device
        dtype = torch.bfloat16
        text_encoder = text_encoder
        processor = processor or self.processor

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        processor=None,
        text_encoder=None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with set_forward_context(current_timestep=0, attn_metadata=None):
            device = device or self._execution_device

            prompt = [prompt] if isinstance(prompt, str) else prompt

            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                processor=processor,
                text_encoder=text_encoder,
                prompt=prompt,
                image=image,
                device=device,
            )

        return prompt_embeds, prompt_embeds_mask

    def before_denosing(
        self,
        batch,
        server_args,
        processor,
        text_encoder,
    ):
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
        from sglang.multimodal_gen.runtime.models.vision_utils import (
            load_image,
            load_video,
        )

        height = batch.height
        width = batch.width
        guidance_scale = batch.guidance_scale
        prompt = batch.prompt
        negative_prompt = batch.negative_prompt
        true_cfg_scale = batch.guidance_scale
        generator = batch.generator
        num_inference_steps = batch.num_inference_steps
        sigmas = batch.sigmas
        num_images_per_prompt = 1
        vae_scale_factor = 8  # self.get_vae_scale_factor()

        processor = processor
        text_encoder = text_encoder

        images = []
        if batch.image_path is not None and not isinstance(batch.image_path, list):
            batch.image_path = [batch.image_path]

        print(f"batch.image_path: {batch.image_path}", flush=True)
        image = []
        if batch.image_path is not None:
            for image_path in batch.image_path:
                print(f"image_path: {image_path}", flush=True)
                if image_path.endswith(".mp4"):
                    sinele_image = load_video(image_path)[0]
                else:
                    sinele_image = load_image(image_path)
                image.append(sinele_image)

        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     height,
        #     width,
        # )

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = get_local_torch_device()

        # 3. Preprocess image
        if image is not None and not (
            isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
        ):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height, _ = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height, _ = calculate_dimensions(
                    VAE_IMAGE_SIZE, image_width / image_height
                )
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(
                    self.image_processor.resize(img, condition_height, condition_width)
                )
                vae_images.append(
                    self.image_processor.preprocess(
                        img, vae_height, vae_width
                    ).unsqueeze(2)
                )

        has_neg_prompt = negative_prompt is not None

        # if true_cfg_scale > 1 and not has_neg_prompt:
        #     logger.warning(
        #         f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        #     )
        # elif true_cfg_scale <= 1 and has_neg_prompt:
        #     logger.warning(
        #         " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        #     )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            device=device,
            processor=processor,
            text_encoder=text_encoder,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                device=device,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = None
        latents, image_latents = (
            self.prepare_latents(  # Consider: use prepare_latents in
                vae_images,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                ),
                *[
                    (
                        1,
                        vae_height // self.vae_scale_factor // 2,
                        vae_width // self.vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 5. Prepare timesteps
        import numpy as np

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            # logger.warning(
            #     f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            # )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )


# 465 latents.shape=torch.Size([1, 3600, 64])
# 466 prompt_embeds[0].shape=torch.Size([1, 1402, 3584])
