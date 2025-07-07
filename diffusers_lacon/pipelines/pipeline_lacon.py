import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor

from ..models.condition_aligner import ConditionAligner

logger = logging.get_logger(__name__)


class LaConPipeline(DiffusionPipeline):
    """
    LaCon: Late-Constraint Diffusion Pipeline for Controlled Image Generation
    
    This pipeline implements the LaCon method for adding controllability to pre-trained
    diffusion models using a condition aligner that operates on intermediate features.
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        condition_aligner: ConditionAligner,
        safety_checker: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            condition_aligner=condition_aligner,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        # Feature extraction configuration
        self.feature_blocks = [[2, 4, 8], [2, 4, 8, 12]]  # Default block indices
        self.condition_size = 64  # Default condition size
        
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Encode text prompt to embeddings"""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        """Decode latents to images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step"""
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """Prepare initial latents"""
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def extract_features(self, latents, timesteps, prompt_embeds):
        """Extract features from UNet at specified blocks"""
        # This is a simplified version - in practice, you'd need to modify the UNet
        # to return intermediate features from specified blocks
        with torch.no_grad():
            noise_pred = self.unet(latents, timesteps, prompt_embeds).sample
            
        # For now, we'll use a placeholder - in real implementation,
        # you'd need to hook into UNet's forward pass to get intermediate features
        features = []
        for block_indices in self.feature_blocks:
            # Placeholder: in practice, extract features from specific blocks
            feature = torch.randn(
                latents.shape[0], 
                1280,  # Typical feature dimension
                self.condition_size, 
                self.condition_size, 
                device=latents.device
            )
            features.append(feature)
            
        return torch.cat(features, dim=1)

    def condition_function(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        target_condition: torch.Tensor,
        condition_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute gradients for condition alignment
        
        Args:
            latents: Current latents
            timesteps: Current timesteps
            prompt_embeds: Text embeddings
            target_condition: Target condition image
            condition_scale: Scale for condition guidance
            
        Returns:
            Gradients for condition alignment
        """
        latents = latents.detach().requires_grad_(True)
        
        # Extract features from UNet
        features = self.extract_features(latents, timesteps, prompt_embeds)
        
        # Predict condition using aligner
        condition_pred = self.condition_aligner(features, timesteps)["condition_pred"]
        
        # Compute loss
        loss = F.mse_loss(condition_pred, target_condition)
        
        # Compute gradients
        grad = torch.autograd.grad(loss, latents)[0]
        
        return grad * condition_scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        condition_image: Union[torch.Tensor, Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        condition_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        truncation_steps: int = 1000,
        **kwargs,
    ):
        """
        Generate images with condition guidance using LaCon
        
        Args:
            prompt: Text prompt for generation
            condition_image: Condition image (e.g., edge map, mask, etc.)
            height: Height of generated images
            width: Width of generated images
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            condition_scale: Condition guidance scale
            negative_prompt: Negative prompt
            num_images_per_prompt: Number of images per prompt
            eta: DDIM eta parameter
            generator: Random generator
            latents: Initial latents
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output type ('pil' or 'np')
            return_dict: Whether to return dict
            callback: Callback function
            callback_steps: Callback frequency
            truncation_steps: Steps to stop applying condition
            
        Returns:
            Generated images
        """
        # Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Check inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # Prepare condition
        if condition_image is not None:
            if isinstance(condition_image, Image.Image):
                condition_image = torch.from_numpy(np.array(condition_image)).permute(2, 0, 1).unsqueeze(0)
            condition_image = condition_image.to(device, dtype=latents.dtype)
            
            # Encode condition to latent space
            condition_latents = self.vae.encode(condition_image).latent_dist.sample()
            condition_latents = condition_latents * self.vae.config.scaling_factor
        else:
            condition_latents = None
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            
            # Perform classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply condition guidance (only for first truncation_steps)
            if condition_latents is not None and i < truncation_steps:
                # Get condition gradients
                condition_grad = self.condition_function(
                    latents,
                    t.unsqueeze(0).expand(latents.shape[0]),
                    prompt_embeds[len(latents):] if do_classifier_free_guidance else prompt_embeds,
                    condition_latents,
                    condition_scale,
                )
                
                # Apply condition guidance
                noise_pred = noise_pred - condition_grad
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            # Call callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        image = self.decode_latents(latents)
        
        # Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return (image,)
        
        return {"images": image}