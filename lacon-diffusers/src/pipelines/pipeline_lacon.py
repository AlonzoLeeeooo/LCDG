import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput, logging, randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.condition_aligner import ConditionAligner
from ..utils.feature_extractor import UNetFeatureExtractor


logger = logging.get_logger(__name__)


@dataclass
class LaConPipelineOutput(BaseOutput):
    """
    Output class for LaCon pipeline.
    
    Args:
        images: List of generated PIL images
        nsfw_content_detected: List of flags indicating whether NSFW content was detected
        predicted_conditions: Optional list of predicted conditions (for visualization)
    """
    images: List[Image.Image]
    nsfw_content_detected: Optional[List[bool]]
    predicted_conditions: Optional[List[Image.Image]] = None


class LaConPipeline(DiffusionPipeline):
    """
    Pipeline for Late-Constraint Diffusion (LaCon) with Stable Diffusion.
    
    This pipeline implements controllable image generation by incorporating
    external conditions (edges, masks, color strokes) during the diffusion
    sampling process through gradient guidance.
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        condition_aligner: ConditionAligner,
        safety_checker: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        requires_safety_checker: bool = False,
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
        self.requires_safety_checker = requires_safety_checker
        
        # Feature extraction settings for different SD versions
        self.feature_blocks = {
            "sd_v1_4": [4, 5, 7, 11],  # Default blocks for SD 1.4
            "sd_v1_5": [4, 5, 7, 11],  # Default blocks for SD 1.5
            "sd_v2_1": [4, 5, 7, 11],  # May need adjustment for SD 2.1
        }
        
        # Initialize feature extractor
        self.feature_extractor_unet = UNetFeatureExtractor(self.unet)
        
    def _extract_unet_features(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_indices: List[int],
    ) -> List[torch.Tensor]:
        """
        Extract intermediate features from UNet at specified block indices.
        
        Args:
            latents: Input latents
            timestep: Current timestep
            encoder_hidden_states: Text embeddings
            block_indices: List of block indices to extract features from
            
        Returns:
            List of feature tensors from specified blocks
        """
        # Extract features using the feature extractor
        feature_dict = self.feature_extractor_unet.extract_features(
            latents, timestep, encoder_hidden_states, block_indices
        )
        
        # Convert dict to list maintaining order
        features = []
        for idx in block_indices:
            block, name = self.feature_extractor_unet._get_block_by_index(idx)
            if name in feature_dict:
                features.append(feature_dict[name])
                
        return features
    
    def _upsample_features(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsample and concatenate features to target size.
        
        Args:
            features: List of feature tensors
            target_size: Target spatial size (H, W)
            
        Returns:
            Concatenated upsampled features
        """
        upsampled = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            upsampled.append(feat)
            
        return torch.cat(upsampled, dim=1)
    
    def _compute_condition_gradient(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_condition: torch.Tensor,
        condition_scale: float = 2.0,
        block_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient from condition aligner to guide the sampling.
        
        Args:
            latents: Current latents requiring gradients
            timestep: Current timestep
            encoder_hidden_states: Text embeddings
            target_condition: Target condition in latent space
            condition_scale: Scale for condition guidance
            block_indices: Indices of UNet blocks to extract features from
            
        Returns:
            Gradient tensor and predicted condition
        """
        if block_indices is None:
            block_indices = self.feature_blocks["sd_v1_4"]
            
        latents = latents.requires_grad_(True)
        
        # Extract features from UNet
        features = self._extract_unet_features(
            latents, timestep, encoder_hidden_states, block_indices
        )
        
        # Upsample and concatenate features
        spatial_size = latents.shape[-2:]
        upsampled_features = self._upsample_features(features, spatial_size)
        
        # Predict condition using the aligner
        predicted_condition = self.condition_aligner(upsampled_features, timestep)
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_condition, target_condition, reduction='sum')
        
        # Compute gradient
        gradient = torch.autograd.grad(loss, latents)[0]
        
        # Normalize gradient
        gradient = gradient * condition_scale
        
        return gradient, predicted_condition
    
    def prepare_condition(
        self,
        condition: Union[torch.Tensor, Image.Image, np.ndarray],
        height: int,
        width: int,
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        condition_type: str = "edge",
    ) -> torch.Tensor:
        """
        Prepare condition input for the pipeline.
        
        Args:
            condition: Input condition (image, tensor, etc.)
            height: Target height
            width: Target width
            batch_size: Batch size
            num_images_per_prompt: Number of images per prompt
            device: Device to place tensor on
            dtype: Data type for tensor
            condition_type: Type of condition (edge, mask, color, etc.)
            
        Returns:
            Prepared condition tensor in latent space
        """
        # Convert to tensor if needed
        if isinstance(condition, Image.Image):
            condition = np.array(condition)
            
        if isinstance(condition, np.ndarray):
            condition = torch.from_numpy(condition).float()
            
        # Ensure correct shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)
        elif condition.dim() == 3:
            condition = condition.unsqueeze(0)
            
        # Resize if needed
        if condition.shape[-2:] != (height, width):
            condition = F.interpolate(
                condition, size=(height, width), mode='bilinear', align_corners=False
            )
            
        # Apply condition-specific preprocessing
        if condition_type == "edge":
            # Threshold for edge maps
            condition = (condition > 0.5).float()
        elif condition_type == "mask":
            # Binary threshold for masks
            condition = (condition > 0.5).float()
            
        # Normalize to [-1, 1]
        condition = condition * 2.0 - 1.0
        
        # Encode to latent space
        condition = condition.to(device=device, dtype=dtype)
        condition_latents = self.vae.encode(condition).latent_dist.sample()
        condition_latents = condition_latents * self.vae.config.scaling_factor
        
        # Duplicate for batch
        condition_latents = condition_latents.repeat(
            batch_size * num_images_per_prompt, 1, 1, 1
        )
        
        return condition_latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        condition: Union[torch.Tensor, Image.Image, np.ndarray] = None,
        condition_type: str = "edge",
        condition_scale: float = 2.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        truncation_steps: int = 500,
        return_predicted_condition: bool = False,
        **kwargs,
    ):
        """
        Generate images with condition guidance using Late-Constraint Diffusion.
        
        Args:
            prompt: Text prompt(s) for generation
            condition: Condition input (edge map, mask, color stroke, etc.)
            condition_type: Type of condition ("edge", "mask", "color", etc.)
            condition_scale: Scale for condition guidance strength
            height: Height of generated image
            width: Width of generated image
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt(s)
            num_images_per_prompt: Number of images to generate per prompt
            eta: DDIM eta parameter
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            output_type: Output format ("pil", "latent", "numpy")
            return_dict: Whether to return a dataclass output
            callback: Callback function during generation
            callback_steps: Steps between callback calls
            truncation_steps: Steps to apply condition guidance
            return_predicted_condition: Whether to return predicted conditions
            
        Returns:
            LaConPipelineOutput or tuple of generated images
        """
        # 1. Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
            
        device = self._execution_device
        
        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            True,  # do_classifier_free_guidance
            negative_prompt,
        )
        
        # 4. Prepare condition
        if condition is not None:
            condition_latents = self.prepare_condition(
                condition,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
                batch_size,
                num_images_per_prompt,
                device,
                text_embeddings.dtype,
                condition_type,
            )
        else:
            raise ValueError("Condition input is required for LaCon pipeline")
        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        predicted_conditions = []
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # Apply condition guidance if within truncation steps
                if t > truncation_steps and condition_latents is not None:
                    # Enable gradients temporarily
                    with torch.enable_grad():
                        latents_with_grad = latents.clone().requires_grad_(True)
                        
                        # Compute condition gradient
                        gradient, pred_condition = self._compute_condition_gradient(
                            latents_with_grad,
                            t,
                            text_embeddings.chunk(2)[1],  # Use conditional embeddings
                            condition_latents[:latents.shape[0]],  # Match batch size
                            condition_scale,
                        )
                        
                        # Store predicted condition if requested
                        if return_predicted_condition and i % 10 == 0:
                            predicted_conditions.append(pred_condition.detach())
                    
                    # Apply gradient to noise prediction
                    # Normalize gradient by relative norms
                    grad_norm = torch.norm(gradient.flatten(), p=2)
                    noise_norm = torch.norm((noise_pred - latents).flatten(), p=2)
                    
                    if grad_norm > 0:
                        gradient = gradient * noise_norm / grad_norm
                        noise_pred = noise_pred - gradient
                
                # Compute previous noisy sample
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                
                # Call callback if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # 9. Post-processing
        image = self.decode_latents(latents)
        
        # 10. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )
        
        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        # 12. Process predicted conditions if requested
        predicted_condition_images = None
        if return_predicted_condition and predicted_conditions:
            # Decode predicted conditions
            pred_cond_tensor = predicted_conditions[-1]  # Use last prediction
            pred_cond_decoded = self.vae.decode(
                pred_cond_tensor / self.vae.config.scaling_factor
            ).sample
            pred_cond_decoded = (pred_cond_decoded / 2 + 0.5).clamp(0, 1)
            pred_cond_decoded = pred_cond_decoded.cpu().permute(0, 2, 3, 1).numpy()
            predicted_condition_images = self.numpy_to_pil(pred_cond_decoded)
        
        if not return_dict:
            return (image, has_nsfw_concept)
        
        return LaConPipelineOutput(
            images=image,
            nsfw_content_detected=has_nsfw_concept,
            predicted_conditions=predicted_condition_images,
        )
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Encode the prompt into text embeddings.
        """
        if prompt_embeds is None:
            # Get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to(device)
            )[0]
            
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        # Duplicate for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        
        # Get unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * len(prompt)
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, "
                    f"but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has length {len(negative_prompt)}, "
                    f"but `prompt`: {prompt} has length {len(prompt)}. "
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
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device)
            )[0]
            
        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                len(prompt) * num_images_per_prompt, seq_len, -1
            )
            
            # Concatenate for classifier-free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        return prompt_embeds
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Prepare latent variables."""
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed {batch_size} prompts, "
                f"but only {len(generator)} generators."
            )
            
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
            
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def decode_latents(self, latents):
        """Decode latents to images."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # We always cast to float32 as this does not cause significant overhead
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for scheduler step."""
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same args
        # eta (η) is only used with the DDIMScheduler
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            
        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def run_safety_checker(self, image, device, dtype):
        """Run safety checker if available."""
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept