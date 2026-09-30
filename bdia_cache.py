import os
from glob import glob
import copy
from pickle import FALSE
from typing import Optional,Dict
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import click

import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)
from video_diffusion.pipelines.BDIAScheduler import BDIAScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange

from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel

from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.logger import get_logger_config_path
from video_diffusion.common.image_util import log_train_samples
from video_diffusion.common.instantiate_from_config import instantiate_from_config
from video_diffusion.pipelines.p2p_validation_loop import P2pSampleLogger


from typing import List, Optional, Tuple, Union
import torch.utils.checkpoint
from diffusers.utils import BaseOutput, logging
from typing import Union, Tuple  # 如果还没导入就加上
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


from types import MethodType
import numpy as np

from fvcore.nn import FlopCountAnalysis

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# logger = get_logger(__name__)

from contextlib import contextmanager

@contextmanager
def teacache_disabled(unet):
    # 保存旧值，关闭，再恢复
    old = getattr(unet, "enable_teacache", False)
    try:
        unet.enable_teacache = False
        yield
    finally:
        unet.enable_teacache = old


def teacache_forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    Args:
        sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
        timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
        encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

    Returns:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.

    # --------------- 懒初始化（不改 __init__） ---------------
    try:
        _ = self.enable_teacache
        self.use_output_extrapolation = True
    except AttributeError:
        # 外部可随时改这些开关/超参
        self.enable_teacache = False  # 开/关 TeaCache
        self.num_steps = 50  # 与调度器步数对齐
        self.rel_l1_thresh =  0.3 # 累计阈值（经验 0.08~0.20）
        print(1234)
        print(self.rel_l1_thresh)
        self.use_output_extrapolation = True  # 跳步时是否做输出残差外推

        # 状态
        self._tc_cnt = 0
        self._tc_accum = 0
        self._tc_prev_mod_inp = None  # 上一次“用于比较的输入”（见下）
        self._tc_prev_out = None  # 上一次最终输出
        self._tc_prev_out_residual = None  # 输出残差 y_{t-1}-y_{t-2}

        # # ===== FLOPs 相关计数（新增）=====
        # self.unet_call_cnt = 0   # 真正完整跑 UNet 的次数（没有被 TeaCache 复用）
        # self.unet_skip_cnt = 0   # 被 TeaCache 直接复用输出的次数

    # ------------ 原版前半：准备尺寸/掩码/时间嵌入 ------------

    default_overall_up_factor = 2 ** self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        logger.info("Forward upsample size to force interpolation output size.")
        forward_upsample_size = True

    # prepare attention_mask
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        emb = emb + class_emb

    # 2. pre-process
    sample = self.conv_in(sample)

    # ------------------ TeaCache 判定（conv_in 之后、down 之前） ------------------
    self._tc_last_cache_hit = False

    if self.enable_teacache:
        # 用简易“尺度归一”后的 sample 作为可比较特征（不引入新参数）
        denom = sample.detach().abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        modulated_inp = sample / denom  # 近似“调制后的输入”

        #if (self._tc_cnt == 0) or (self._tc_cnt == self.num_steps - 1) or (self._tc_prev_mod_inp is None):
        if (self._tc_cnt == 0) or (self._tc_cnt == self.num_steps - 1):
            should_compute = True
            self._tc_accum = 0.0
        else:
            # 相对 L1
            rel_l1 = (modulated_inp - self._tc_prev_mod_inp).abs().mean() / (
                    self._tc_prev_mod_inp.abs().mean() + 1e-8
            )
            x = rel_l1.detach().float()

            # 四次多项式重标定（Horner 形式）
            a0 = 7.33226126e+02
            a1 = -4.01131952e+02
            a2 = 6.75869174e+01
            a3 = -3.14987800e+00
            a4 = 9.61237896e-02
            rescaled = (((a0 * x + a1) * x + a2) * x + a3) * x + a4
            rescaled = torch.clamp(rescaled, min=0.0)

            self._tc_accum += float(rescaled)

            if (self._tc_accum < self.rel_l1_thresh) and (self._tc_prev_out is not None):
                should_compute = False
            else:
                should_compute = True
                self._tc_accum = 0.0

        # 记录 & 推进计数
        self._tc_prev_mod_inp = modulated_inp.detach()
        self._tc_cnt += 1
        if self._tc_cnt == self.num_steps:
            self._tc_cnt = 0

        # 跳步：直接复用上次输出（可选加一次输出残差外推）
        if not should_compute:
            self._tc_last_cache_hit = True

            self.unet_skip_cnt += 1
            fast_out = self._tc_prev_out
            if (fast_out is not None) and self.use_output_extrapolation and (
                    self._tc_prev_out_residual is not None):
                fast_out = fast_out + self._tc_prev_out_residual
            return UNet2DConditionOutput(sample=fast_out)

    # 3. down
    # ===== FLOPs 相关计数（新增）=====
    if self.enable_teacache:
        self.unet_call_cnt += 1   # ← 新增：只有没被跳步的 forward 才会走到这里

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    # 4. mid
    sample = self.mid_block(
        sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
    )

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
            )

    # 6. post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    # ------------------ TeaCache 更新（conv_out 之后） ------------------
    # if self.enable_teacache:
    #     if self._tc_prev_out is not None:
    #         self._tc_prev_out_residual = (sample - self._tc_prev_out).detach()
    #     self._tc_prev_out = sample.detach()
    #
    # if not return_dict:
    #     return (sample,)

    # ------------------ TeaCache 更新（conv_out 之后） ------------------
    if self.enable_teacache:
        if self._tc_prev_out is not None:
            self._tc_prev_out_residual = (sample - self._tc_prev_out).detach()
        self._tc_prev_out = sample.detach()

    # === 记录每一步的输出特征，用于 PCA / 误差曲线 ===
    if getattr(self, "_save_feature_traj", False):
        if not hasattr(self, "_feature_traj"):
            self._feature_traj = []
        self._feature_traj.append(
            sample.detach().flatten().cpu()
        )


    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)


def collate_fn(examples):
    """Concat a batch of sampled image in dataloader
    """
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
    }
    return batch

def reset_teacache(unet):
    # 清理 TeaCache 的内部记忆
    unet._tc_cnt = 0
    unet._tc_accum = 0.0
    unet._tc_prev_mod_inp = None
    unet._tc_prev_out = None
    unet._tc_prev_out_residual = None
    # 清空特征轨迹
    if hasattr(unet, "_feature_traj"):
        unet._feature_traj = []
    # 清空 FLOPs 统计次数
    if hasattr(unet, "unet_call_cnt"):
        unet.unet_call_cnt = 0
    if hasattr(unet, "unet_skip_cnt"):
        unet.unet_skip_cnt = 0

def test(
    config: str,
    pretrained_model_path: str,
    dataset_config: Dict,
    logdir: str = None,
    editing_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    batch_size: int = 1,
    model_config: dict={},
    verbose: bool=True,
    **kwargs

):
    args = get_function_args()

    time_string = get_time_string()
    if logdir is None:
        logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))
    logger = get_logger_config_path(logdir)

    if seed is not None:
        set_seed(seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )

    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config
    )

    #cache
    unet.teacache_forward = MethodType(teacache_forward, unet)
    unet.forward = unet.teacache_forward
    unet.enable_teacache =  True
    unet.num_steps = editing_config['num_inference_steps']
    unet.rel_l1_thresh = 0.3
    print(unet.rel_l1_thresh)

    if 'target' not in test_pipeline_config:
        test_pipeline_config['target'] = 'video_diffusion.pipelines.stable_diffusion.SpatioTemporalStableDiffusionPipeline'

    pipeline = instantiate_from_config(
        test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=BDIAScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        ),
        disk_store=kwargs.get('disk_store', False)
    )

    sched = BDIAScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    pipeline.scheduler.gamma = 0.5
    print("===== EXPERIMENT GAMMA =", pipeline.scheduler.gamma, "=====")

    print(type(sched))  # 期望：<class '...BDIAScheduler'>
    print("is BDIAScheduler:", isinstance(sched, BDIAScheduler))  # True
    print("is DDIMScheduler:", isinstance(sched, DDIMScheduler))  # True（继承关系）

    pipeline.scheduler.set_timesteps(editing_config['num_inference_steps'])
    assert len(pipeline.scheduler.timesteps) == unet.num_steps, \
        "TeaCache num_steps 与 scheduler timesteps 不一致"

    pipeline.set_progress_bar_config(disable=True)
    pipeline.print_pipeline(logger)

    print("scheduler type:", type(pipeline.scheduler).__name__)  # 应该是 BDIAScheduler
    pipeline.scheduler.set_timesteps(50)
    t0 = int(pipeline.scheduler.timesteps[0])
    print("first t:", t0, "len(alpha):", len(pipeline.scheduler.alphas_cumprod))

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    prompt_ids = tokenizer(
        dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids)

    train_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    log_train_samples(save_path=train_sample_save_path, train_dataloader=train_dataloader)

    # ================== CPU 上测一次 UNet 的 FLOPs（latent 形状） ==================
    if accelerator.is_main_process:
        # 1. 取一批真实的数据，只用来读出 batch / 帧数 / 分辨率
        batch_for_flops = next(iter(train_dataloader))
        images_flops = batch_for_flops["images"]  # [B, 3, F, H, W]
        B, C_img, F, H, W = images_flops.shape

        # 2. latent 的通道数和尺寸（4 × 64 × 64）
        in_channels = unet.config.in_channels  # 一般 = 4
        down_factor = 8  # Stable Diffusion VAE 下采样 8 倍
        H_lat = H // down_factor
        W_lat = W // down_factor

        # 构造一个“形状正确”的随机 latent： [B*F, 4, 64, 64]
        sample_flops = torch.randn(
            B * F, in_channels, H_lat, W_lat,
            dtype=torch.float32,
        )

        # 3. 文本 embedding：和真实一样，先算 [B, seq, dim]，再复制到每一帧
        with torch.no_grad():
            text_outputs = text_encoder(batch_for_flops["prompt_ids"])
            text_emb = text_outputs[0].to(torch.float32)  # [B, seq_len, dim]

        text_emb = text_emb.repeat_interleave(F, dim=0)  # [B*F, seq_len, dim]

        # 4. timestep：任意 long 值即可，和 FLOPs 无关
        timestep_flops = torch.full(
            (sample_flops.shape[0],),
            fill_value=500,
            dtype=torch.long,
        )

        # 5. 用 CPU 上的 UNet 做 trace（避免 accelerator / GPU / xformers 干扰）
        unet_flops = unet.to("cpu").float()
        unet_flops.eval()

        # 🔧 关键一步：禁用 xformers memory_efficient_attention（否则 CPU 上会报 NotImplementedError）
        for m in unet_flops.modules():
            if hasattr(m, "set_use_memory_efficient_attention_xformers"):
                m.set_use_memory_efficient_attention_xformers(False)
            if hasattr(m, "use_memory_efficient_attention_xformers"):
                m.use_memory_efficient_attention_xformers = False
            # 有些自定义 attention 可能用这个 flag
            if hasattr(m, "use_memory_efficient_attention"):
                m.use_memory_efficient_attention = False

        from fvcore.nn import FlopCountAnalysis
        with teacache_disabled(unet_flops):
            flops_analyser = FlopCountAnalysis(
                unet_flops, (sample_flops, timestep_flops, text_emb)
            )
            single_flops = flops_analyser.total()

        # 把结果挂回原 unet，后面乘以调用次数
        unet.single_flops = single_flops
        logger.info(
            f"[FLOPs] 单次 UNet forward ≈ {single_flops / 1e9:.2f} GFLOPs "
            f"(batch={sample_flops.shape[0]}, frames={F}, H_lat={H_lat}, W_lat={W_lat})"
        )
    # ================== FLOPs 统计结束 ==================

    # 这里再交给 accelerator.wrap

    unet, train_dataloader  = accelerator.prepare(
        unet, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print('use fp16')
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # These models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  # , config=vars(args))
    logger.info("***** wait to fix the logger path *****")

    if editing_config is not None and accelerator.is_main_process:
        validation_sample_logger = P2pSampleLogger(**editing_config, logdir=logdir, source_prompt=dataset_config['prompt'])
        # validation_sample_logger.log_sample_images(
        #     pipeline=pipeline,
        #     device=accelerator.device,
        #     step=0,
        # )
    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    
    batch = next(train_data_yielder)
    if editing_config.get('use_invertion_latents', False):
        # Precompute the latents for this video to align the initial latents in training and test
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        # we only inference for latents, no training
        vae.eval()
        text_encoder.eval()
        unet.eval()

        text_embeddings = pipeline._encode_prompt(
                dataset_config.prompt,
                device = accelerator.device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = True,
                negative_prompt=None
        )
       
        use_inversion_attention =  editing_config.get('use_inversion_attention', False)

        with teacache_disabled(pipeline.unet):
            batch['latents_all_step'] = pipeline.prepare_latents_ddim_inverted(
                rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
                batch_size = 1,
                num_images_per_prompt = 1,  # not sure how to use it
                text_embeddings = text_embeddings,
                prompt = dataset_config.prompt,
                store_attention=use_inversion_attention,
                LOW_RESOURCE = True, # not classifier-free guidance
                save_path = logdir if verbose else None
                )

        batch['ddim_init_latents'] = batch['latents_all_step'][-1]

    else:
        batch['ddim_init_latents'] = None

    vae.eval()
    text_encoder.eval()
    unet.eval()

    # with accelerator.accumulate(unet):
    # Convert images to latent space
    images = batch["images"].to(dtype=weight_dtype)
    images = rearrange(images, "b c f h w -> (b f) c h w")


    if accelerator.is_main_process:

        if validation_sample_logger is not None:
            core_unet = pipeline.unet
            core_unet.eval()
            pipeline.unet.enable_teacache = True
            core_unet._save_feature_traj = True
            reset_teacache(core_unet)  # ← 采样前重置

            validation_sample_logger.log_sample_images(
                image=images, # torch.Size([8, 3, 512, 512])
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents = batch['ddim_init_latents'],
                save_dir = logdir if verbose else None
            )

            if hasattr(core_unet, "_feature_traj") and len(core_unet._feature_traj) > 0:
                # stack 成 [T, D]
                feats_tensor = torch.stack(core_unet._feature_traj, dim=0)  # [T, D]
                features_bdia = feats_tensor.numpy()
                np.save(os.path.join(logdir, "features_ddim.npy"), features_bdia)
                print("saved bdia features to:", os.path.join(logdir, "features_ddim.npy"))
            else:
                print("Warning: no feature trajectory recorded; check _save_feature_traj and forward usage")
        # accelerator.log(logs, step=step)
    print(pipeline.unet.enable_teacache)
    accelerator.end_training()

    if validation_sample_logger is not None:
        core_unet = pipeline.unet
        core_unet.eval()
        pipeline.unet.enable_teacache = True
        reset_teacache(core_unet)  # ← 采样前重置
        validation_sample_logger.log_sample_images(
            image=images,  # torch.Size([8, 3, 512, 512])
            pipeline=pipeline,
            device=accelerator.device,
            step=0,
            latents=batch['ddim_init_latents'],
            save_dir=logdir if verbose else None
        )

        # ====== 采样结束后统计总 FLOPs（新增） ======
        if hasattr(core_unet, "single_flops") and hasattr(core_unet, "unet_call_cnt"):
            total_flops = core_unet.single_flops * core_unet.unet_call_cnt
            logger.info(
                f"[FLOPs] 实际完整 UNet 调用次数: {core_unet.unet_call_cnt}, "
                f"TeaCache 跳过次数: {getattr(core_unet, 'unet_skip_cnt', 0)}, "
                f"单次 UNet FLOPs: {core_unet.single_flops / 1e9:.2f} GFLOPs, "
                f"总 UNet FLOPs ≈ {total_flops / 1e12:.2f} TFLOPs"
            )


@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    Omegadict = OmegaConf.load(config)
    if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
        test(config=config, **Omegadict)
    else:
        # Go through all ckpt if possible
        # Go through all ckpt if possible
        checkpoint_list = sorted(glob(os.path.join(Omegadict['pretrained_model_path'], 'checkpoint_*')))
        print('checkpoint to evaluate:')
        for checkpoint in checkpoint_list:
            epoch = checkpoint.split('_')[-1]

        for checkpoint in tqdm(checkpoint_list):
            epoch = checkpoint.split('_')[-1]
            if 'pretrained_epoch_list' not in Omegadict or int(epoch) in Omegadict['pretrained_epoch_list']:
                print(f'Evaluate {checkpoint}')
                # Update saving dir and ckpt
                Omegadict_checkpoint = copy.deepcopy(Omegadict)
                Omegadict_checkpoint['pretrained_model_path'] = checkpoint

                if 'logdir' not in Omegadict_checkpoint:
                    logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
                    logdir +=  f"/{os.path.basename(checkpoint)}"

                Omegadict_checkpoint['logdir'] = logdir
                print(f'Saving at {logdir}')

                test(config=config, **Omegadict_checkpoint)


if __name__ == "__main__":
    run()
