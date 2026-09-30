from diffusers.schedulers import DDIMScheduler
from video_diffusion.pipelines.p2p_ddim_spatial_temporal import P2pDDIMSpatioTemporalPipeline
from video_diffusion.pipelines.ddim_spatial_temporal import DDIMSpatioTemporalStableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from typing import Optional, Union, Tuple, List
import numpy as np
import torch

class BDIASpatioTemporalStableDiffusionPipeline(DDIMSpatioTemporalStableDiffusionPipeline):
    def next_clean2noise_step(self, model_output, timestep: int, sample):
        """
        BDIA inversion (clean->noise) for DDIM (eta=0), mirroring your BDIA sampling formula.
        sample = x_t, timestep = t. Returns x_{t+Δ}^{BDIA}.
        """
        device, dtype = sample.device, sample.dtype

        # ---- 1) 步进索引（统一用均匀步；若用 timesteps 网格也可改成 searchsorted） ----
        ntrain = int(self.scheduler.config.num_train_timesteps)
        ninfer = int(getattr(self.scheduler, "num_inference_steps", ntrain))
        step = max(1, ntrain // max(1, ninfer))
        last_idx = ntrain - 1

        t_curr = int(timestep)
        t_next = min(t_curr + step, last_idx)

        # ---- 2) 取 ᾱ 并对齐 dtype/device ----
        def _ab(idx: int):
            a = self.scheduler.alphas_cumprod[idx] if idx >= 0 else self.scheduler.final_alpha_cumprod
            return a.to(device=device, dtype=dtype)

        alpha_t = _ab(t_curr)
        alpha_nxt = _ab(t_next)
        beta_t = 1.0 - alpha_t
        import pdb; pdb.set_trace()

        # ---- 3) 统一成 epsilon 语义（若 prediction_type = v_prediction）----
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            eps_t = model_output
        elif pred_type == "v_prediction":
            # eps = sqrt(ᾱ_t) * v + sqrt(1-ᾱ_t) * x_t
            eps_t = torch.sqrt(torch.clamp(alpha_t, min=torch.finfo(dtype).tiny)) * model_output \
                    + torch.sqrt(torch.clamp(beta_t, min=torch.finfo(dtype).tiny)) * sample
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        # ---- 4) 反解 x0_hat（必须用 *当前* t 的 ᾱ_t/β̄_t）----
        sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=torch.finfo(dtype).tiny))
        sqrt_beta_t = torch.sqrt(torch.clamp(beta_t, min=torch.finfo(dtype).tiny))
        x0_hat = (sample - sqrt_beta_t * eps_t) / sqrt_alpha_t

        # ---- 5) 纯 DDIM 的 clean->noise：x_{t+Δ}^{DDIM} ----
        sqrt_alpha_nxt = torch.sqrt(torch.clamp(alpha_nxt, min=torch.finfo(dtype).tiny))
        sqrt_beta_nxt = torch.sqrt(torch.clamp(1.0 - alpha_nxt, min=torch.finfo(dtype).tiny))
        x_next_ddim = sqrt_alpha_nxt * x0_hat + sqrt_beta_nxt * eps_t

        # ---- 6) BDIA 校正：+ γ * ( x_last - x_last^{DDIM(from current)} ) ----
        # 其中 x_last^{DDIM(from current)} = sqrt(ᾱ_{t_last}) * x0_hat + sqrt(1-ᾱ_{t_last}) * eps_t
        if not hasattr(self.scheduler, "gamma"):
            self.scheduler.gamma = 0.0  # 默认 γ
            print(1,float(self.scheduler.gamma))
        gamma = float(self.scheduler.gamma)

        if getattr(self.scheduler, "x_last", None) is not None and getattr(self.scheduler, "t_last", None) is not None:
            a_last = _ab(int(self.scheduler.t_last))
            x_last_ddim = torch.sqrt(torch.clamp(a_last, min=torch.finfo(dtype).tiny)) * x0_hat \
                          + torch.sqrt(torch.clamp(1.0 - a_last, min=torch.finfo(dtype).tiny)) * eps_t
            x_next = x_next_ddim + gamma * (self.scheduler.x_last - x_last_ddim)
        else:
            x_next = x_next_ddim

        # ---- 7) 更新状态，供下一步使用（与采样侧保持同一语义）----
        self.scheduler.x_last = sample.detach()  # 保存 x_t
        self.scheduler.t_last = t_curr  # 保存 t
        return x_next

    # def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    #     """
    #     Assume the eta in DDIM=0
    #     """
    #     timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
    #     beta_prod_t = 1 - alpha_prod_t
    #     next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    #     next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    #     next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    #     return next_sample

class P2pBDIASpatioTemporalStableDiffusionPipeline(P2pDDIMSpatioTemporalPipeline):
    def next_clean2noise_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],  # 这里默认是 ε（epsilon）
        timestep: int,                                       # 当前“较干净”时刻 t
        sample: Union[torch.FloatTensor, np.ndarray],        # x_t
    ):
        """
        BDIA clean->noise（DDIM 反演）一步。
        假设 eta=0（DDIM 无随机噪声）。默认 prediction_type='epsilon'。
        """
        # === 计算索引 ===
        step = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        t_curr = timestep                      # 当前时刻 t
        print(f"{t_curr=}")
        # t_prev = max(t_curr - step, -1)        # t-Δ（更干净）
        # print(f"{t_prev=}")
        # t_next = t_curr                        # 为了与原实现命名保持一致，下面仍用 alpha_prev / alpha_next
        t_next = min(t_curr + step, 981)
        print(f"{t_next=}")
        t_last = max(timestep -step ,1)
        print(f"{t_last=}")

        # 1. get previous step value (=t+1)
        # prev_timestep = min(timestep + step, 981)
        # print("prev_timestep:", prev_timestep)
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        print("timestep:", timestep)

        # alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # === 取 ᾱ_t 与 ᾱ_{t_next}（注意：alpha_cumprod 是升噪方向的累计乘积）===
        # alpha_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
        alpha_next = self.scheduler.alphas_cumprod[t_next]
        beta_next = 1.0 - alpha_next
        alpha_curr = self.scheduler.alphas_cumprod[t_curr] if t_curr>= 0 else self.scheduler.final_alpha_cumprod

        # ---- 2) 统一到 epsilon 语义（若 prediction_type = v_prediction 先转 ε）----
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        next_sample_ddim = alpha_next**0.5 * pred_original_sample + (1.0 - alpha_next)**0.5 * model_output

        if not getattr(self.scheduler, "_runtime_inited", False):
            self.scheduler.x_last = None
            self.scheduler.t_last = None
            # γ 可从外部设置；未设置则给个默认值（与你 step() 中一致）
            if not hasattr(self.scheduler, "gamma"):
                self.scheduler.gamma = 0.0
            self.scheduler._runtime_inited = True

        if (self.scheduler.x_last is not None) and (self.scheduler.t_last is not None):
            a_last = (
                self.scheduler.alphas_cumprod[self.scheduler.t_last]
                if self.scheduler.t_last >= 0 else self.scheduler.final_alpha_cumprod
            )
            print("self.scheduler.t_last", self.scheduler.t_last)
            prev_from_curr_ddim = a_last**0.5 * pred_original_sample + (1.0 - a_last)**0.5 * model_output
            next_sample = next_sample_ddim + self.scheduler.gamma * (self.scheduler.x_last - prev_from_curr_ddim)
            print(self.scheduler.gamma)
        else:
            # 第一帧没有历史，退化为纯 DDIM
            next_sample = next_sample_ddim
            print("start")

        self.scheduler.x_last = sample     # 保存当前帧 x_t
        self.scheduler.t_last = t_curr     # 保存当前时刻 t

        return next_sample

