from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from typing import Optional, Union, Tuple, List
import torch

class BDIAScheduler(DDIMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        if not getattr(self, "_runtime_inited", False):
            # first-time initialization
            self.x_last = None
            # self.x_last2 = None
            self.t_last = None
            if not hasattr(self, "gamma"):
                self.gamma = 0.0

            self._runtime_inited = True
            #print("gamma:", self.gamma)
            # self.gamma1 = 0.9
            # self.gamma2 = 0
            # print("gamma1:", self.gamma1)
            # print("gamma2:", self.gamma2)
            self._runtime_inited = True

        # reset whenever timestep == 981
        if timestep ==981:
            self.x_last = None
            self.t_last = None
            print("gamma:", self.gamma)

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        print("prev_timestep:",prev_timestep)
        print("timestep:",timestep)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        print("timestep:",timestep)

        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.x_last is None:
            print("self.x_last is None")
        else:
            print(timestep)
            a_last = self.alphas_cumprod[timestep + self.config.num_train_timesteps // self.num_inference_steps]
            print("start")
            print(timestep + self.config.num_train_timesteps // self.num_inference_steps)
            print("a_last:",a_last)
            print("end")

        # if self.x_last2 is None:
        #     print("self.x_last2 is None")
        # else:
        #     print(timestep)
        #     a_last2 = self.alphas_cumprod[timestep + 2*self.config.num_train_timesteps // self.num_inference_steps]
        #     print("start")
        #     print(timestep + 2*self.config.num_train_timesteps // self.num_inference_steps)
        #     print("a_last2:",a_last2)
        #     print("end")
                                                                         
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # # additional code added by Guoqiang
        if self.gamma==0:
            prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        #
        if self.x_last is not None:
            # 与调度一致的步长和索引
            step = self.config.num_train_timesteps // self.num_inference_steps
            t_int = int(timestep)
            print(f"{timestep=}")
            # t+Δ（a_last 你上面已算过；若没有，可取消注释下一行）
            a_last = self.alphas_cumprod[min(t_int + step, len(self.alphas_cumprod) - 1)]
            print(t_int + step)
            # # a_last2 = self.alphas_cumprod[min(t_int + 2 * step, len(self.alphas_cumprod) - 1)]
            # # print(t_int + 2* step)
            # # 把 t+Δ / t+2Δ 在时刻 t 的等价表示（仅用 x0_hat 与 eps_hat）
            # x_next_to_t = a_last.sqrt() * pred_original_sample + (1. - a_last).sqrt() * model_output
            # x_next2_to_t = a_last2.sqrt() * pred_original_sample + (1. - a_last2).sqrt() * model_output
            #
            # # gamma1/gamma2（若未显式赋值，则回退到 self.gamma）
            # self.gamma1 = getattr(self, "gamma1", getattr(self, "gamma", 0.95))
            # self.gamma2 = getattr(self, "gamma2", getattr(self, "gamma", 0))
            #
            # # 若还没有两步历史，先用 x_{t+2Δ→t} 近似
            # self.x_last2 = getattr(self, "x_last2", x_next2_to_t)

            print("12345")
            prev_sample = (self.x_last - (1 - self.gamma) * (self.x_last - sample)
                           - self.gamma * (a_last.sqrt() * pred_original_sample
                                           + (1. - a_last).sqrt() * model_output - sample)
                           + alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction - sample
                           )
            print("gamma:", self.gamma)
            # prev_sample = (
            #         self.x_last2
            #         - (1.0 - self.gamma2) * (self.x_last2 - self.x_last)
            #         - self.gamma2 * (x_next2_to_t - x_next_to_t)
            #         - (1.0 - self.gamma1) * (self.x_last - sample)
            #         - self.gamma1 * (x_next_to_t - sample)
            #         + (alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction - sample)
            # )

        else:
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            print("54321")
            prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                if device.type == "mps":
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise

            prev_sample = prev_sample + variance

        print("adding")
        self.x_last = sample
        self.t_last = timestep
        # self.t_last = timestep
        # self.x_last = sample
        # self.x_last2 = self.x_last

        if not return_dict:
             return (
                 prev_sample,
                 pred_original_sample,
             )

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
