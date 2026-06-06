import torch
import numpy as np

class LerobotDPPOPolicyWrapper:
    """
    Wraps the LeRobot Diffusion Policy to be compatible with DSRL interfaces.
    """
    def __init__(self, base_policy, device="cuda", preprocessor=None, postprocessor=None):
        self.base_policy = base_policy
        self.device = device
        self.base_policy.to(self.device)
        self.base_policy.eval()

        # In modern LeRobot, input normalization (state MIN_MAX, image MEAN_STD) and
        # output un-normalization (action MIN_MAX) live in a SEPARATE processor pipeline,
        # NOT inside DiffusionModel.generate_actions(). Calling generate_actions() on raw
        # observations feeds the network out-of-distribution inputs and returns actions in
        # normalized space. These MUST be applied around generate_actions().
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        if self.preprocessor is None or self.postprocessor is None:
            print("[WARNING] LerobotDPPOPolicyWrapper got no pre/post-processor. "
                  "The diffusion policy will receive UNNORMALIZED inputs and emit "
                  "UNNORMALIZED actions -> base policy is effectively broken.")

        # DSRL requires DDIM. Overriding if necessary.
        if hasattr(self.base_policy, 'config') and getattr(self.base_policy.config, 'noise_scheduler_type', '') != 'DDIM':
            print("[INFO] Overriding LeRobot policy noise scheduler to DDIM for DSRL sampling.")
            self.base_policy.config.noise_scheduler_type = "DDIM"
            try:
                from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
                self.base_policy.diffusion.noise_scheduler = _make_noise_scheduler(
                    "DDIM",
                    num_train_timesteps=self.base_policy.config.num_train_timesteps,
                    beta_start=self.base_policy.config.beta_start,
                    beta_end=self.base_policy.config.beta_end,
                    beta_schedule=self.base_policy.config.beta_schedule,
                    clip_sample=self.base_policy.config.clip_sample,
                    clip_sample_range=self.base_policy.config.clip_sample_range,
                    prediction_type=getattr(self.base_policy.config, 'prediction_type', 'epsilon'),
                )
            except Exception as e:
                print(f"[Warning] Failed to override noise scheduler: {e}")

    def __call__(self, batch, initial_noise, return_numpy=True):
        """
        DSRL expects this signature.
        `batch` is already formatted by LerobotDiffusionPolicyEnvWrapper as expected by LeRobot.
        `initial_noise` is the RL-modified noise tensor (B, horizon, action_dim).
        """
        if not isinstance(batch, dict):
            raise ValueError(
                "DSRL-NA is not supported for image-based policies because it requires denoising actions during the optimization step "
                "using observations from the replay buffer, which only stores low-dimensional states. Please use 'dsrl_sac' instead."
            )

        initial_noise = initial_noise.to(self.device)

        # Build full horizon noise expected by LeRobot's DiffusionModel
        B = initial_noise.shape[0]
        horizon = self.base_policy.config.horizon
        action_dim = self.base_policy.config.output_features["action"].shape[0]
        n_obs_steps = self.base_policy.config.n_obs_steps

        full_noise = torch.randn((B, horizon, action_dim), dtype=initial_noise.dtype, device=initial_noise.device)

        # Overlay DSRL noise onto n_action_steps window
        start = n_obs_steps - 1
        end = start + self.base_policy.config.n_action_steps
        full_noise[:, start:end] = initial_noise

        with torch.no_grad():
            # 1) NORMALIZE observations exactly as during DP training/inference
            #    (state -> MIN_MAX, images -> MEAN_STD, + device transfer). The processor
            #    pipeline broadcasts correctly over the (B, n_obs_steps, ...) batch.
            if self.preprocessor is not None:
                batch = self.preprocessor(dict(batch))
            else:
                batch = dict(batch)

            # 2) If batch has separate image features, stack them into observation.images
            if hasattr(self.base_policy.config, "image_features") and self.base_policy.config.image_features:
                # Stack along dim=-4, which corresponds to the camera dimension.
                # Shape of batch[key] is (B, n_obs_steps, C, H, W)
                # Stacking them results in (B, n_obs_steps, num_cameras, C, H, W)
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in self.base_policy.config.image_features],
                    dim=-4
                )

            # 3) Call generate_actions directly on the underlying DiffusionModel.
            #    Output is in NORMALIZED action space (MIN_MAX, ~[-1, 1]).
            diffused_actions = self.base_policy.diffusion.generate_actions(batch, noise=full_noise)

            # 4) UN-NORMALIZE actions back to the real robot command space (MIN_MAX inverse).
            if self.postprocessor is not None:
                diffused_actions = self.postprocessor.process_action(diffused_actions)

        diffused_actions = diffused_actions.to(self.device)
        if return_numpy:
            diffused_actions = diffused_actions.cpu().numpy()
        return diffused_actions

def load_lerobot_policy(model_path, device="cuda"):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    print(f"Loading LeRobot DP from {model_path}...")
    base_policy = DiffusionPolicy.from_pretrained(model_path)

    # Load the SAME normalization/denormalization pipeline saved with the checkpoint.
    preprocessor, postprocessor = None, None
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=base_policy.config,
            pretrained_path=model_path,
        )
        print("[INFO] Loaded DP pre/post-processor (normalization) pipeline.")
    except Exception as e:
        print(f"[WARNING] Failed to load pre/post-processor: {e}")

    return LerobotDPPOPolicyWrapper(
        base_policy, device=device,
        preprocessor=preprocessor, postprocessor=postprocessor,
    )
