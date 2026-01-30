# ===============================================
# ✅ Full Modular PPO Training Script for CarRacing-v3 (LOCAL UBUNTU)
# ===============================================
import os
import re
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ============================
# 1️⃣ Local Save Directory
# ============================
SAVE_DIR = os.path.expanduser("~/ppo_carracing_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"💾 Checkpoints will be saved to: {SAVE_DIR}")

# ============================
# 2️⃣ Modular Reward Components
# ============================

class RewardComponent:
    def compute(self, obs, env_state) -> float:
        raise NotImplementedError

# ---- On Track Reward ----
class OnTrackRewardSim(RewardComponent):
    def __init__(self, green_thresh=0.22, reward_per_step=0.1):
        self.green_thresh = green_thresh
        self.reward_per_step = reward_per_step

    def compute(self, obs, env_state):
        img = np.asarray(obs, dtype=np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:
            green_mask = (img[..., 1] > img[..., 0] + 12) & (img[..., 1] > img[..., 2] + 12)
            prop_green = np.mean(green_mask)
            if prop_green < self.green_thresh:
                return self.reward_per_step
        return 0.0

# ---- Progress Reward ----
class ProgressRewardSim(RewardComponent):
    def __init__(self, factor=0.1):
        self.factor = factor

    def compute(self, obs, env_state):
        try:
            lv = env_state["unwrapped_env"].car.hull.linearVelocity
            speed = (lv.x**2 + lv.y**2)**0.5
            return speed * self.factor
        except Exception:
            return 0.0

# ---- Offroad Penalty ----
class OffroadPenaltySim(RewardComponent):
    def __init__(self, green_thresh=0.22, penalty=-1.0):
        self.green_thresh = green_thresh
        self.penalty = penalty

    def compute(self, obs, env_state):
        img = np.asarray(obs, dtype=np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:
            green_mask = (img[..., 1] > img[..., 0] + 12) & (img[..., 1] > img[..., 2] + 12)
            prop_green = np.mean(green_mask)
            if prop_green > self.green_thresh:
                return self.penalty * (prop_green / 0.8)
        return 0.0

# ---- Stuck Penalty ----
class StuckPenaltySim(RewardComponent):
    def __init__(self, speed_thresh=0.5, penalty=-5):
        self.speed_thresh = speed_thresh
        self.penalty = penalty

    def compute(self, obs, env_state):
        try:
            lv = env_state["unwrapped_env"].car.hull.linearVelocity
            speed = (lv.x**2 + lv.y**2)**0.5
            if speed < self.speed_thresh:
                return self.penalty
        except Exception:
            pass
        return 0.0

# ---- NEW: Curvature-Aware Speed Reward ----
class CurvatureSpeedRewardSim(RewardComponent):
    def __init__(self,
                 straight_speed_bonus=0.5,
                 turn_speed_penalty=1.0,
                 safe_turn_speed=1.2):
        self.straight_speed_bonus = straight_speed_bonus
        self.turn_speed_penalty = turn_speed_penalty
        self.safe_turn_speed = safe_turn_speed

    def compute(self, obs, env_state):
        try:
            env = env_state["unwrapped_env"]
            car = env.car
            car_pos = np.array(car.hull.position)
            lv = car.hull.linearVelocity
            speed = np.sqrt(lv.x**2 + lv.y**2)

            # Extract track points
            track = np.array(env.track, dtype=object)
            track_centers = np.array([t[2] for t in track])
            dists = np.linalg.norm(track_centers - car_pos, axis=1)
            nearest_idx = np.argmin(dists)

            # Compute local curvature using track angles
            if 2 < nearest_idx < len(track_centers) - 3:
                prev_angle = track[nearest_idx - 1][1]
                next_angle = track[nearest_idx + 1][1]
                curvature = abs(next_angle - prev_angle)
                curvature = (curvature + np.pi) % (2 * np.pi) - np.pi
                curvature = abs(curvature)

                # Straight section → encourage speed
                if curvature < 0.05:
                    return self.straight_speed_bonus * speed
                # Turn → penalize excessive speed
                elif curvature > 0.2:
                    if speed > self.safe_turn_speed:
                        return -self.turn_speed_penalty * (speed - self.safe_turn_speed)
        except Exception:
            pass
        return 0.0


# ============================
# 3️⃣ Modular Reward Wrapper
# ============================
class ModularRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_components):
        super().__init__(env)
        self.reward_components = reward_components

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_state = {"unwrapped_env": self.env.unwrapped}
        shaped = sum(comp.compute(obs, env_state) for comp in self.reward_components)
        total_reward = float(reward) + shaped
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ============================
# 4️⃣ Create Vectorized Env
# ============================
n_envs = 4
reward_components = [
    OnTrackRewardSim(),
    ProgressRewardSim(),
    OffroadPenaltySim(),
    StuckPenaltySim(),
    CurvatureSpeedRewardSim(straight_speed_bonus=0.5,
                            turn_speed_penalty=1.0,
                            safe_turn_speed=1.2)
]

env = make_vec_env(
    "CarRacing-v3",
    n_envs=n_envs,
    wrapper_class=ModularRewardWrapper,
    wrapper_kwargs={"reward_components": reward_components}
)

# ============================
# 5️⃣ Checkpoint Handling
# ============================
def find_latest_checkpoint(save_dir):
    pattern = re.compile(r"checkpoint_(\d+)\.zip$")
    max_idx = -1
    max_path = None
    for fname in os.listdir(save_dir):
        m = pattern.search(fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
                max_path = os.path.join(save_dir, fname)
    return max_idx, max_path

latest_idx, latest_path = find_latest_checkpoint(SAVE_DIR)

if latest_path:
    print(f"🔄 Resuming from checkpoint #{latest_idx}: {latest_path}")
    model = PPO.load(latest_path, env=env, device="auto")
    timesteps_done = latest_idx
else:
    print("✨ No checkpoint found. Starting from scratch.")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        device="auto",
        tensorboard_log=os.path.expanduser("~/ppo_carracing_tensorboard")
    )
    timesteps_done = 0

# ============================
# 6️⃣ Training Loop
# ============================
CHECKPOINT_INTERVAL = 100_000
TOTAL_TIMESTEPS = 5_000_000

remaining_timesteps = TOTAL_TIMESTEPS - (timesteps_done * CHECKPOINT_INTERVAL)
num_chunks = remaining_timesteps // CHECKPOINT_INTERVAL

print(f"🏁 Starting training: {remaining_timesteps} timesteps remaining over {num_chunks} chunks.")

for i in range(timesteps_done, timesteps_done + num_chunks):
    print(f"\n--- Training chunk {i+1} ---")
    start_time = time.time()
    model.learn(total_timesteps=CHECKPOINT_INTERVAL, reset_num_timesteps=False)
    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_{i+1}.zip")
    model.save(ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path} ({(time.time()-start_time)/60:.1f} min)")

print("🎯 Training completed!")
