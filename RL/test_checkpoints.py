import os
import gymnasium as gym
import pygame
from stable_baselines3 import PPO

# checkpoint_path = r"/home/ilyes/Documents/COURS/IA_embarquee/checkpoints/checkpoint_50.zip"
checkpoint_path = r"/home/ilyes/Documents/COURS/IA_embarquee/ppo_carracing_checkpoints/checkpoint_19.zip"

model = PPO.load(checkpoint_path)
print(f"✅ Modèle chargé depuis : {checkpoint_path}")

env = gym.make("CarRacing-v3", render_mode="human")

try:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("\n✅ Simulation terminée.")
    print(f"🎯 Récompense totale de l'épisode : {total_reward:.2f}")
    print("👉 Vous pouvez fermer la fenêtre maintenant.")

finally:
    env.close()
    pygame.display.quit()
    pygame.quit()
    print("🧹 Environnement et rendu pygame fermés proprement.")
    os._exit(0)  # 🔥 Force l'arrêt complet du processus Python


# ============================
# Premiers modèle aux alentours de -50
# Meilleurs modèle entre checkpoint 17 et 21 (600 à 750)
# Derniers modèle aux alentours de 250 
# ============================