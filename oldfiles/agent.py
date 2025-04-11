from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import pandas as pd
from environment import PhishingEnv
import os


def create_and_train_agent(data_path, model_save_path="./models/phishing_model", timesteps=100000):

    data = pd.read_csv(data_path)
    data = data.head(1000)

    env = PhishingEnv(data)
    check_env(env)

    print("Feature columns:", env.feature_cols)
    print("Normalized features (X):", env.X)

    model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    model.learn(total_timesteps=timesteps)

    dir_path = os.path.dirname(model_save_path)
    if dir_path:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")

    return model


def evaluate_agent(model, env, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    data_path = "./dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
    timesteps = 100000

    # Train the agent
    model = create_and_train_agent(data_path, timesteps=timesteps)

    # Evaluate the trained agent
    data = pd.read_csv(data_path).head(1000)
    env = PhishingEnv(data)
    evaluate_agent(model, env)
