import pandas as pd
from env import Environment
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def load_process_data():
    path = "./dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
    data_frame = pd.read_csv(path)
    process_data = Environment(data_frame)
    return process_data

def main():
    env = load_process_data()
    check_env(env, warn=True)
    
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    total_rewards = 0
    
    obs, _ = env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        env.render()
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info: {info}")
        print("\n")
        print("=====================================")
    print(f"Episode finished with total reward: {total_rewards}")
    

if __name__ == "__main__":
    main()