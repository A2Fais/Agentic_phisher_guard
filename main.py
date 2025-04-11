import pandas as pd
from env import Environment

def load_process_data():
    path = "./dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
    data_frame = pd.read_csv(path)
    process_data = Environment(data_frame)
    return process_data

def main():
    processed_data = load_process_data()
    print("Data processing completed successfully")

if __name__ == "__main__":
    main()