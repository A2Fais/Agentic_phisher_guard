Phishing URL Detection Training System
================================

What is this?
------------
This is the main training system that coordinates the training process for the phishing URL detection AI. It handles data loading,
model initialization, training execution, and result visualization.

Key Components
-------------
The system consists of these main components:

1. **Data Loading**:
   - Loads the phishing URL dataset from CSV
   - Prepares the data for training through the environment

2. **Model Setup**:
   - Initializes a Deep Q-Network (DQN) model
   - Configures the training parameters
   - Validates the environment compatibility

3. **Training Process**:
   - Executes the training loop
   - Processes observations and actions
   - Collects rewards and feedback
   - Updates the model

Technical Details
---------------
Implementation Components
~~~~~~~~~~~~~~~~~~~~~~~
- Uses stable-baselines3 for DQN implementation
- Implements environment validation
- Handles model saving and loading
- Provides real-time training feedback

Core Functions
~~~~~~~~~~~~
load_process_data()
    Loads and prepares the dataset for training

main()
    Orchestrates the entire training process:
    - Sets up the environment
    - Initializes the DQN model
    - Runs the training loop
    - Saves the trained model

Training Parameters
----------------
- Model: DQN with MLP Policy
- Training Steps: 1000
- Environment: Phishing URL Detection Environment
- Model Save Location: "./model/dqn_model"

Output and Monitoring
------------------
The system provides:
- Real-time action and reward information
- Environment state visualization
- Training progress updates
- Final reward summaries