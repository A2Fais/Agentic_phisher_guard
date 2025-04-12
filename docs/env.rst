Environment
===========

Overview
--------
The Environment class is a Gymnasium-based environment specifically designed for phishing URL detection. 
It implements a custom environment that follows the OpenAI Gym interface.

Key Features
-----------
- Built on Gymnasium (successor to OpenAI Gym)
- Specialized for phishing URL detection
- Uses integer features from the dataset
- Binary action space (2 possible actions)
- Tracks environment state and progress

Implementation Details
--------------------
Observation Space
~~~~~~~~~~~~~~~~
The environment leverages ``gym.spaces.Box`` with numpy array bounds for the observation space. 
The implementation:

1. **Feature Selection**: Automatically selects integer columns from the input dataset
2. **Feature Processing**: Works with integer features from the dataset
3. **Numpy Arrays**: Uses numpy arrays for efficient computation
4. **Float32 Dtype**: Uses float32 for state representation

Action Space
~~~~~~~~~~~
Uses ``gym.spaces.Discrete(2)`` to implement binary classification:
- Action 0: Negative classification
- Action 1: Positive classification

API Reference
------------
.. py:class:: Environment(data)

   :param data: Input dataset containing features for phishing detection
   :type data: pandas.DataFrame

   The Environment class initializes with the following steps:
   
   1. Processes the input DataFrame
   2. Selects integer columns
   3. Sets up observation and action spaces
   4. Initializes current state

Methods
~~~~~~~
.. py:method:: get_state()

   Returns the current state values from the dataset
   
   :return: Current state values
   :rtype: numpy.ndarray

.. py:method:: reset()

   Resets the environment to initial state
   
   :return: Initial state
   :rtype: numpy.ndarray

.. py:method:: step(action)

   Executes one step in the environment
   
   :param action: The action to take (0 or 1)
   :type action: int
   :return: Tuple of (next_state, reward, done, info)
   :rtype: tuple

.. py:method:: render(mode="human")

   Renders the current state of the environment
   
   :param mode: Rendering mode (currently only supports "human")
   :type mode: str

Instance Variables
~~~~~~~~~~~~~~~~
.. py:attribute:: data
   
   Processed dataset containing integer features

.. py:attribute:: int_columns
   
   Selected integer columns from the dataset

.. py:attribute:: n_features
   
   Number of features in the processed dataset

.. py:attribute:: observation_space
   
   Gymnasium Box space for observations

.. py:attribute:: action_space
   
   Gymnasium Discrete space for actions

.. py:attribute:: current_state
   
   Tracks the current state in the environment