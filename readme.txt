Run 'RL_for_matching_values' to run the whole thing.

You might have to change the specified path as I used Google Colab to run the files and the path was as specified in My Google Drive.

Training might take some time thus try and exercise patience :)

I used PPO (Proximal Policy Optimization) over DDPG (Deep Deterministic Policy Gradient) as PPO is straightforward to implement, robust to hyperparameters, and easy to get working.
Also, since the environment we are using is cheap to sample from, PPO is most suitable.