import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False, episodes=1000):

    env = gym.make('Pendulum-v1', render_mode='human' if render else None)

    # Hyperparameters
    learning_rate_a = 0.1        # Alpha (learning rate)
    discount_factor_g = 0.9      # Gamma (discount factor)
    epsilon = 1                  # Initial epsilon (100% random actions)
    epsilon_decay_rate = 0.0005  # Epsilon decay rate
    epsilon_min = 0.05           # Minimum epsilon
    divisions = 15               # Discretization of state and action space

    # Divide observation space into discrete segments
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divisions)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divisions)
    w = np.linspace(env.observation_space.low[2], env.observation_space.high[2], divisions)

    # Divide action space into discrete segments
    a = np.linspace(env.action_space.low[0], env.action_space.high[0], divisions)

    # Initialize Q-table
    if is_training:
        q = np.zeros((len(x) + 1, len(y) + 1, len(w) + 1, len(a) + 1))
    else:
        with open('pendulum.pkl', 'rb') as f:
            q = pickle.load(f)

    best_reward = -float('inf')
    rewards_per_episode = []

    for episode in range(episodes):

        state = env.reset()[0]  # Reset environment
        s_i0 = np.digitize(state[0], x)
        s_i1 = np.digitize(state[1], y)
        s_i2 = np.digitize(state[2], w)

        rewards = 0
        steps = 0

        while steps < 1000 or not is_training:
            if is_training and np.random.rand() < epsilon:
                # Choose a random action
                action = env.action_space.sample()
                action_idx = np.digitize(action, a)
            else:
                # Choose the best action from Q-table
                action_idx = np.argmax(q[s_i0, s_i1, s_i2, :])
                action = a[action_idx - 1]

            # Take action and observe new state
            new_state, reward, _, _, _ = env.step([action])

            # Discretize new state
            ns_i0 = np.digitize(new_state[0], x)
            ns_i1 = np.digitize(new_state[1], y)
            ns_i2 = np.digitize(new_state[2], w)

            # Update Q-table
            if is_training:
                q[s_i0, s_i1, s_i2, action_idx] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, :])
                    - q[s_i0, s_i1, s_i2, action_idx]
                )

            state = new_state
            s_i0, s_i1, s_i2 = ns_i0, ns_i1, ns_i2

            rewards += reward
            steps += 1

        # Save best reward Q-table
        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                with open('pendulum.pkl', 'wb') as f:
                    pickle.dump(q, f)

        # Store rewards per episode
        rewards_per_episode.append(rewards)

        # Print stats
        if is_training and episode % 100 == 0:
            mean_reward = np.mean(rewards_per_episode[-100:])
            print(f'Episode: {episode}, Epsilon: {epsilon:.2f}, Best Reward: {best_reward}, Mean Rewards: {mean_reward:.1f}')

            # Plot mean rewards
            mean_rewards = [np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(episode)]
            plt.plot(mean_rewards)
            plt.savefig('pendulum.png')

        elif not is_training:
            print(f'Episode: {episode}, Reward: {rewards:.1f}')

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

if __name__ == '__main__':
    # Train for 5000 episodes
    # run(is_training=True, render=False, episodes=5000)

    # Test for 10 episodes
    run(is_training=False, render=True, episodes=10)
