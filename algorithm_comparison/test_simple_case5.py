import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import pickle
import tensorflow as tf
import keras
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from algorithm_comparison.custom_env import ShowerEnv, MultipathChannel
from gym.envs.registration import register


# How long should training run?
batch_size = 8
num_iterations = 64 * 100
# How many initial random steps, before training start, to
# collect initial data.
initial_collect_steps = 100
# How many steps should we run each iteration to collect
# data from.
collect_steps_per_iteration = 1
# How much data should we store for training examples.
replay_buffer_max_length = 100000


learning_rate = 1e-3
# How often should the program provide an update.
log_interval = 100

# How many episodes should the program use for each evaluation.
num_eval_episodes = 10
# How often should an evaluation occur.
eval_interval = 300

# env_name = 'shower_environment_v1'
# register(
#     id=env_name,
#     entry_point=f'{__name__}:ShowerEnv',
# )

env_name = 'multipath_channel_v1'
register(
    id=env_name,
    entry_point=f'{__name__}:MultipathChannel',
)

# env_name = 'CartPole-v0'

number_of_actions = 2
number_of_iterations = 30

number_of_states = number_of_iterations

train_py_env = suite_gym.load(env_name,
                              gym_kwargs={"number_of_actions": number_of_actions, "number_of_states": number_of_states,
                                          "number_of_iterations": number_of_iterations})
eval_py_env = suite_gym.load(env_name,
                             gym_kwargs={"number_of_actions": number_of_actions, "number_of_states": number_of_states,
                                         "number_of_iterations": number_of_iterations})

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100,50)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    activation_fn=tf.keras.activations.relu,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)


agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
#tf.losses.binary_crossentropy
#    td_errors_loss_fn=common.element_wise_squared_loss, keras.losses.binary_crossentropy, keras.losses.BinaryCrossentropy   tf.losses.binary_crossentropy
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# Testing random policy  (First N steps just using random choose)
print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

ALL_REWARDS = []
ALL_REWARDS_FOR_ONE_GAME = []
ONE_GAME_REWARD = 0.0
SPETS_PER_GAME = 0


def collect_step(environment, policy, buffer, ONE_GAME_REWARD, STEPS_PER_GAME):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)

    ALL_REWARDS.append(next_time_step.reward.numpy()[0])
    if not next_time_step.is_last():
        ONE_GAME_REWARD += next_time_step.reward.numpy()[0]
        STEPS_PER_GAME += 1
    else:
        ALL_REWARDS_FOR_ONE_GAME.append(ONE_GAME_REWARD)
        ONE_GAME_REWARD = 0.0
        STEPS_PER_GAME = 0
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)
    return ONE_GAME_REWARD, STEPS_PER_GAME


def collect_data(env, policy, buffer, ONE_GAME_REWARD, SPETS_PER_GAME, steps=100):
    for _ in range(steps):
        ONE_GAME_REWARD, SPETS_PER_GAME = collect_step(env, policy, buffer, ONE_GAME_REWARD, SPETS_PER_GAME)


collect_data(train_env, random_policy, replay_buffer, ONE_GAME_REWARD, SPETS_PER_GAME, steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph
# using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy,
                                num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and
    # save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        ONE_GAME_REWARD, SPETS_PER_GAME = collect_step(train_env, agent.collect_policy, replay_buffer, ONE_GAME_REWARD,
                                                       SPETS_PER_GAME)

    # Sample a batch of data from the buffer and update
    # the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy,
                                        num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

# from tf_agents.trajectories import time_step as ts
# result = agent.policy.action(ts.restart( observation=np.array([[0.9], [100.0], [0.1], [-100.0]], dtype=np.float32), ))
# print(result)

iterations = range(0, num_iterations + 1, eval_interval)

figures_path = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn/DL_experiments"
scenario = "simple"
pickle.dump(iterations, open(
    f"{figures_path}/{scenario}_iterations_lr{learning_rate}.pickle",
    'wb'))

pickle.dump(returns, open(
    f"{figures_path}/{scenario}_returns_lr{learning_rate}.pickle",
    'wb'))

pickle.dump(ALL_REWARDS, open(
    f"{figures_path}/{scenario}_ALL_REWARDS_lr{learning_rate}.pickle",
    'wb'))

pickle.dump(ALL_REWARDS_FOR_ONE_GAME, open(
    f"{figures_path}/{scenario}_ALL_REWARDS_FOR_ONE_GAME_lr{learning_rate}.pickle",
    'wb'))

plt.figure("eval")
plt.plot(iterations, returns)
plt.title(f"Evaluation, number of episodes = {num_eval_episodes}")
plt.ylabel('Average reward')
plt.xlabel('Iterations')
# plt.ylim(top=250)
plt.grid()

plt.figure("training")
cumulative_average_reward = np.cumsum(ALL_REWARDS) / (np.arange(len(ALL_REWARDS)) + 1)
plt.plot(cumulative_average_reward)
plt.title("Training")
plt.ylabel('Average reward')
plt.xlabel('Iterations')
# plt.ylim(top=250)
plt.grid()

plt.figure("training_for_one_game")
plt.plot(ALL_REWARDS_FOR_ONE_GAME)
plt.title("Training")
plt.ylabel('Average reward')
plt.xlabel('Iterations')
# plt.ylim(top=250)
plt.grid()

plt.show()
