import torch
from movement_handling import *
from prey.brain import RecurrentNetwork
from numpy.random import random
from typing import Literal


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Prey:
    def __init__(self, box_size: int, n: int = 50, radius: float = 2.0, speed: float = 0.5,
                 acceleration: float = 0.0125, latent_period: int = 10, network_memory: int = 10,
                 discount: float = 0.95, epsilon_start: float = 0.9, epsilon_scale: int = 25,
                 epsilon_final: float = 0.01, episode_length: int = 500, batch_size: int = 100,
                 buffer_size: int = 25000, buffer_behaviour: Literal["until_full", "discard_old"] = "until_full",
                 buffer_append_prob: float = 0.3, network_params: tuple = (2, 20, 2), learning_rate: float = 0.001,
                 preprocessing: Literal["max_distance", "box_size"] = "max_distance",  reinforce_period: int = 100):

        self.box_size = box_size
        self.n = n
        self.radius = radius
        self.positions = np.random.random((n, 2))
        self.positions *= box_size

        angles = np.random.random(n) * 2 * np.pi
        self.speed = speed
        self.count = 0
        self.latent_period = latent_period
        self.network_memory = network_memory
        self.v = np.zeros((n, 2))
        self.v[:, 0] = np.cos(angles) * self.speed
        self.v[:, 1] = np.sin(angles) * self.speed
        self.acceleration = acceleration
        self.a = np.zeros_like(self.v)

        self.epsilon_start = epsilon_start
        self.epsilon_scale = epsilon_scale
        self.epsilon_final = epsilon_final
        self.episode = 0
        self.episode_length = episode_length
        self.n_features = 2
        self.history = np.zeros((self.n, self.episode_length, self.n_features))
        self.dead_agents = np.zeros((0, 2), dtype='intc')
        self.old_states = torch.zeros((0, self.network_memory, 2), dtype=torch.double)
        self.states = torch.zeros((0, self.network_memory, 2), dtype=torch.double)
        self.rewards = torch.zeros((0, 1), dtype=torch.double)
        self.buffer_size = buffer_size
        self.buffer_behaviour = buffer_behaviour
        self.buffer_append_prob = buffer_append_prob

        self.batch_size = batch_size
        self.network = RecurrentNetwork(*network_params)
        self.discount = discount
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss(reduction="mean").to(device)
        self.total_loss = 0
        self.preprocessing = preprocessing
        self.reinforce_period = reinforce_period

    def reinitialize(self):
        self.positions = np.random.random((self.n, 2))
        self.positions *= self.box_size
        angles = np.random.random(self.n) * 2 * np.pi
        self.v = np.zeros((self.n, 2))
        self.v[:, 0] = np.cos(angles) * self.speed
        self.v[:, 1] = np.sin(angles) * self.speed
        self.a = np.zeros_like(self.v)
        self.count = 0
        self.history = np.zeros((self.n, self.episode_length, 2))
        self.dead_agents = np.zeros((0, 2), dtype='intc')

    def get_epsilon(self):
        return max(self.epsilon_start - self.episode / self.epsilon_scale, self.epsilon_final)

    def exploration(self, indices):
        epsilon = self.get_epsilon()
        n = indices.size
        acc = self.acceleration

        random_n = random(n)
        exploration_indices = indices[random_n < epsilon]
        dead_indices = np.argwhere(np.isin(self.dead_agents[:, 0], exploration_indices))
        exploration_indices = np.delete(exploration_indices, dead_indices)
        n_exploration = exploration_indices.size

        uniform_nums = random(n_exploration)

        for i in range(2):
            if i == 0:
                coord_exploration = exploration_indices[uniform_nums < .5]
            else:
                coord_exploration = exploration_indices[uniform_nums > .5]

            n_coord = coord_exploration.size
            random_coord = random(n_coord)
            coord_neg, coord_pos = coord_exploration[random_coord < .5], coord_exploration[random_coord > .5]
            self.a[coord_neg, i] = -acc
            self.a[coord_pos, i] = acc

        return indices[random_n > epsilon]

    def exploitation(self, exploit_indices, pred_position, pred_v):
        network = self.network
        network.eval()
        dead_indices = np.argwhere(np.isin(self.dead_agents[:, 0], exploit_indices))
        exploit_indices = np.delete(exploit_indices, dead_indices)
        if exploit_indices.size > 0:

            positions = self.positions[exploit_indices]
            n_exploit = exploit_indices.size

            # (batch size, sequence length, n_features)
            first_features = torch.zeros((n_exploit, self.network_memory, 2), dtype=torch.double).to(device)
            scores = torch.zeros((n_exploit, 4), dtype=torch.double).to(device)
            t = self.count
            first_features[:, :self.network_memory-1] = \
                torch.tensor(self.history[exploit_indices, t+1-self.network_memory:t])

            acc = self.acceleration
            operations = ('-', '+')
            inds = (0, 1)
            counter = 0

            # left, right, down, up
            for coord in inds:
                for operation in operations:
                    velocities_coord = np.zeros((exploit_indices.size, 2))

                    if operation == '-':
                        velocities_coord[:, coord] = -acc * self.latent_period ** 2 / 2
                    else:
                        velocities_coord[:, coord] = acc * self.latent_period ** 2 / 2

                    illegal = ((positions[:, coord] + velocities_coord[:, coord])
                               > self.box_size - self.radius).astype('intc') + \
                              ((positions[:, coord] + velocities_coord[:, coord])
                               < self.radius).astype('intc')

                    illegal_inds = np.argwhere(illegal).flatten()
                    legal_indices = np.argwhere(~illegal).flatten()

                    if legal_indices.size > 0:
                        future_positions = positions[legal_indices] + velocities_coord[legal_indices]
                        future_distances = torch.tensor(get_distances_to_predator(
                            pred_position, future_positions)).to(device)
                        future_angles = torch.tensor(get_angle_to_predator(
                            pred_position, positions[legal_indices], pred_v)).to(device)

                        features = first_features.clone()
                        features[legal_indices, -1, 0] = future_distances
                        features[legal_indices, -1, 1] = future_angles
                        network.preprocess_input(features, self.preprocessing, self.box_size)
                        q_values = network(features[legal_indices]).detach()[:, 0]
                        scores[legal_indices, counter] = q_values
                    scores[illegal_inds, counter] = -float('inf')

                    counter += 1

            directions = torch.argmax(scores, dim=1)
            for i, direction in enumerate(directions):
                if direction == 0:
                    self.a[i] = np.array((-acc, 0))
                elif direction == 1:
                    self.a[i] = np.array((acc, 0))
                elif direction == 2:
                    self.a[i] = np.array((0, -acc))
                else:
                    self.a[i] = np.array((0, acc))

    def select_actions(self, pred_position, pred_radius, pred_v):

        if self.count % self.latent_period == 0 and self.count >= self.network_memory-1:
            living_indices = np.delete(np.arange(self.n), self.dead_agents[:, 0])
            living_history = self.history[living_indices, self.count-1, 0]
            collision_inds = living_indices[get_collision_indices(living_history, pred_radius, self.radius)]
            collision_inds = collision_inds.reshape((collision_inds.size, 1))
            collision_inds = np.append(collision_inds, self.count * np.ones_like(collision_inds), axis=1)
            self.dead_agents = np.append(self.dead_agents, collision_inds, axis=0)

            # Exploration
            exploit_indices = self.exploration(living_indices)

            # Exploitation
            self.exploitation(exploit_indices, pred_position, pred_v)

        update_v(self.v, self.a, self.speed)
        move(self.positions, self.v)
        confine_particles(self.positions, self.v, self.box_size, self.box_size, self.radius)
        self.history[:, self.count, 0] = get_distances_to_predator(pred_position, self.positions)
        self.history[:, self.count, 1] = get_angle_to_predator(pred_position, self.positions, pred_v)

        self.count += 1

    def reward(self):
        count = self.count
        episode_length = self.episode_length
        network_memory = self.network_memory
        t = count + 1
        if t == episode_length:
            self.episode += 1
            t_diff = self.latent_period
            dead_indices = self.dead_agents[:, 0]
            for index in range(self.n):
                if index in dead_indices:
                    time_of_death = self.dead_agents[:, 1][dead_indices == index][0]
                    time1 = np.arange(network_memory, time_of_death, t_diff)
                    time2 = np.arange(t_diff + network_memory, time_of_death+t_diff, t_diff)
                    old_states = np.zeros((len(time2), self.network_memory, self.n_features))
                    new_states = np.zeros((len(time2), self.network_memory, self.n_features))
                    rewards = np.zeros((len(time2), 1))

                    i = 0
                    for t_1, t_2 in zip(time1, time2):
                        new_states[i] = self.history[index, t_2 - network_memory:t_2]
                        old_states[i] = self.history[index, t_1 - network_memory:t_1]
                        rewards[i] = -(t_2 / self.episode_length)**2 * (self.episode_length/time_of_death)
                        i += 1
                    self.append_to_buffer(old_states, new_states, rewards)
                else:
                    time1 = np.arange(network_memory, self.episode_length, t_diff)
                    time2 = np.arange(t_diff + network_memory, self.episode_length + t_diff, t_diff)
                    old_states = np.zeros((len(time2), self.network_memory, self.n_features))
                    new_states = np.zeros((len(time2), self.network_memory, self.n_features))
                    rewards = np.zeros((len(time2), 1))

                    i = 0
                    for t_1, t_2 in zip(time1, time2):
                        new_states[i] = self.history[index, t_2 - network_memory:t_2]
                        old_states[i] = self.history[index, t_1 - network_memory:t_1]
                        rewards[i] = (t_2 / self.episode_length)**2
                        i += 1
                    self.append_to_buffer(old_states, new_states, rewards)

            self.reinforce(train_size=self.rewards.shape[0])
            self.reinitialize()
            return True

    def append_tensors(self, old_state, new_state, reward):
        self.old_states = torch.cat((self.old_states, torch.tensor(old_state)))
        self.states = torch.cat((self.states, torch.tensor(new_state)))
        self.rewards = torch.cat((self.rewards, torch.tensor(reward)))

    def pop_tensors(self):
        surplus = self.rewards.shape[0] - self.buffer_size
        self.rewards = self.rewards[surplus:]
        self.old_states = self.old_states[surplus:]
        self.states = self.states[surplus:]

    def append_to_buffer(self, old_state, new_state, reward):
        if self.buffer_behaviour == "until_full":
            if self.rewards.shape[0] < self.buffer_size and np.random.rand() < self.buffer_append_prob:
                # self.replay_buffer.append((torch.tensor(old_state), torch.tensor(new_state), reward))
                self.append_tensors(old_state, new_state, reward)

        elif self.buffer_behaviour == "discard_old":
            if np.random.rand() < self.buffer_append_prob:
                # self.replay_buffer.append((torch.tensor(old_state), torch.tensor(new_state), reward))
                self.append_tensors(old_state, new_state, reward)
            if self.rewards.shape[0] > self.buffer_size:
                self.pop_tensors()

    def reinforce(self, train_size=1000, epochs=1):
        if (self.count + 1) % self.reinforce_period == 0 and self.episode > 0:
            train_count = 0
            batch_size = self.batch_size
            for _ in range(epochs):
                dataset_size = self.rewards.shape[0]
                permutation_indices = torch.randperm(dataset_size)
                n_train = train_count * batch_size
                while n_train < train_size and n_train < dataset_size != 0:
                    indices = permutation_indices[n_train:n_train+batch_size]
                    self.step_gradient(indices)
                    train_count += 1
                    n_train = train_count * batch_size
            if train_count == 0:
                train_count = 1
            total_loss = self.total_loss / train_count
            self.total_loss = 0
            return total_loss
        return None

    def step_gradient(self, indices):
        self.network.train()
        self.optimizer.zero_grad()

        old_states = self.old_states[indices]
        states = self.states[indices]
        rewards = self.rewards[indices]

        old_states = old_states.to(device)
        states = states.to(device)
        self.network.preprocess_input(old_states, self.preprocessing, self.box_size)
        self.network.preprocess_input(states, self.preprocessing, self.box_size)
        rewards = rewards.to(device)

        old_q_values = self.network(old_states)
        q_values = self.network(states)

        y = self.discount * q_values.clone() + rewards

        loss = self.loss_function(y, old_q_values)
        self.total_loss += loss.detach().cpu().item()
        loss.backward()
        self.optimizer.step()