import sys; args = sys.argv[1:]
import numpy as np
import matplotlib.pyplot as plt

class GridWorld():

    def __init__(self, filename):

        # Read in the grid
        f = open(filename, 'r')
        lines = f.read().splitlines()
        grid_size, gamma, noise = lines[:3]
        rewards = lines[4:]
        f.close()

        # Process the grid
        self.grid_size = int(grid_size)
        self.gamma = float(gamma)
        self.noise = list(map(float, noise.split(',')))
        if len(self.noise) < 4: self.noise += [0] * (4 - len(self.noise))
        
        self.n_states = self.grid_size ** 2
        self.n_actions = 4

        # Generate rewards function
        self.reward_func = np.array([
            list(map(float, 
                     line.replace('X', '0').split(','))) for line in rewards
            ])
        
        # Generate transition function
        self.transition_func = self.generate_transition_func()

    def pos_to_state(self, x, y):
        return x + y * self.grid_size
    
    def state_to_pos(self, s):
        return s // self.grid_size, s % self.grid_size
    
    def generate_transition_func(self):
        transition_model = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            y, x = self.state_to_pos(s)
            new_states = np.zeros(self.n_actions)

            if self.reward_func[y, x] == 0:
                for a in range(self.n_actions):
                    if a == 0: x_new, y_new = x, y - 1
                    if a == 1: x_new, y_new = x + 1, y
                    if a == 2: x_new, y_new = x, y + 1
                    if a == 3: x_new, y_new = x - 1, y
                    if x_new < 0 or x_new >= self.grid_size or y_new < 0 or y_new >= self.grid_size:
                        x_new, y_new = x, y

                    s_prime = self.pos_to_state(x_new, y_new)
                    new_states[a] = s_prime

            else:
                new_states = np.ones(self.n_actions) * s

            for a in range(self.n_actions):
                transition_model[s, a, int(new_states[a])] = self.noise[0]
                transition_model[s, a, int(new_states[(a + 1) % self.n_actions])] = self.noise[1]
                transition_model[s, a, int(new_states[(a - 1) % self.n_actions])] = self.noise[2]
                transition_model[s, a, int(new_states[(a + 2) % self.n_actions])] = self.noise[3]
        
        return transition_model

    def generate_initial_policy(self):
        return np.zeros(self.n_actions, size=self.n_states)

class ValueIteration():
    
    def __init__(self, reward_func, transition_func, gamma):
        self.grid_size = reward_func.shape[0]
        self.n_states = transition_func.shape[0]
        self.n_actions = transition_func.shape[1]
        self.reward_func = reward_func
        self.transition_func = transition_func
        self.gamma = gamma
        self.values = np.zeros(self.n_states)
        self.policy = None

    def pos_to_state(self, x, y):
        return x + y * self.grid_size
    
    def state_to_pos(self, s):
        return s // self.grid_size, s % self.grid_size

    def iteration(self):
        delta = 0
        for s in range(self.n_states):
            temp = self.values[s]
            v_list = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                p = self.transition_func[s, a]
                v_list[a] = self.reward_func[self.state_to_pos(s)] + self.gamma * np.sum(p * self.values)

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def policy_evaluation(self):
        pi = np.ones(self.n_states) * -1
        for s in range(self.n_states):
            v_list = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                p = self.transition_func[s, a]
                v_list[a] = self.reward_func[self.state_to_pos(s)] + self.gamma * np.sum(p * self.values)

            max_index = []
            max_val = np.max(v_list)
            for a in range(self.n_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            pi[s] = np.random.choice(max_index)
        return pi.astype(int)

    def train(self, tol=1e-5):
        epoch = 0
        delta = self.iteration()
        delta_history = [delta]
        while delta > tol:
            epoch += 1
            delta = self.iteration()
            delta_history.append(delta)
            print('Epoch: {}, delta: {}'.format(epoch, delta))
            if delta < tol:
                break

        self.policy = self.policy_evaluation()


if __name__ == '__main__':

    world = GridWorld(args[0])
    solver = ValueIteration(world.reward_func, 
                            world.transition_func, 
                            world.gamma)
    solver.train(tol=1e-6)

    rewards = world.reward_func.reshape(world.grid_size, world.grid_size)
    arrows = solver.policy.reshape(world.grid_size, world.grid_size)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8,8))

    # Display the grid using imshow
    ax.imshow(rewards, cmap='PiYG')

    # Add arrows to each box
    for i in range(world.grid_size):
        for j in range(world.grid_size):
            direction = arrows[i, j]
            
            if direction == 0: dx, dy = 0, 0.5
            if direction == 1: dx, dy = 0.5, 0
            if direction == 2: dx, dy = 0, -0.5
            if direction == 3: dx, dy = -0.5, 0

            if rewards[i, j] != 0: ax.annotate(rewards[i, j], (j, i), ha='center')
            else: ax.annotate("", (j + dx, i - dy), (j, i),
                        arrowprops=dict(arrowstyle="->", lw=1.5))


    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

