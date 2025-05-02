import numpy as np
import torch
import random
import time
import imageio
import os
from collections import deque
from gym import Env, spaces
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')


# Maze Environment
class MazeEnv:
    def __init__(self):
        self.maze, self.start_pos, self.end_pos = self.generate_valid_maze()
        self.agent_pos = list(self.start_pos)
        self.max_steps = 200
        self.current_step = 0

    def generate_maze(self):
        maze = np.ones((6, 6), dtype=int)
        for i in range(6):
            for j in range(6):
                if random.random() < 0.7:
                    maze[i, j] = 0
        maze[0, 0] = maze[5, 5] = 0  # Aseguramos que el inicio y final estén libres
        return maze

    def generate_start_end_pos(self, maze, avoid_pos=None):
        while True:
            pos = (random.randint(0, 5), random.randint(0, 5))
            if maze[pos] == 0 and pos != avoid_pos:
                return pos

    def is_solvable(self):
        queue = deque([self.start_pos])
        visited = set()
        visited.add(self.start_pos)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.end_pos:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 6 and 0 <= ny < 6 and self.maze[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def generate_valid_maze(self):
        while True:
            self.maze = self.generate_maze()
            self.start_pos = self.generate_start_end_pos(self.maze)
            self.end_pos = self.generate_start_end_pos(self.maze, avoid_pos=self.start_pos)
            if self.is_solvable():
                break

        self.save_maze_image()
        return self.maze, self.start_pos, self.end_pos

    def save_maze_image(self):
        output_dir = './output/maze_register/'
        os.makedirs(output_dir, exist_ok=True)

        maze_name = f"maze_{int(time.time())}.png"
        maze_image_path = os.path.join(output_dir, maze_name)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.maze, cmap="gray")
        ax.set_title(f"Generated Maze")
        ax.axis("off")
        plt.savefig(maze_image_path, bbox_inches='tight', pad_inches=0)

        np.save(os.path.join(output_dir, 'maze.npy'), self.maze)
        np.save(os.path.join(output_dir, 'start_pos.npy'), self.start_pos)
        np.save(os.path.join(output_dir, 'end_pos.npy'), self.end_pos)

    def reset(self):
        """Restablece la posición del agente y los pasos."""
        self.agent_pos = list(self.start_pos)
        self.current_step = 0
        return np.array(self.agent_pos, dtype=np.int32)

    def step(self, action):
        self.current_step += 1
        row, col = self.agent_pos

        new_pos = {
            0: [row-1, col],  # Up
            1: [row, col+1],  # Right
            2: [row+1, col],  # Down
            3: [row, col-1]   # Left
        }.get(action, [row, col])

        if (0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6 and self.maze[tuple(new_pos)] != 1):
            self.agent_pos = new_pos

        done = (self.agent_pos == list(self.end_pos)) or (self.current_step >= self.max_steps)
        reward = 50 if done and self.agent_pos == list(self.end_pos) else -0.1
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {}

    def render(self, mode='rgb_array'):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.clear()

        grid = np.full((6, 6, 3), [255, 255, 0], dtype=np.uint8)
        grid[self.maze == 1] = [0, 0, 0]  # Paredes en negro
        grid[self.start_pos] = [0, 200, 0]  # Inicio en verde
        grid[self.end_pos] = [255, 0, 0]  # Fin en rojo
        grid[tuple(self.agent_pos)] = [0, 0, 255]  # Agente en azul
        
        ax.imshow(grid)
        ax.set_xticks([])  
        ax.set_yticks([])  
        fig.tight_layout(pad=0)
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return img[..., :3]

    def print_maze(self):
        """Imprimir el laberinto en el terminal"""
        for row in self.maze:
            print(' '.join('⬛' if cell == 1 else '🟨' for cell in row))


# DQN Agent
class DQNAgent:
    def __init__(self, state_size=2, action_size=4, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.batch_size = 512
        self.tau = 0.01
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
    
        state_normalized = state / 5.0  # Normalize the state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([t[0] for t in minibatch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([t[3] for t in minibatch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).to(self.device)

        targets = self.model(states)
        next_q = self.target_model(next_states)
        
        batch_index = torch.arange(self.batch_size).to(self.device)
        targets[batch_index, actions] = rewards + self.gamma * next_q.max(dim=1)[0] * (1 - dones)

        loss = nn.MSELoss()(targets, self.model(states))
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.soft_update()

    def soft_update(self):
        """Update target model with soft update.""" 
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".weights.pth")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


# Dyna-Q Agent
class DynaQAgent:
    def __init__(self, num_states=36, num_actions=4, device=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_table = torch.zeros((num_states, num_actions), device=self.device)
        self.model = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.planning_steps = 10

    def state_to_index(self, state):
        return state[0] * 6 + state[1]

    def act(self, state):
        state_idx = self.state_to_index(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.num_actions)
        return torch.argmax(self.q_table[state_idx]).item()

    def update(self, state, action, reward, next_state, done):
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)

        best_next = self.q_table[next_state_idx].max() if not done else 0
        td_target = reward + self.gamma * best_next
        self.q_table[state_idx, action] += self.alpha * (td_target - self.q_table[state_idx, action])

        self.model[(state_idx, action)] = (reward, next_state_idx)

        # Planning through model
        if len(self.model) > 0:
            samples = random.choices(list(self.model.items()), k=self.planning_steps)
            for (s, a), (r, ns) in samples:
                best_next = self.q_table[ns].max()
                self.q_table[s, a] += self.alpha * (r + self.gamma * best_next - self.q_table[s, a])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_table.cpu(), path + ".qtable.pth")

    def load(self, path):
        self.q_table = torch.load(path).to(self.device)


# Training and Testing Functions
def train(agent_type, num_episodes, model_path, video_dir, device, env=None):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model_path', model_path)
    video_dir = os.path.join(output_dir, video_dir)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    if env is None:
        env = MazeEnv()  # Se crea únicamente si no se pasa una instancia

    if agent_type == 'dqn':
        agent = DQNAgent(device=device)
    elif agent_type == 'dynaq':
        agent = DynaQAgent(device=device)

    rewards = []
    avg_rewards = []
    epsilons = []
    num_steps = []  # Para llevar el número de pasos
    best_avg_reward = -float('inf')
    
    success_train_count = 0  # Contador de episodios exitosos en entrenamiento

    print(f"Starting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_frames = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            shaped_reward = reward + (1 - np.sqrt((5 - next_state[0])**2 + (5 - next_state[1])**2) / 14.14) * 0.5

            if agent_type == 'dqn':
                agent.remember(state, action, shaped_reward, next_state, done)
                agent.replay()
            elif agent_type == 'dynaq':
                agent.update(state, action, shaped_reward, next_state, done)

            state = next_state
            total_reward += shaped_reward
            steps += 1

            if episode % 10 == 0 and len(episode_frames) < 100:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(np.clip(frame, 0, 255).astype(np.uint8))

        num_steps.append(steps)
        if episode_frames:
            gif_path = os.path.join(video_dir, f'episode_{episode+1:04d}.gif')
            imageio.mimsave(gif_path, episode_frames, fps=5)

        rewards.append(total_reward)
        avg_rewards.append(np.mean(rewards[-100:]) if rewards else 0)
        epsilons.append(agent.epsilon)
        
        # Verificar si el episodio fue exitoso (el agente alcanzó la meta)
        if np.array_equal(state, np.array(env.end_pos)):
            success_train_count += 1

        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Avg Reward (Last 100): {avg_rewards[-1]:.2f} | Steps: {steps} | Epsilon: {agent.epsilon:.3f} | Successes: {success_train_count}")

        # Condición de early stopping: detiene el entrenamiento si se han alcanzado más de 30 éxitos
        if success_train_count > 100:
            print("Early stopping training as agent has reached the goal more than 100 times.")
            break

        if avg_rewards[-1] > best_avg_reward:
            agent.save(f"{model_path}_best")
            best_avg_reward = avg_rewards[-1]

    agent.save(f"{model_path}_best")
    print(f"Training completed! Best average reward: {best_avg_reward:.2f}")

def test(agent_type, model_path, video_dir, device, env=None):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model_path', model_path)
    video_dir = os.path.join(output_dir, video_dir)
    os.makedirs(video_dir, exist_ok=True)

    if env is None:
        env = MazeEnv()

    if agent_type == 'dqn':
        agent = DQNAgent(device=device)
    elif agent_type == 'dynaq':
        agent = DynaQAgent(device=device)

    agent.load(f"{model_path}_best.weights.pth" if agent_type == 'dqn' else f"{model_path}_best.qtable.pth")
    agent.epsilon = 0.0  # Sin exploración
    
    successful_episodes = 0
    total_steps = []  # Para rastrear el número de pasos en cada episodio de test

    for episode in range(10):
        state = env.reset()
        done = False
        frames = []
        steps = 0

        while not done and len(frames) < 100:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            frames.append(env.render())
            steps += 1

        gif_path = os.path.join(video_dir, f'test_ep_{episode+1}.gif')
        imageio.mimsave(gif_path, frames, fps=5)

        if done:
            successful_episodes += 1
        total_steps.append(steps)
        print(f"Test {episode+1}: {'Success' if done else 'Fail'} | Steps: {steps}")

    success_rate = successful_episodes / 10
    avg_steps = np.mean(total_steps)
    print(f"\nSuccess Rate: {success_rate:.0%} | Avg Steps per Episode: {avg_steps:.2f}")
    return success_rate


if __name__ == "__main__":
    num_runs = 50  # Número deseado de ejecuciones totales
    for run in range(num_runs):
        print(f"\n===== Ejecución {run + 1} de {num_runs} =====")
        device = 'cpu'
        maze_solved = False
        trial = 1

        while not maze_solved:
            print(f"\n----- Intento de entrenamiento {trial} -----")
            # Se genera un nuevo maze para este intento
            shared_env = MazeEnv()
            
            # Entrena y testea con DQN
            train('dqn', 3000, 'maze_agent_dqn', 'dqn_training_gifs', device, env=shared_env)
            dqn_success_rate = test('dqn', 'maze_agent_dqn', 'dqn_test_gifs', device, env=shared_env)
            
            # Entrena y testea con Dyna-Q
            train('dynaq', 3000, 'maze_agent_dynaq', 'dynaq_training_gifs', device, env=shared_env)
            dynaq_success_rate = test('dynaq', 'maze_agent_dynaq', 'dynaq_test_gifs', device, env=shared_env)
            
            # Si alguno logra al menos 1 éxito en test, se considera que el maze fue resuelto.
            if dqn_success_rate > 0 or dynaq_success_rate > 0:
                maze_solved = True
                print("¡El maze fue resuelto por al menos un agente!")
            else:
                print("El maze no fue resuelto. Reiniciando entrenamiento con un nuevo maze...\n")
                trial += 1


