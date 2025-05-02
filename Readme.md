# Maze Labyrinth 6x6

Reinforcement learning project comparing **DQN** and **Dyna-Q** agents in a procedurally generated 6x6 maze environment.

Each training session begins with a **randomly generated maze**, guaranteed to be solvable. Agents are trained for **3000 episodes** each. If **neither agent succeeds**, a new maze is generated and training restarts. If **at least one agent solves the maze 30+ times**, the training concludes and results are recorded.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ushio2580/Dynamic-6x6-Maze-Solver-with-DQN-and-Dyna-Q-Agents.git
   cd Maze_Laberinto_6x6
   ```

2. **Install dependencies:**
   ```bash
   pip install gym torch imageio matplotlib numpy
   ```

---

## 🚀 Usage

1. Run the script or open `Maze_No_Rand.ipynb` in Colab or Jupyter.
2. Training and test GIFs will be saved in these folders:
   - `output/dqn_training_gifs/`
   - `output/dqn_test_gifs/`
   - `output/dynaq_training_gifs/`
   - `output/dynaq_test_gifs/`

---

## ⚙️ Code Overview

### 🧱 Environment: `MazeEnv`

A customizable 6x6 grid environment:
- Randomly generates walls with a fixed probability.
- Guarantees a valid path from random start to goal.
- Includes visualization (`render`), printing, and `step()` logic for moving the agent.
- Stores each generated maze image and configuration for reproducibility.

### 🧠 Agents

- **DQNAgent**: Uses a deep neural network for Q-learning with experience replay and soft updates.
- **DynaQAgent**: Combines Q-learning with a simple internal model for planning simulated updates.

Both agents support:
- Action selection (`act`)
- Training (`replay` or `update`)
- Saving/loading learned models

### 🏃 Training and Testing Functions

- `train(agent_type, ...)`: Runs up to 3000 episodes, logs reward and saves GIFs every 10 episodes. If an agent solves the maze over 30 times, it stops early.
- `test(agent_type, ...)`: Evaluates the trained agent over 10 episodes with zero exploration (greedy policy).
- If both agents fail to solve the maze, the code regenerates a new random maze and retrains both models again until success.

---

## 📊 Results

### 📡 **DQN - Test Episode**

![DQN Test Episode](https://raw.githubusercontent.com/ushio2580/Dynamic-6x6-Maze-Solver-with-DQN-and-Dyna-Q-Agents/f52be82ec287b3dfd5066b7e417e5a4200d41c31/assets/dqn/test_ep_1.gif)

- **DQN**: Success rate: **100%**, Average steps: **10.00**

---

### 🤖 **Dyna-Q - Test Episode**

![Dyna-Q Test Episode](https://raw.githubusercontent.com/ushio2580/Dynamic-6x6-Maze-Solver-with-DQN-and-Dyna-Q-Agents/main/assets/dynaq/test_ep_1.gif)

- **Dyna-Q**: Success rate: **100%**, Average steps: **10.00**

---

## 📝 License

This project is licensed under the terms of the **MIT License**. See the `LICENSE` file for more information.

---

## 🎯 Getting Started

1. **Run training:**
   ```bash
   python Laberinto_Dinamico.py
   ```
   Or step through `Dyna_Maze.ipynb` to visualize results.

2. **Check outputs:**
   - Models saved in `output/model_path/`
   - GIFs in respective output folders

3. **Tune & Retrain:**  
   Adjust hyperparameters or maze settings to explore agent behavior.

---

## 📚 References

- [DQN (Deep Q-Network)](https://www.nature.com/articles/nature14236)
- [Dyna-Q Algorithm](https://www.cs.cmu.edu/~./mmv/papers/95-1.pdf)
