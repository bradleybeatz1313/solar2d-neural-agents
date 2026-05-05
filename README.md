# 🧬 Neural Agent Simulation — Solar2D

Autonomous agents with neural network brains evolve via neuroevolution to navigate, forage, and survive. A pure-Lua implementation of feedforward neural networks and genetic algorithms running inside Solar2D (formerly Corona SDK).

![Solar2D](https://img.shields.io/badge/Solar2D-2024.3-yellow)
![Lua](https://img.shields.io/badge/Lua-5.1-blue?logo=lua)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Features

### Neural Network Library (`ai/neural_net.lua`)
- **Arbitrary topology** — Any number of hidden layers and sizes
- **Xavier initialization** — Proper weight scaling for stable training
- **Multiple activations** — Sigmoid, tanh, ReLU, leaky ReLU, softmax
- **Serialization** — Flatten/restore weights for save/load and crossover
- **Zero dependencies** — Pure Lua, no C extensions

### Neuroevolution (`ai/evolution.lua`)
- **Tournament selection** — Configurable tournament size for selection pressure
- **Uniform crossover** — Per-weight parent selection for maximum recombination
- **Adaptive mutation** — Rates increase automatically when fitness stagnates
- **Elitism** — Top performers preserved unchanged across generations
- **Population save/load** — Serialize entire populations for checkpointing

### Simulation (`main.lua`)
- **8-dimensional sensor model** — Food direction/distance, poison direction/distance, wall proximity, energy level
- **4 output actions** — Movement X/Y, boost toggle, foraging radius
- **Fitness function** — Multi-objective: food collected, survival time, energy efficiency, movement penalty
- **Speed control** — 1x / 5x / 10x simulation speed for faster evolution
- **Real-time visualization** — Agent color reflects energy level; stats overlay shows generation progress

---

## 📂 Project Structure

```
solar2d-neural-agents/
├── main.lua              # Simulation loop and world management
├── config.lua            # Display configuration
├── build.settings        # Build settings
├── ai/
│   ├── neural_net.lua    # Feedforward NN with genetic operators
│   └── evolution.lua     # GA: selection, crossover, mutation
├── entities/
│   └── agent.lua         # Agent visual + state
└── utils/
```

---

## 🚀 Getting Started

1. Install [Solar2D](https://solar2d.com/download/)
2. Open project folder in Solar2D Simulator
3. Run (Cmd+R / Ctrl+R)
4. Watch agents evolve over generations — click speed button to accelerate

---

## 🔬 AI Research Applications

- **Neuroevolution benchmark** — Compare GA variants (tournament vs roulette, adaptive vs fixed mutation)
- **Sensor model experiments** — Modify `get_agent_inputs()` to test different perception modalities
- **Fitness shaping** — Adjust reward components to study emergent behaviors (foraging vs hoarding vs exploration)
- **Topology search** — Test different `network_topology` configurations for optimal agent intelligence
- **Batch simulation** — Solar2D's lightweight runtime enables running multiple populations in parallel

---

## 📄 License

MIT

---

## Quick Start

1. Open the project in Solar2D (Corona SDK)
2. Press Run -- agents spawn and begin evolving immediately
3. Watch neural fitness improve across generations in the console
4. Tweak hyperparameters in `config.lua` (mutation rate, population size, etc.)

---

## Neuroevolution Algorithm

1. **Initialize**: Random population of N agents with Xavier-initialized weights
2. **Evaluate**: Each agent runs its neural network for one full episode; fitness = survival time * distance traveled
3. **Select**: Elitism (top-k copied) + Tournament selection for parents
4. **Reproduce**: Uniform crossover of parent weights, then Gaussian mutation
5. **Replace**: New generation replaces old; repeat from step 2

---

## Configuration Reference

Edit `config.lua` to tune the experiment:

| Key | Default | Description |
|-----|---------|-------------|
| POPULATION_SIZE | 30 | Agents per generation |
| MUTATION_RATE | 0.12 | Weight mutation probability |
| MUTATION_STRENGTH | 0.25 | Gaussian noise std dev |
| ELITISM_COUNT | 4 | Top agents copied unchanged |
| CROSSOVER_PROB | 0.6 | Crossover vs clone probability |

<!-- last updated 2026-04-28 -->

---

## Algorithm Variants

| Variant | Config | Description |
|---------|--------|-------------|
| Standard GA | default | Elitism + tournament + uniform crossover |
| Rank-based | SELECTION_STRATEGY=rank | Less scale-sensitive than roulette |
| Adaptive mutation | ADAPTIVE_MUTATION=true | Scales noise with population diversity |
| With speciation | SPECIATION_ENABLED=true | NEAT-style species isolation |
