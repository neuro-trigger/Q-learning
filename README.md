## Q-learning for Dynamic Cache Memory Management (MDP)

This project models dynamic cache memory management as a Markov Decision Process (MDP) and learns an effective replacement policy using Q-learning. It accompanies the Spanish report `Proyecto Final Estocasticos.pdf` (Universidad Nacional de Colombia, Feb 2024), which provides the theoretical background: Markov processes, MDPs, Reinforcement Learning, Zipf workloads, and Q-learning.

### Key ideas
- **Problem**: Decide which object to keep/evict in a fixed-size cache to improve latency and hit rate under skewed demand.
- **Workload**: Client requests follow a **Zipf** distribution (few hot objects, many cold ones).
- **Model**: Cache state is the set of cached object IDs. An action selects which cached object to evict on a miss. Reward uses object latency; future value is discounted by γ.
- **Learning**: Tabular **Q-learning** with ε-greedy exploration and a decaying learning rate.

### Code structure
- `cacheManagement.py`
  - `ProbGenerator.gen_zipf_prob(elements_list, s=1)`: Generates Zipf probabilities for ordered items.
  - `System(objs_list)`: Holds per-object latency and request probabilities; samples client requests.
  - `Agent(system, discount_factor, cache_capacity)`: Learns a cache replacement policy via Q-learning; tracks metrics.
  - `Test.get_client_request(system, trials)`: Utility to validate sampled frequencies.
  - `main()`: Example setup and training run; plots metrics.
- Report: `Proyecto Final Estocasticos.pdf` (Spanish). Extracted text for quick viewing: `Proyecto_Final_Estocasticos.txt`.

### Metrics (auto-plotted)
- **Accumulated reward**
- **Cache hit rate**
- **Mean latency (ms)**

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install numpy matplotlib
```

### Run
```bash
python cacheManagement.py
```
You can adjust the example in `main()`:
- `states = [(id, latency_ms), ...]`  # ordered from most to least frequent
- `discount_factor = 0.9`
- `cache_capacity = 1`
- `random_seed = 33`

### How it works
- The `System` derives request probabilities with Zipf and returns sampled requests.
- The `Agent` maintains Q(s, a) over states `s` (cached set) and actions `a` (which cached item to evict on a miss).
- On a miss, it explores (ε-greedy) or exploits (argmax Q), applies the eviction, observes latency-based reward, and updates Q with the Bellman target using discount `γ`.
- ε decays gradually; learning rate α = 100 / (100 + trials).

### Acknowledgements
- Student: Carlos Andrés Ríos Rojas (criosro@unal.edu.co)
- Professor: Jorge Eduardo Ortiz Triviño
- Course: Modelos Estocásticos y Simulación en Computación y Comunicaciones, Universidad Nacional de Colombia (Feb 2024)

### Notes
- The full theoretical background, including MDPs, RL, Zipf modeling, and Q-learning updates, is detailed in `Proyecto Final Estocasticos.pdf`.
- The report is in Spanish; this README provides a brief English summary. 