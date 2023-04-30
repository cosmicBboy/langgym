# 🦜🦾 LangGym

A framework for building Natural Language RL Agents.

| NOTE ℹ️     |
| ----------- |
| This project is currently a proof of concept for a broader framework for using language models as the agent's behavior policy instead of traditional reinforcement learning techniques to power agents. |

## ⚙️ Setup

Create a virtual environment

```bash
python -m venv ~/venvs/langgym
source ~/venvs/langgym/bin/activate
pip install -e .
```

Export your OpenAI API key:

```bash
export OPENAI_API_KEY=...
```

## 👟 Run a Simulation

Run a simulation with 

```bash
python langgym/simple.py --max-cycles 20 --goal-reward -0.05 --output-dir experiments
```

## 🎮 Game Dashboard

Visualize simulations with the streamlit dashboard:

```
streamlit run langgym/dashboard.py
```

# 💡 Ideas for Improvement

These are a few ideas to improve the NLRLA (natural-language generative agent):

- 🧠 **Memory**: Add a text representation of the agent's experiences so far into
  its internal memory.
- 📝 **Summarized Memories**: Store an external memory buffer of observations,
  actions, and rewards, and ask the agent to summarize its experience so far.
  Add this to the agent's internal memory.
- 💭 **Strategic Reflection**: Ask the agent to create high-level strategies, which
  are feed into its own prompt when generating an action.


# 🛣 Roadmap

These are items in the roadmap, in no particular order:

- 👉 Support multi-agent `Environment`s in the Multi Particle Environment suite
  from the `PettingZoo` library (see [here](https://pettingzoo.farama.org/environments/mpe/))
- 👉 Support additional environment suites, like [Atari](https://pettingzoo.farama.org/environments/atari/), [Classic](https://pettingzoo.farama.org/environments/classic/), etc.
- 👉 Create LangGym API for traversing a `Universe` of different `Environment`s. Agents
  can choose which environments they want to play in.
- 👉 Create persistent agents that can store internal memories across their
  experiences across the different `Environment`s.
- 👉 Support use of lighter-weight language models like Alpaca, Pythia, etc.
