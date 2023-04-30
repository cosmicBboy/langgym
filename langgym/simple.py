import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Union

import faiss
import pettingzoo
import pygame
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.experimental import GenerativeAgentMemory, GenerativeAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from pettingzoo.mpe import simple_v2


GOAL_REWARD = -0.05
ACTION_LIST = ["⏹", "⬅️", "➡️", "⬇️", "⬆️"]
STOP_ACTION = "🛑"
DIAGONAL_ACTION_LIST = {
    "↖️": ["⬅️", "⬆️"],
    "↗️": ["➡️", "⬆️"],
    "↙️": ["⬅️", "⬇️"],
    "↘️": ["➡️", "⬇️"],
}
WORD_ACTION_LIST_1 = ["do nothing", "move left", "move right", "move down", "move up"]
WORD_ACTION_LIST_2 = ["do nothing", "moving left", "moving right", "moving down", "moving up"]
ACTION_SET = frozenset(ACTION_LIST)
INSTRUCTION = (
    "Based on what you know, interpret the meaning of each number in the observation "
    "as it relates to your current velocity and the landmark's relative position, "
    "and relate it to the reward and previous_reward value to inform what to do next: {state}. "
    'Make sure to format the action as json: {{"action": ["<emoji>", "<emoji>", ..., "<emoji>"]}} where <emoji> is one of ⏹, ⬅️, ➡️, ⬇️, or ⬆️. '
    "If you don't have enough information, pick a random action."
)
MEMORY_FORMATION = ()
DISAMBIGUATE_INSTRUCTION = "Your previous response contained multiple possible actions: {actions}. Which one did you want to pick?"
NO_ACTION_INSTRUCTION = (
    'No action was found in your previous response. Based on the observation {state}, pick one of ⏹, ⬅️, ➡️, ⬇️, ⬆️ in the format {{"action": ["<emoji>", "<emoji>", ..., "<emoji>"]}}. '
    "If you don't have enough information, pick a random action."
)
RETRIES = 10

OBSERVATION_FIELDS = ["vel_x_axis", "vel_y_axis", "landmark_rel_x_pos", "landmark_rel_y_pos"]
INITIAL_OBSERVATIONS = [
    (
        "Pog is playing a game environment called 'simple', which is a turn-based 2D environment. "
        "The environment consists of a single agent, Pog, and a single landmark position. The goal of the game is to find the landmark."
    ),
    (
        "At the beginning of each turn, Pog receives an `observation` and `reward` in the format: "
        '{"vel_x_axis": <value>, "vel_y_axis": <value>, "landmark_rel_x_pos": <value>, "landmark_rel_y_pos": <value>, "reward": <reward_value>, "previous_actions": <previous_actions>}. '
        "A positive vel_x_axis means Pog is moving to the right and a negative value means Pog is moving to the left. "
        "A positive vel_y_axis means Pog is moving up and a negative value means Pog is moving down. "
        "A positive landmark_rel_x_pos means that the landmark is to the right of Pog and a negative value means its to the left of Pog. "
        "A positive landmark_rel_y_pos means that the landmark is above Pog and a negative value means its below Pog. "
    ),
    (
        "`reward` represents how close Pog is to the landmark based on its previous action, where higher values means its closer to the landmark. "
        "`previous_actions` corresponds to Pog's last action that's associated with the `reward`. "
        "If `reward` is greater than `previous_reward`, it means that you are doing better and the `actions` from the prior turn probably led to a better outcome."
        'Once Pog reaches a reward of at least {goal}, you can say {{"action": ["{stop_action}"]}} to stop because Pog has won the game!'.format(goal=GOAL_REWARD, stop_action=STOP_ACTION)
    ),
    (
        "Based on the `observation` and the `reward` from the previous turn, the agent needs to pick an action in the form of an emoji: ⏹, ⬅️, ➡️, ⬇️, ⬆️"
        "⏹ means to do nothing, which will cause the velocity to go closer to 0. "
        "⬅️ means to move left, which will cause the velocity_x_axis to decrease. "
        "➡️ means to move right, which will cause the velocity_x_axis to increase. "
        "⬆️ means to move up, which will cause the velocity_y_axis to increase. "
        "⬇️ means to move down, which will cause the velocity_y_axis to decrease. "
        "Pog CANNOT move in a diagonal in this environment, and Pog can only pick one action per turn."
    ),
    (
        "Pog knows that it takes a few turns for their actions to take effect on their own velocity. "
        "If the velocity_x_axis is negative, it'll take a few actions to the right to change their direction to the right. "
        "If the velocity_x_axis is positive, it'll take a few actions to the left to change their direction to the left. "
        "If the velocity_y_axis is negative, it'll take a few actions going up to change their direction upwards. "
        "If the velocity_y_axis is positive, it'll take a few actions going down to change their direction downwards."
    ),
    "When you decide to change your strategy, it might take a few turns to start seeing the effect of your change.",
    "If you're choosing the same actions and the reward is decreasing it means Pog probably needs to pick change their strategy.",
    (
        "Pog needs to output the action that gets it as close to the landmark as possible. "
        'For example, of Pog wants to move left, the output should be {"action": ["⬅️"]}'
        'For example, of Pog wants to move right, the output should be {"action": ["➡️"]}'
        'For example, of Pog wants to move up, the output should be {"action": ["⬆️"]}'
        'For example, of Pog wants to move down, the output should be {"action": ["⬇️"]}'
        'For example, of Pog wants to do nothin, the output should be {"action": ["⏹"]}'
        'Pog can also output a sequence of actions, for example {"action": ["⬅️", "⬆️"]} to go left and up.'
    ),
    (
        "While it's good to slow down when Pog get close the landmark, it's also important to get the reward as close to 0.0 as possible. "
        "If the reward is not changing from one turn to the next try changing my velocity so as to get closer to the landmark."
    ),
]


LLM = ChatOpenAI(max_tokens=150, model_name="gpt-3.5-turbo")
EMBEDDINGS_MODEL = OpenAIEmbeddings()
embedding_size = 1536


def embed_fn(text: str) -> List[float]:
    embedding = EMBEDDINGS_MODEL.embed_query(text)
    return embedding


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    relevance = 1.0 - score / math.sqrt(2)
    relevance = max(relevance, 0.0)
    relevance = min(relevance, 1.0)
    return relevance


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embed_fn, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=0.0, other_score_keys=["importance"], k=5)


def create_agent():
    print("🤖 Creating Pog agent")
    pog_memory = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=False,
        reflection_threshold=8,
    )

    pog = GenerativeAgent(
        name="Pog",
        traits="competitive, enjoys winning games",
        status="wants to learn new games",
        memory_retriever=create_new_memory_retriever(),
        llm=LLM,
        memory=pog_memory,
    )

    return pog


def extract_actions(response: str) -> List[str]:
    """Get the emoji corresponding to the action."""

    match = re.findall(r"\{'action'\: \[.+\]\}", response, flags=re.DOTALL)
    if not match:
        match = re.findall(r'\{"action"\: \[.+\]\}', response, flags=re.DOTALL)

    if len(match) > 1:
        raise RuntimeError("cannot output more than one action json")
    if len(match) == 0:
        return []

    action_json = match[0]
    action_dict = json.loads(action_json.replace("'", '"'))
    return action_dict["action"]


def create_environment(max_cycles: int = 100):
    print("🌍 Creating environment")
    env = simple_v2.env(render_mode="human", continuous_actions=False, max_cycles=max_cycles)
    return env


def run_simulation(
    env: pettingzoo.AECEnv,
    pog: GenerativeAgent,
    output_dir: Union[Path, str] = None,
    initial_observations: List[str] = None,
    nudge: bool = False,
    goal_reward: float = GOAL_REWARD,
):
    output_dir = Path(output_dir or "experiments")
    output_dir.mkdir(exist_ok=True)
    experiment_dir = output_dir / str(int(datetime.now().timestamp()))
    experiment_dir.mkdir()

    initial_observations = initial_observations or INITIAL_OBSERVATIONS
    for observation in initial_observations:
        pog.memory.add_memory(observation)

    prev_reward = None
    prev_raw_actions = None
    env.reset()
    for i, _ in enumerate(env.agent_iter()):
        print(f"👟 Step {i}")
        observation, reward, termination, truncation, info = env.last()

        if i > 0 and reward > goal_reward:
            print(f"🌍 Yay, Pog wins! Goal reward: {goal_reward} - End Reward: {reward}")
            break

        # state reports the reward delta instead of the actual reward. The agent seems to get confused by negative
        # rewards, even if it's higher than the previous one.
        reward_advantage = None
        if reward is not None and prev_reward is not None:
            reward_advantage = reward - prev_reward
        state = {
            **dict(zip(OBSERVATION_FIELDS, [float(x) for x in observation])),
            "reward": reward_advantage or "???",
            "previous_actions": prev_raw_actions or "???",
        }

        prompt = INSTRUCTION.format(state=state)

        if nudge:
            if i < 2:
                # only provide nudges after the second step
                nudge_msg = None
            elif reward > prev_reward:
                nudge_msg = " Pog is getting closer to the landmark."
            elif reward == prev_reward:
                nudge_msg = " Pog is as close to the landmark in this turn compared to the previous turn."
            else:
                nudge_msg = " Pog is going further away from the landmark."

            prompt += nudge_msg

        actions = []
        raw_actions = []

        if termination or truncation:
            print(f"🌍 Simulation complete at step {i}")
            break
        
        for _ in range(RETRIES):
            print(f"👀 PROMPT: {prompt}")
            _, response = pog.generate_dialogue_response(prompt)

            if "same direction as last time" in response:
                raw_actions = prev_raw_actions
                actions = [ACTION_LIST.index(i) for i in raw_actions]
                break

            raw_actions = extract_actions(response)
            print(f"🗣 RESPONSE: {response}")

            if "🛑" in raw_actions:
                print(f"🌍 Pog wins! Landmark found with reward {reward} and goal {GOAL_REWARD}")
                break
            if len(raw_actions) == 0:
                print(f"🤔 No action found {raw_actions}")
                prompt = NO_ACTION_INSTRUCTION.format(state=state, reward=reward)
            else:
                actions = [ACTION_LIST.index(i) for i in raw_actions]
                print(f"👊 ACTIONS: {raw_actions}")
                break

            if actions is None:
                raise ValueError("Action not defined")

        for action in actions:
            env.step(action)

        snapshot = {
            "prompt": prompt,
            "response": response,
            "state": state,
            "raw_actions": raw_actions,
            "actions": actions,
        }

        pygame.image.save(env.unwrapped.screen, experiment_dir / f"frame_{i}.png")
        with (experiment_dir / f"snapshot_{i}.txt").open("w") as f:
            json.dump(snapshot, f)
            
        prev_reward = reward or None
        prev_raw_actions = raw_actions

    env.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--goal-reward", type=float, default=GOAL_REWARD)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    env = create_environment(args.max_cycles)
    pog = create_agent()
    run_simulation(
        env,
        pog,
        output_dir=args.output_dir,
        goal_reward=args.goal_reward,
    )
