from pettingzoo.mpe import simple_speaker_listener_v3


env = simple_speaker_listener_v3.env(render_mode="human", continuous_actions=False, max_cycles=100)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    print(f"agent: {agent}; action {action}; reward {reward}; observation: {list(observation)}")
    input("continue...")
    env.step(action)
env.close()
