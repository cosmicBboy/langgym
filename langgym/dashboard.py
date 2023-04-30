"""A dashboard to visualize the games."""

import json
from pathlib import Path

import streamlit as st


GAME_LOG_PREFIX = "snapshot"
GAME_VIEW_PREFIX = "frame"

st.set_page_config(layout="wide")


with st.sidebar:
    st.title("🦜🦾 LangGym")

    exps_dir = Path(st.text_input("Experiment Directory", value="experiments"))
    experiments = exps_dir.glob("*")
    exp_names = sorted([int(e.name) for e in experiments], reverse=True)
    view_exp = st.selectbox(
        "Pick an experiment you want to view.", exp_names, index=0,
    )
    exp_dir = exps_dir / str(view_exp)


# Get experiment logs and views
st.header(f"Experiment `{view_exp}`")
game_log_files = [*exp_dir.glob(f"{GAME_LOG_PREFIX}_*")]
game_log_files.sort(key=lambda x: int(x.stem.replace(f"{GAME_LOG_PREFIX}_", "")))
game_logs = []
for log_file in game_log_files:
    with log_file.open() as f:
        game_logs.append(json.load(f))

view_paths = [*exp_dir.glob(f"{GAME_VIEW_PREFIX}_*")]
view_paths.sort(key=lambda x: int(x.stem.replace(f"{GAME_VIEW_PREFIX}_", "")))

assert len(game_logs) == len(view_paths)
n_steps = len(game_logs)

# Render logs and views
st.write("The Agent is grey, the Goal is in red")
step_num = st.slider("Playback", min_value=0, max_value=n_steps - 1)

col1, col2 = st.columns(2, gap="medium")

i = step_num
with col1:
    st.subheader("View")
    st.write(f"`{view_paths[i]}`")
    st.image(str(view_paths[i]))
            
with col2:
    st.subheader("Log")
    st.write(f"`{game_log_files[i]}`")
    st.write(game_logs[i])
