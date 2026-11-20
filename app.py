"""Run with: python -m streamlit run app.py"""

import sqlite3

import pandas as pd
import streamlit as st

from chatbot.llm import OllamaClient, OllamaError
from config import PROJECT_ROOT, load_settings
from database.db import Database
from database.models import Action
from evaluation.metrics import load_interactions, strategy_metrics, summary_metrics
from evaluation.visualizations import average_reward_chart
from feedback.feedback_manager import FeedbackManager
from reinforcement_learning.agent import ContextualBandit
from reinforcement_learning.reward import REWARD_MAP
from reinforcement_learning.state import generate_state


@st.cache_resource
def get_database(path: str) -> Database:
    database = Database(path)
    database.initialize()
    return database


def render_user_message(content: str, message_key: str) -> None:
    with st.chat_message("user"):
        # A native container lets Streamlit measure the bubble's padding correctly.
        with st.container(key=f"user_message_{message_key}", width="content"):
            st.markdown(content)


def render_feedback(message: dict, manager: FeedbackManager, learning_enabled: bool) -> None:
    strategy = Action(message["action"]).name.title()
    with st.container(key=f"response_feedback_{message['id']}", gap="small"):
        if message["feedback_label"] is not None:
            update = "Policy updated." if message["learning_event_id"] is not None else "Saved without a policy update."
            st.caption(f"{strategy} · {message['feedback_label']} ({message['reward']:+d}) · {update}")
            return
        st.caption(f"{strategy} · Was this helpful?")
        with st.container(horizontal=True, gap="small"):
            for label in REWARD_MAP:
                if st.button(label, key=f"feedback_{message['id']}_{label}"):
                    result = manager.submit(message["id"], label, learning_enabled)
                    st.session_state["feedback_notice"] = result.message
                    st.rerun()


@st.dialog("Learning details", width="large")
def show_learning_details(database: Database, agent: ContextualBandit, enabled: bool) -> None:
    """Keep the inspectable policy one click away from an uncluttered chat."""
    interactions = load_interactions(database)
    summary = summary_metrics(interactions)
    average = "not available" if summary["average_reward"] is None else f"{summary['average_reward']:.2f}"
    st.caption(
        f"Run {database.current_run_id()} · {summary['responses']} responses · "
        f"{summary['feedback_submissions']} ratings · Average reward: {average}"
    )
    if not enabled:
        st.caption("Learning is paused. Feedback is saved, but Q-values are frozen and random exploration is off.")
    elif summary["learning_updates"] == 0:
        st.caption("Waiting for valid feedback. No Q-values have been updated in this run.")
    else:
        st.caption(f"{summary['learning_updates']} feedback-driven policy updates. Language-model weights are unchanged.")

    with st.expander("Current policy", expanded=True):
        st.caption(
            f"Learning rate α = {agent.settings.alpha:g}. Exploration starts at ε = {agent.settings.epsilon:g} "
            f"and decays toward {agent.settings.min_epsilon:g} as each context receives feedback. "
            "Unseen state-action pairs have Q = 0. Equal best values are chosen at random."
        )
        st.dataframe(pd.DataFrame(agent.inspect_policy(enabled)), hide_index=True, width="stretch")
    with st.expander("Observed feedback", expanded=True):
        include_earlier = st.checkbox("Include earlier learning runs", key="include_earlier_runs")
        observed = load_interactions(database, current_run_only=False) if include_earlier else interactions
        st.dataframe(strategy_metrics(observed), hide_index=True, width="stretch")
        if observed["feedback_id"].notna().any():
            st.pyplot(average_reward_chart(observed), clear_figure=True)
            st.caption(
                "The last-10 average uses the available ratings until 10 exist. Missing feedback is excluded. "
                "These are descriptive results: changing questions and selective ratings can change the average "
                "without a better policy. Use a repeated evaluation set before drawing conclusions."
            )
        else:
            st.caption("Rate a response to start the reward-over-time chart. No improvement can be assessed yet.")


def main() -> None:
    st.set_page_config(page_title="Mini RL Chatbot", layout="centered", initial_sidebar_state="auto")
    st.html(PROJECT_ROOT / "styles.css")
    with st.container(key="chat_header"):
        st.title("Mini RL Chatbot")
    try:
        settings = load_settings()
        database = get_database(str(settings.db_path))
        client = OllamaClient(settings)
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(f"Could not initialize the application: {exc}")
        st.stop()
    agent = ContextualBandit(database, settings)
    feedback_manager = FeedbackManager(database, agent)

    with st.sidebar:
        with st.container(key="sidebar_brand"):
            st.markdown("### Mini RL")
        conversations = database.list_conversations()
        if st.button("New conversation", key="new_conversation", width="stretch") or not conversations:
            st.session_state["conversation_id"] = database.create_conversation()
            conversations = database.list_conversations()
        labels = {item["id"]: item["title"] or "New conversation" for item in conversations}
        if st.session_state.get("conversation_id") not in labels:
            st.session_state["conversation_id"] = conversations[0]["id"]
        st.caption("Your chats")
        conversation_id = st.radio(
            "Conversation", options=list(labels), format_func=labels.get, key="conversation_id",
            label_visibility="collapsed", width="stretch",
        )
        with st.container(key="sidebar_controls"):
            learning_enabled = st.toggle(
                "Learning enabled", value=True, key="learning_enabled",
                help="Only explicit feedback can update the policy. Paused ratings are saved but never replayed.",
            )
            if st.button("Learning details", key="learning_details", type="tertiary", width="stretch"):
                show_learning_details(database, agent, learning_enabled)
            status = client.check_connection()
            st.caption(f"{settings.model} · {'Connected locally' if status.ready else 'Unavailable'}")
            with st.expander("Settings"):
                if not status.ready:
                    st.caption(status.detail)
                st.button("Check model connection", width="stretch")
                if st.button("Reset learning", width="stretch"):
                    run_id = database.reset_learning()
                    st.session_state["feedback_notice"] = f"Started learning run {run_id}. Your chat history is saved."
                    st.rerun()
                st.caption("Reset starts fresh Q-values and keeps saved chats and ratings.")

    if notice := st.session_state.pop("feedback_notice", None):
        st.caption(notice)
    messages = database.get_messages(conversation_id)
    for message in messages:
        if message["role"] == "user":
            render_user_message(message["content"], str(message["id"]))
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                render_feedback(message, feedback_manager, learning_enabled)

    # A failed request is kept for an explicit retry, not stored as a fake answer.
    pending = st.session_state.get("pending_message")
    if pending and pending["conversation_id"] != conversation_id:
        st.session_state.pop("pending_message", None)
        st.session_state.pop("generation_error", None)
        pending = None
    summary = summary_metrics(load_interactions(database))
    if not learning_enabled:
        learning_status = "Learning paused"
    elif summary["feedback_submissions"]:
        learning_status = f"Learning on · {summary['feedback_submissions']} ratings"
    else:
        learning_status = "Learning on · Waiting for feedback"

    empty_chat = not messages and pending is None
    welcome = st.empty() if empty_chat else None
    if empty_chat:
        with welcome.container():
            with st.container(key="welcome"):
                st.html('<h2 class="welcome-heading">What can I help with?</h2>')
                submitted = st.chat_input(
                    "Message Mini RL", max_chars=settings.max_message_chars, key="chat_prompt",
                )
                with st.container(key="chat_status"):
                    st.caption(f"Runs on your device · {learning_status}")
    else:
        submitted = st.chat_input(
            "Message Mini RL", max_chars=settings.max_message_chars,
            disabled=pending is not None, key="chat_prompt",
        )
    should_generate = False
    if submitted and submitted.strip():
        if welcome is not None:
            welcome.empty()
        pending = {"conversation_id": conversation_id, "text": submitted.strip()}
        st.session_state["pending_message"] = pending
        st.session_state.pop("generation_error", None)
        should_generate = True
    if pending:
        render_user_message(pending["text"], "pending")
        if not should_generate:
            retry, discard = st.columns(2)
            should_generate = retry.button("Retry response")
            if discard.button("Discard unsent message"):
                st.session_state.pop("pending_message", None)
                st.session_state.pop("generation_error", None)
                st.rerun()
        if should_generate:
            state = generate_state(pending["text"])
            selection = agent.select_action(state, learning_enabled)
            try:
                with st.spinner("Generating locally with Ollama..."):
                    response = client.generate(pending["text"], selection.action, messages)
                database.save_interaction(conversation_id, pending["text"], response, selection, settings.model)
            except (OllamaError, ValueError, sqlite3.Error) as exc:
                st.session_state["generation_error"] = str(exc)
                st.rerun()
            st.session_state.pop("pending_message", None)
            st.session_state.pop("generation_error", None)
            st.rerun()
        if error := st.session_state.get("generation_error"):
            st.error(error)
            st.caption("No chatbot response, reward, or learning update was recorded for this failed request.")

    if not empty_chat:
        with st.container(key="chat_status"):
            st.caption(f"Runs on your device · {learning_status}")


if __name__ == "__main__":
    try:
        main()
    except sqlite3.Error:
        st.error("The local database is unavailable. Check its location, free disk space, and file permissions, then reload.")
