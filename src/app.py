"""
Streamlit chat UI for TailorTalk.

Run locally:
    streamlit run src/app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent_executor
import search


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TailorTalk — Saree Similarity Search",
    page_icon="🥻",
)

st.title("🥻 TailorTalk")
st.caption(
    "Chat about sarees, upload a photo (or paste a link), "
    "and I'll find visually similar ones from the catalogue."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@st.cache_resource
def get_agent():
    return build_agent_executor()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image_path" not in st.session_state:
    st.session_state.pending_image_path = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_upload(uploaded_file) -> str:
    """Save uploaded image to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    tmp.write(uploaded_file.getvalue())
    tmp.close()

    return tmp.name


def render_matches(matches: list[dict]):
    """Render catalogue matches returned by the similarity tool."""

    if not matches:
        return

    st.markdown("### Similar sarees")

    cols = st.columns(min(len(matches), 5) or 1)

    for i, match in enumerate(matches):
        with cols[i % len(cols)]:

            image_url = match.get("image_url")

            if image_url:
                st.image(
                    image_url,
                    use_column_width=True,
                )

            name = match.get("name", "Saree")

            st.markdown(f"**{name}**")

            if match.get("score") is not None:
                st.caption(f"Similarity: {match['score']}")

            if match.get("discounted_price"):
                st.caption(
                    f"₹{match['discounted_price']}"
                )

            elif match.get("retail_price"):
                st.caption(
                    f"₹{match['retail_price']}"
                )

            if match.get("product_url"):
                st.markdown(
                    f"[View product]({match['product_url']})"
                )


def extract_matches(result: dict) -> list[dict]:
    """
    Extract structured catalogue matches from the tool's
    intermediate output.
    """

    matches = []

    for step in result.get("intermediate_steps", []):

        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue

        action, observation = step

        tool_name = getattr(
            action,
            "tool",
            "",
        )

        if tool_name != "search_similar_sarees":
            continue

        try:

            if isinstance(observation, str):
                parsed = json.loads(observation)

            elif isinstance(observation, dict):
                parsed = observation

            else:
                continue

            tool_matches = parsed.get(
                "matches",
                [],
            )

            if isinstance(tool_matches, list):
                matches = tool_matches

        except (
            json.JSONDecodeError,
            TypeError,
            AttributeError,
        ):
            continue

    return matches


# ---------------------------------------------------------------------------
# Render previous conversation
# ---------------------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message.get("query_image"):

            st.image(
                message["query_image"],
                width=160,
                caption="Your photo",
            )

        if message.get("matches"):

            render_matches(
                message["matches"]
            )


# ---------------------------------------------------------------------------
# Sidebar — image upload
# ---------------------------------------------------------------------------

with st.sidebar:

    st.subheader("📷 Query image")

    uploaded = st.file_uploader(
        "Upload a saree photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="saree_uploader",
    )

    # -------------------------------------------------------
    # Only process a file when it is actually a NEW upload.
    # -------------------------------------------------------

    if uploaded is not None:

        file_id = (
            f"{uploaded.name}_"
            f"{uploaded.size}_"
            f"{uploaded.type}"
        )

        if (
            st.session_state.uploaded_file_id
            != file_id
        ):

            path = save_upload(uploaded)

            st.session_state.pending_image_path = path

            st.session_state.uploaded_file_id = file_id

    # -------------------------------------------------------
    # Display currently staged image
    # -------------------------------------------------------

    if st.session_state.pending_image_path:

        st.image(
            st.session_state.pending_image_path,
            caption="Ready to search",
            use_column_width=True,
        )

        if st.button(
            "Clear image",
            key="clear_image",
        ):

            st.session_state.pending_image_path = None

            st.session_state.uploaded_file_id = None

            st.rerun()


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_text = st.chat_input(
    "Ask me anything, or say "
    "'find sarees like this' after uploading a photo…"
)


# ---------------------------------------------------------------------------
# Process user message
# ---------------------------------------------------------------------------

if user_text:

    # Capture the staged image BEFORE modifying session state.
    image_path = (
        st.session_state.pending_image_path
    )

    # -------------------------------------------------------
    # Build the input sent to the agent.
    #
    # Only attach an image when the user actually has one
    # staged for THIS turn.
    # -------------------------------------------------------

    agent_input = user_text

    if image_path:

        agent_input += (
            "\n\n"
            f"[Attached image file path: {image_path}]"
        )

    # -------------------------------------------------------
    # Save user message
    # -------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
            "query_image": image_path,
        }
    )

    # -------------------------------------------------------
    # Display user message immediately
    # -------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_text)

        if image_path:

            st.image(
                image_path,
                width=160,
                caption="Your photo",
            )

    # -------------------------------------------------------
    # Build conversation history.
    #
    # The current user message is excluded because it is
    # already passed separately through `input`.
    # -------------------------------------------------------

    history = []

    for message in st.session_state.messages[:-1]:

        if message["role"] == "user":

            history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # -------------------------------------------------------
    # Run agent
    # -------------------------------------------------------

    agent_executor = get_agent()

    with st.chat_message("assistant"):

        with st.spinner("Thinking…"):

            try:

                result = agent_executor.invoke(
                    {
                        "input": agent_input,
                        "chat_history": history,
                    }
                )

                answer = result.get(
                    "output",
                    "Sorry, I couldn't generate a response.",
                )

            except Exception as exc:

                st.error(
                    "Something went wrong while talking "
                    "to the AI."
                )

                st.exception(exc)

                answer = (
                    "I couldn't process that request. "
                    "Please try again."
                )

        # ---------------------------------------------------
        # Display AI response
        # ---------------------------------------------------

        st.markdown(answer)

        # ---------------------------------------------------
        # Extract similarity results
        # ---------------------------------------------------

        matches = extract_matches(result)

        if matches:

            render_matches(matches)

    # -------------------------------------------------------
    # Save assistant response
    # -------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "matches": matches,
        }
    )

    # -------------------------------------------------------
    # IMPORTANT:
    #
    # Clear the staged image AFTER it has been used.
    #
    # This prevents the uploaded image from being silently
    # attached to the user's NEXT question.
    # -------------------------------------------------------

    st.session_state.pending_image_path = None

    st.session_state.uploaded_file_id = None