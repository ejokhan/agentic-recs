"""
LLM with automatic fallback across Groq models.
Tries best model first, falls back to smaller ones on quota/rate limit errors.
"""
import os
import streamlit as st
from langchain_groq import ChatGroq

MODELS = [
    "llama-3.3-70b-versatile",    # best quality
    "llama-3.1-8b-instant",       # fast fallback
    "gemma2-9b-it",               # second fallback
]


def get_llm_with_fallback():
    """Try each model in order until one works."""
    api_key = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))

    for model in MODELS:
        try:
            llm = ChatGroq(
                model=model,
                api_key=api_key,
                temperature=0.0,
            )
            # Quick test to see if this model is available
            llm.invoke("test")
            return llm, model
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "limit" in error_msg or "quota" in error_msg:
                continue  # try next model
            else:
                continue  # try next model anyway

    return None, None


def get_llm():
    """Get the best available LLM. Returns (llm, model_name) tuple."""
    if "current_llm" not in st.session_state or "current_model" not in st.session_state:
        llm, model = get_llm_with_fallback()
        if llm:
            st.session_state["current_llm"] = llm
            st.session_state["current_model"] = model
        else:
            st.session_state["current_llm"] = None
            st.session_state["current_model"] = None

    return st.session_state["current_llm"], st.session_state["current_model"]


def invoke_llm(prompt):
    """Invoke LLM with automatic fallback. Returns response text or None."""
    api_key = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))

    for model in MODELS:
        try:
            llm = ChatGroq(model=model, api_key=api_key, temperature=0.0)
            response = llm.invoke(prompt)
            return response.content, model
        except Exception:
            continue

    return None, None
