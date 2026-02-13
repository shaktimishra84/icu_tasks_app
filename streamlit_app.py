from __future__ import annotations

import traceback

import streamlit as st


try:
    from app import *  # noqa: F401,F403
except Exception as error:  # pragma: no cover
    try:
        st.set_page_config(page_title="ICU Task Assistant", layout="wide")
    except Exception:
        # app.py may have already called set_page_config before raising.
        pass
    st.title("Startup Error")
    st.error("The app failed to start. See traceback below.")
    st.code(f"{type(error).__name__}: {error}")
    st.code(traceback.format_exc())
