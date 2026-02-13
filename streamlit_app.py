from __future__ import annotations

import runpy
import traceback
from pathlib import Path

import streamlit as st

APP_PATH = Path(__file__).with_name("app.py")

try:
    runpy.run_path(str(APP_PATH), run_name="__main__")
except Exception as error:  # pragma: no cover
    try:
        st.set_page_config(page_title="ICU Task Assistant", layout="wide")
    except Exception:
        # app.py may have already called set_page_config before raising.
        pass
    st.title("Startup Error")
    st.error("The app failed during startup or rerun. See traceback below.")
    st.code(f"{type(error).__name__}: {error}")
    st.code(traceback.format_exc())
