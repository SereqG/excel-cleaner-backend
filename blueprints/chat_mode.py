from flask import Blueprint, request, jsonify

from utils.last_k import last_k
from utils.get_session_history import get_session_history
from helpers.llm_with_rate_limiter import set_llm_model

import uuid
import traceback

import pandas as pd

chat_mode_bp = Blueprint("chat", __name__)

@chat_mode_bp.route("/chat", methods=["POST"])
def chat():
    from app import logger

    try:
        user_prompt = request.form.get("user_prompt")
        file = request.files["file"]
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        preview_rows = min(10, len(df))
        preview_cols = min(12, len(df.columns))
        df_preview = (
            "Data preview (top rows):\n"
            + df.iloc[:preview_rows, :preview_cols].to_csv(index=False)
        )

        if not user_prompt:
            return jsonify({"error": "No user prompt provided"}), 400

        chain_with_history = set_llm_model("ask")

        result = chain_with_history.invoke(
            {"input": user_prompt, "df_preview": df_preview},
            config={"configurable": {"session_id": session_id}},
        )

        # Trim history window
        hist = get_session_history(session_id)
        trimmed = last_k(hist.messages, k=6)
        hist.clear()
        hist.add_messages(trimmed)

        return jsonify({
            "session_id": session_id,
            "response": getattr(result, "content", str(result)),
            "used_preview_rows": preview_rows,
            "used_preview_cols": preview_cols,
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

