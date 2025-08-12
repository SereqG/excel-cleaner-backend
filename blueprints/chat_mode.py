from flask import Blueprint, request, jsonify

import pandas as pd
import uuid
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from variables.ask_mode_prompt import SYSTEM_PROMPT_ASK_MODE
from utils.last_k import last_k
from utils.get_session_history import get_session_history


chat_mode_bp = Blueprint("chat", __name__)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_ASK_MODE),
    MessagesPlaceholder("history"),
    ("system", "Context: A compact DataFrame preview may follow.\n{df_preview}"),
    ("human", "{input}"),
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # matches {input} above
    history_messages_key="history",
)

@chat_mode_bp.route("/chat", methods=["POST"])
def chat():
    from utils.is_meta_request import is_meta_request
    from utils.is_xlsx_file import is_valid_xlsx_file
    from app import logger
    from utils.get_session_history import get_session_history
    try:
        user_prompt = request.form.get("user_prompt")
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        if not user_prompt:
            return jsonify({"error": "No user prompt provided"}), 400

        meta = is_meta_request(user_prompt)

        file = request.files.get("file")

        df_preview = "No file provided."
        preview_rows = 0
        preview_cols = 0

        if not meta:
            # Only enforce file for non-meta requests
            if not file or not is_valid_xlsx_file(file):
                return jsonify({"error": "Invalid or no file provided"}), 400

            try:
                df = pd.read_excel(file)
                if df.empty:
                    return jsonify({"error": "The uploaded file is empty"}), 400
            except Exception as e:
                logger.error(f"Error reading Excel file: {e}")
                return jsonify({"error": "Could not read Excel file"}), 400

            preview_rows = min(10, len(df))
            preview_cols = min(12, len(df.columns))
            df_preview = (
                "Data preview (top rows):\n"
                + df.iloc[:preview_rows, :preview_cols].to_csv(index=False)
            )

        # Invoke
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

