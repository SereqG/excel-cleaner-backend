from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.last_k import last_k
from utils.get_session_history import get_session_history

from flask import Blueprint, request, jsonify

import os
import getpass
import uuid
import traceback
from variables.ask_mode_prompt import SYSTEM_PROMPT_ASK_MODE
import pandas as pd

chat_mode_bp = Blueprint("chat", __name__)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.3,
    max_bucket_size=3
)

llm_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.3, max_tokens=200, max_retries=3, rate_limiter=rate_limiter)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_ASK_MODE),
    MessagesPlaceholder("history"),
    ("system", "Context: A compact DataFrame preview may follow.\n{df_preview}"),
    ("human", "{input}"),
])

chain = prompt | llm_model

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

