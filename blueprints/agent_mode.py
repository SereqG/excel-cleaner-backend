from flask import Blueprint, jsonify, request

from utils.last_k import last_k
from utils.get_session_history import get_session_history
from helpers.llm_with_rate_limiter import set_llm_model

import uuid
import pandas as pd
import numpy as np

agent_mode_blueprint = Blueprint('agent_mode', __name__)

@agent_mode_blueprint.route('/agent-mode', methods=['POST'])
def agent_mode():
    try:
        user_prompt = request.form.get("user_prompt")
        file = request.files["file"]
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        chain_with_history = set_llm_model("agent")

        print(chain_with_history)

        print("\n\n==================\n\n")

        result = chain_with_history.invoke(
            {"input": user_prompt, "df_preview": df},
            config={"configurable": {"session_id": session_id}},
        )

        hist = get_session_history(session_id)
        trimmed = last_k(hist.messages, k=6)
        hist.clear()
        hist.add_messages(trimmed)

        # Import all tool functions
        from helpers.tools import (
            rename_columns,
            replace_values,
            replace_null_values,
            change_string_case,
            do_nothing,
        )

        print("tool calls:", result.tool_calls)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                if name == "rename_columns":
                    result = rename_columns.invoke({
                        "df": df,
                        "column_mapping": args.get("column_mapping"),
                    })
                    df = result["df"]

                if name == "replace_values":
                    result = replace_values.invoke({
                        "df": df,
                        "value_mapping": args.get("value_mapping"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]

                if name == "replace_null_values":
                    result = replace_null_values.invoke({
                        "df": df,
                        "replacement_value": args.get("replacement_value"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]

                if name == "change_string_case":
                    result = change_string_case.invoke({
                        "df": df,
                        "case": args.get("case"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]

                if name == "do_nothing":
                    result = do_nothing.invoke({
                        "df": df,
                    })
                    df = result["df"]

        print("\n\n================\n\n")
        df.replace(np.nan, None, inplace=True)
        print(df.head())
        return jsonify({"session_id": session_id, "df_preview": df.head().to_dict(orient="records")}), 200

    except Exception as e:
        print(f"Error in agent mode: {e}")
        return jsonify({"error": "Internal server error"}), 500
