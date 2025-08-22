from flask import Blueprint, jsonify, request

from utils.last_k import last_k
from utils.get_session_history import get_session_history
from helpers.llm_with_rate_limiter import set_llm_model

import uuid
import pandas as pd

agent_mode_blueprint = Blueprint('agent_mode', __name__)

def make_compact_preview(df: pd.DataFrame, sample_rows: int = 20) -> str:
    schema = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
    head = df.head(sample_rows).astype(str).to_dict(orient="records")
    # basic stats without raw PII: counts, nulls, uniques (capped)
    nulls = {c: int(df[c].isna().sum()) for c in df.columns}
    uniques = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    preview = {
        "row_count": int(len(df)),
        "schema": schema,
        "nulls": nulls,
        "uniques": uniques,
        "sample": head  # keep small; redact/omit sensitive columns if needed
    }
    import json
    return json.dumps(preview, ensure_ascii=False)  # pass as compact JSON string

@agent_mode_blueprint.route('/agent-mode', methods=['POST'])
def agent_mode():
    from app import logger
    try:
        user_prompt = request.form.get("user_prompt")
        file = request.files["file"]
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        chain_with_history = set_llm_model("agent")

        result = chain_with_history.invoke(
            {"input": user_prompt, "df_preview": df},
            config={"configurable": {"session_id": session_id}},
        )

        print(f"LLM result: {result}")

        hist = get_session_history(session_id)
        trimmed = last_k(hist.messages, k=6)
        hist.clear()
        hist.add_messages(trimmed)

        # Import all tool functions
        from helpers.tools import (
            rename_column,
            replace_value,
            replace_empty,
            format_date,
            remove_duplicates_from_column,
        )

        if result.tool_calls:
            for tool_call in result.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                if name == "replace_value":
                    df = replace_value.invoke({
                        "df": df,
                        "column": args.get("column"),
                        "old_value": args.get("old_value"),
                        "new_value": args.get("new_value"),
                    })
                elif name == "replace_empty":
                    df = replace_empty.invoke({
                        "df": df,
                        "column": args.get("column"),
                        "new_value": args.get("new_value"),
                    })
                elif name == "rename_column":
                    df = rename_column.invoke({
                        "df": df,
                        "old_name": args.get("old_name"),
                        "new_name": args.get("new_name"),
                    })
                elif name == "format_date":
                    df = format_date.invoke({
                        "df": df,
                        "column": args.get("column"),
                        "current_format": args.get("current_format"),
                        "new_format": args.get("new_format"),
                    })
                elif name == "remove_duplicates_from_column":
                    df = remove_duplicates_from_column.invoke({
                        "df": df,
                        "column": args.get("column"),
                    })
        # Return preview of modified DataFrame
        print(df.head())
        preview = make_compact_preview(df)
        return jsonify({"session_id": session_id, "preview": preview}), 200

    except Exception as e:
        print(f"Error in agent mode: {e}")
        return jsonify({"error": "Internal server error"}), 500
