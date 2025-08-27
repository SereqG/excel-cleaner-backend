from flask import Blueprint, jsonify, request

from utils.last_k import last_k
from utils.get_session_history import get_session_history
from helpers.llm_with_rate_limiter import set_llm_model

import uuid
import pandas as pd
import numpy as np
import json
import pickle
import zstandard as zstd

agent_mode_blueprint = Blueprint('agent_mode', __name__)

@agent_mode_blueprint.route('/agent-mode', methods=['POST'])
def agent_mode():
    from app import r
    try:
        user_prompt = request.form.get("user_prompt")
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        raw_df_from_redis = r.get(session_id)
        df = pickle.loads(zstd.ZstdDecompressor().decompress(raw_df_from_redis))

        print(df.head())
        
        chain_with_history = set_llm_model("agent")

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
            set_date_format,
        )

        used_tools = set()
        affected_columns = set()

        if result.tool_calls:
            for tool_call in result.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                # Track tool usage
                used_tools.add(name)

                if name == "rename_columns":
                    result = rename_columns.invoke({
                        "df": df,
                        "column_mapping": args.get("column_mapping"),
                    })
                    df = result["df"]
                    # Add affected columns
                    affected_columns.update(list(args.get("column_mapping", {}).keys()))

                if name == "replace_values":
                    result = replace_values.invoke({
                        "df": df,
                        "value_mapping": args.get("value_mapping"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]
                    # Add affected columns
                    cols = args.get("columns")
                    if cols:
                        affected_columns.update(cols)
                    else:
                        affected_columns.update(list(df.columns))

                if name == "replace_null_values":
                    result = replace_null_values.invoke({
                        "df": df,
                        "replacement_value": args.get("replacement_value"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]
                    cols = args.get("columns")
                    if cols:
                        affected_columns.update(cols)
                    else:
                        affected_columns.update(list(df.columns))

                if name == "change_string_case":
                    result = change_string_case.invoke({
                        "df": df,
                        "case": args.get("case"),
                        "columns": args.get("columns"),
                    })
                    df = result["df"]
                    cols = args.get("columns")
                    if cols:
                        affected_columns.update(cols)
                    else:
                        # Only object columns
                        affected_columns.update(list(df.select_dtypes(include=["object"]).columns))

                if name == "do_nothing":
                    result = do_nothing.invoke({
                        "df": df,
                    })
                    df = result["df"]
                    # No affected columns

                if name == "set_date_format":
                    result = set_date_format.invoke({
                        "df": df,
                        "column": args.get("column"),
                        "date_format": args.get("date_format"),
                    })
                    df = result["df"]
                    affected_columns.add(args.get("column"))

        r.set(session_id, zstd.ZstdCompressor(level=10).compress(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)), ex=3*60*60)

        df.replace(np.nan, None, inplace=True)
        print(df.head())
        return jsonify({
            "session_id": session_id,
            "df_preview": df.head().to_dict(orient="records"),
            "full_df": df.to_dict(orient="records"),
            "used_tools": list(used_tools),
            "affected_columns": list(affected_columns)
        }), 200

    except Exception as e:
        print(f"Error in agent mode: {e}")
        return jsonify({"error": "Internal server error"}), 500
