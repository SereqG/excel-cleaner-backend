from flask import Blueprint, request, jsonify
import uuid
import pandas as pd
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.last_k import last_k
from utils.get_session_history import get_session_history

from variables.agent_mode_prompt import SYSTEM_PROMPT_AGENT_MODE

agent_mode_blueprint = Blueprint('agent_mode', __name__)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_AGENT_MODE),
    MessagesPlaceholder("history"),
    ("system", "Context: Excel represented by CSV.\n{df_csv}"),
    ("human", "{input}"),
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # matches {input} above
    history_messages_key="history",
)

@agent_mode_blueprint.route('/agent-mode', methods=['POST'])
def agent_mode():
    from app import logger
    from utils.is_xlsx_file import is_valid_xlsx_file
    
    file = request.files.get('file')
    user_prompt = request.form.get('user_prompt')
    session_id = request.form.get('session_id') or str(uuid.uuid4())

    if not file or not is_valid_xlsx_file(file):
        return jsonify({"error": "Invalid file format. Please upload an .xlsx file."}), 400
    
    if not user_prompt:
        return jsonify({"error": "No user prompt provided"}), 400
    
    try:
        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "The uploaded file is empty"}), 400
        
        data_str = df.to_csv(index=False)
        result = chain_with_history.invoke(
            {"input": user_prompt, "df_csv": data_str},
            config={"configurable": {"session_id": session_id}},
        )

        hist = get_session_history(session_id)
        trimmed = last_k(hist.messages, k=6)
        hist.clear()
        hist.add_messages(trimmed)

        print(result)

        return jsonify({
            "session_id": session_id,
            "response": result.content,  # Fixed: use result.content instead of result.get("preview")
            "preview": ""  # Add preview if needed
        }), 200

    except Exception as e:
        logger.error(f"Error in agent mode: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500
