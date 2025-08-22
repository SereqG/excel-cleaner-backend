from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from variables.agent_mode_prompt import SYSTEM_PROMPT_AGENT_MODE
from variables.ask_mode_prompt import SYSTEM_PROMPT_ASK_MODE
from utils.get_session_history import get_session_history
from helpers.tools import tools

def set_llm_model(mode):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,
        check_every_n_seconds=0.3,
        max_bucket_size=3
    )

    llm_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.3, max_tokens=200, max_retries=3, rate_limiter=rate_limiter)

    llm_model_with_tools = llm_model.bind_tools(tools)

    system_prompt = SYSTEM_PROMPT_AGENT_MODE if mode == "agent" else SYSTEM_PROMPT_ASK_MODE

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("history"),
        ("system", "Context: A compact DataFrame preview may follow.\n{df_preview}"),
        ("human", "{input}"),
    ])

    chain = prompt | llm_model
    chain_with_tools = prompt | llm_model_with_tools

    chain_for_history = chain if mode == "ask" else chain_with_tools

    chain_with_history = RunnableWithMessageHistory(
        chain_for_history,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history