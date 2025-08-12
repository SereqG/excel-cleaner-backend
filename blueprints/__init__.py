from blueprints.chat_mode import chat_mode_bp
from blueprints.download_formatted_data import download_formatted_file_blueprint
from blueprints.format_file import format_file_blueprint
from blueprints.get_columns import get_columns_blueprint
from blueprints.agent_mode import agent_mode_blueprint

blueprints = [
    chat_mode_bp,
    agent_mode_blueprint,
    download_formatted_file_blueprint,
    format_file_blueprint,
    get_columns_blueprint
]
