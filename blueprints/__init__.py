from blueprints.download_formatted_data import download_formatted_file_blueprint
from blueprints.format_file import format_file_blueprint
from blueprints.get_columns import get_columns_blueprint
from blueprints.convert_to_dataframe import convert_to_dataframe

blueprints = [
    download_formatted_file_blueprint,
    format_file_blueprint,
    get_columns_blueprint,
    convert_to_dataframe
]
