from flask import Blueprint, request, jsonify
import json
import pandas as pd
from io import BytesIO
from flask import send_file
import traceback

download_formatted_file_blueprint = Blueprint('download_formatted_file', __name__)

@download_formatted_file_blueprint.route("/download-formatted-file", methods=["POST"])
def download_formatted_file():
    from app import logger
    data = request.form.get("formattedData")
    columns = request.form.get("columns")


    if not data:
        return jsonify({"error": "No formatted data provided"}), 400
    try:
        formatted_data = json.loads(data)
        df = pd.DataFrame(formatted_data)
        new_df = pd.DataFrame(columns=json.loads(columns))
        for col in json.loads(columns):
            new_df[col] = df[col]
        print(new_df.head())
        output = BytesIO()
        new_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="formatted_file.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )  
    except Exception as e:
        logger.error(f"Error in download_formatted_file: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to generate formatted file"}), 500