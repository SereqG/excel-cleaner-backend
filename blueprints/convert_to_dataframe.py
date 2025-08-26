from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np

convert_to_dataframe = Blueprint('convert_to_dataframe', __name__)

@convert_to_dataframe.route('/convert', methods=['POST'])
def convert():
    try:
        file = request.files['file']
        df = pd.read_excel(file)
        df.replace(np.nan, None, inplace=True)
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        return jsonify({"data": df.to_dict(orient="records")}), 200
    except Exception as e:
        print(f"Error in convert: {e}")
        return jsonify({"error": "Internal server error"}), 500
