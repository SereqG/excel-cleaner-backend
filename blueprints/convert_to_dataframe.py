import pickle
import zstandard as zstd

from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np


convert_to_dataframe = Blueprint('convert_to_dataframe', __name__)

@convert_to_dataframe.route('/convert', methods=['POST'])
def convert():
    from app import r
    try:
        file = request.files['file']
        session_id = request.form.get('sessionId')

        df = pd.read_excel(file)
        df.replace({np.nan: pd.NA}, inplace=True)

        payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        payload = zstd.ZstdCompressor(level=10).compress(payload)
        if session_id and payload:
            r.set(session_id, payload)

        df_to_send = pd.read_excel(file, dtype=str)
        df_to_send.replace({np.nan: pd.NA}, inplace=True)

        if df_to_send.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        return jsonify({"data": df_to_send.to_dict(orient="records")}), 200
    except Exception as e:
        print(f"Error in convert: {e}")
        return jsonify({"error": "Internal server error"}), 500
