from flask import Blueprint, request, jsonify
import pandas as pd
import traceback

get_columns_blueprint = Blueprint('get_columns', __name__)

@get_columns_blueprint.route('/get-columns', methods=['POST'])
def get_columns():
    from app import logger
    from utils.is_xlsx_file import is_valid_xlsx_file
    try:
        file = request.files.get("file")

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not is_valid_xlsx_file(file):
            return jsonify({"error": "File must be a valid Excel (.xlsx) file"}), 400
        
        try:
            df = pd.read_excel(file)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return jsonify({"error": "Could not read Excel file. The file may be corrupted."}), 400
        
        if df.empty:
            return jsonify({"error": "The uploaded file is empty"}), 400
        
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_string_dtype(dtype):
                col_type = "text"
            elif pd.api.types.is_numeric_dtype(dtype):
                col_type = "number"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "date"
            else:
                col_type = "other"
            
            columns.append({"name": str(col), "type": col_type})
        
        logger.info(f"Extracted columns: {columns}")
        
        return jsonify({"columns": columns}), 200
    
    except Exception as e:
        logger.error(f"Error in get_columns: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to process file"}), 500