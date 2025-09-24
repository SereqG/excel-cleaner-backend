from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import traceback
import json

format_file_blueprint = Blueprint("format_file", __name__)

@format_file_blueprint.route("/format-file", methods=["POST"])
def format_file():
    from app import logger
    from utils.is_xlsx_file import is_valid_xlsx_file
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files.get("file")
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        formatting_roles = request.form.get("formattingRoles")
        if not formatting_roles:
            return jsonify({"error": "No formatting rules provided"}), 400
        
        if not is_valid_xlsx_file(file):
            return jsonify({"error": "File must be a valid Excel (.xlsx) file"}), 400
        
        try:
            df = pd.read_excel(file)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return jsonify({"error": "Could not read Excel file. The file may be corrupted."}), 400
        
        if df.empty:
            return jsonify({"error": "The uploaded file is empty"}), 400
        
        try:
            roles = json.loads(formatting_roles)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid formatting rules format"}), 400
        
        for role in roles:
            if "columnName" not in role:
                continue

            if role.get("replaceBlankSpacesWith") is not None:
                column_name = role.get("columnName")
                if column_name not in df.columns:
                    logger.warning(f"Column {column_name} not found in dataframe")
                    continue
                
                replace_value = role["replaceBlankSpacesWith"]
                df[column_name] = df[column_name].fillna(replace_value)
                
            column_name = role.get("columnName")
            if column_name not in df.columns:
                logger.warning(f"Column {column_name} not found in dataframe")
                continue
                
            if role.get("roundDecimals") is not None:
                try:
                    decimals = int(role["roundDecimals"])
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df[column_name].dtype):
                        df[column_name] = df[column_name].map(lambda x: f"{x:.{decimals}f}")
                        
                except (ValueError, TypeError):
                    logger.warning(f"Invalid decimal value for column {column_name}")
            
            if pd.api.types.is_string_dtype(df[column_name].dtype) or df[column_name].dtype == 'object':
                if role.get('trimWhitespace'):
                    df[column_name] = df[column_name].astype(str).str.strip()
                
                if role.get("textCase"):
                    if role["textCase"] == "uppercase":
                        df[column_name] = df[column_name].astype(str).str.upper()
                    elif role["textCase"] == "lowercase":
                        df[column_name] = df[column_name].astype(str).str.lower()
                    elif role["textCase"] == "titlecase":
                        df[column_name] = df[column_name].astype(str).str.title()
            
            if role.get("dateFormat"):
                try:
                    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
                    
                    date_format_map = {
                        "DD/MM/YYYY": '%d/%m/%Y',
                        "MM/DD/YYYY": '%m/%d/%Y',
                        "DD-MM-YYYY": '%d-%m-%Y',
                        "YYYY-MM-DD": '%Y-%m-%d'
                    }
                    
                    format_str = date_format_map.get(role["dateFormat"])
                    if format_str:
                        df[column_name] = df[column_name].dt.strftime(format_str)
                except Exception as e:
                    logger.warning(f"Error formatting date for column {column_name}: {e}")

        df.replace(np.nan, None, inplace=True)
        df.replace("nan", None, inplace=True)
        formatted_data = df.where(pd.notnull(df), None).to_dict(orient='records')

        return jsonify({"formattedData": formatted_data}), 200
        
    except Exception as e:
        logger.error(f"Error in format_file: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to process file"}), 500
