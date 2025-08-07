from io import BytesIO
from flask import Flask, request, jsonify
import json
import os
import logging
from flask_cors import CORS
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import traceback
from flask import send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)

ALLOWED_EXTENSIONS = {'xlsx'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 # 2MB limit
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def is_xlsx_file(file):
    """Check if the uploaded file is a valid Excel (.xlsx) file"""
    if not file or not file.filename:
        return False
    
    try:
        # Check file extension
        filename = secure_filename(file.filename.lower())
        if not filename.endswith('.xlsx'):
            logger.warning(f"Invalid file extension: {filename}")
            return False
        
        # Check MIME type
        expected_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        if file.content_type != expected_mime:
            logger.warning(f"Invalid MIME type: {file.content_type}")
            return False
        
        # Check file signature (magic numbers)
        current_pos = file.tell()
        file.seek(0)
        header = file.read(4)
        file.seek(current_pos)  # Reset position
        
        # XLSX files are ZIP archives, so they start with PK
        if header[:2] != b'PK':
            logger.warning("Invalid file signature")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False


@app.route("/get-columns", methods=["POST"])
def get_columns():
    try:
        file = request.files.get("file")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not is_xlsx_file(file):
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
            
            # Determine column type based on data
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


@app.route("/format-file", methods=["POST"])
def format_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files.get("file")
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        formatting_roles = request.form.get("formattingRoles")
        if not formatting_roles:
            return jsonify({"error": "No formatting rules provided"}), 400
        
        if not is_xlsx_file(file):
            return jsonify({"error": "File must be a valid Excel (.xlsx) file"}), 400
        
        try:
            df = pd.read_excel(file)
            df.replace(np.nan, "", inplace=True)
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
            # Validate role has required fields
            if "columnName" not in role:
                continue
                
            column_name = role.get("columnName")
            if column_name not in df.columns:
                logger.warning(f"Column {column_name} not found in dataframe")
                continue
                
            # Apply rounding if specified
            if role.get("roundDecimals") is not None:
                try:
                    decimals = int(role["roundDecimals"])
                    if pd.api.types.is_numeric_dtype(df[column_name].dtype):
                        df[column_name] = df[column_name].round(decimals)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid decimal value for column {column_name}")
            
            # Handle string operations safely
            if pd.api.types.is_string_dtype(df[column_name].dtype) or df[column_name].dtype == 'object':
                # Apply whitespace trimming if specified
                if role.get('trimWhitespace'):
                    df[column_name] = df[column_name].astype(str).str.strip()
                
                # Apply text case transformation if specified
                if role.get("textCase"):
                    if role["textCase"] == "uppercase":
                        df[column_name] = df[column_name].astype(str).str.upper()
                    elif role["textCase"] == "lowercase":
                        df[column_name] = df[column_name].astype(str).str.lower()
                    elif role["textCase"] == "titlecase":
                        df[column_name] = df[column_name].astype(str).str.title()
            
            # Apply date formatting if specified
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

        # Convert to dict for JSON serialization
        formatted_data = df.where(pd.notnull(df), None).to_dict(orient='records')

        return jsonify({"formattedData": formatted_data}), 200
        
    except Exception as e:
        logger.error(f"Error in format_file: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to process file"}), 500
    

@app.route("/download-formatted-file", methods=["POST"])
def download_formatted_file():
    data = request.form.get("formattedData")
    columns = request.form.get("columns")

    print(f"columns: {columns}")  # Debugging line to check columns

    if not data:
        return jsonify({"error": "No formatted data provided"}), 400
    try:
        formatted_data = json.loads(data)
        df = pd.DataFrame(formatted_data)
        new_df = pd.DataFrame(columns=json.loads(columns))
        for col in json.loads(columns):
            new_df[col] = df[col]
        print(new_df.head())  # Debugging line to check the DataFrame structure
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # In production, set debug=False
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)