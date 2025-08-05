from flask import Flask, request
import json
import os
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})


def is_xlsx_file(file):
    """Check if the uploaded file is a valid Excel (.xlsx) file"""
    if not file or not file.filename:
        return False
    
    # Check file extension
    filename = file.filename.lower()
    if not filename.endswith('.xlsx'):
        return False
    
    # Check MIME type
    expected_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    if file.content_type != expected_mime:
        return False
    
    # Check file signature (magic numbers)
    current_pos = file.tell()
    file.seek(0)
    header = file.read(4)
    file.seek(current_pos)  # Reset position
    
    # XLSX files are ZIP archives, so they start with PK
    return header[:2] == b'PK'


@app.route("/get-columns", methods=["POST"])  # Changed to POST for file uploads
def get_columns():
    file = request.files.get("file")

    if not file:
        return {"error": "No file provided"}, 400
    
    if not is_xlsx_file(file):
        return {"error": "File must be a valid Excel (.xlsx) file"}, 400
    
    df = pd.read_excel(file)
    print("types:", df.dtypes)  # Debugging line to print data types of columns
    if df.empty:
        return {"error": "The uploaded file is empty"}, 400
    
    columns = []
    for col in df.columns:
        dtype = df[col].dtype
        print("Column:", df[col])  # Debugging line to print column name
        if pd.api.types.is_string_dtype(dtype):
            col_type = "text"
        elif pd.api.types.is_numeric_dtype(dtype):
            col_type = "number"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_type = "date"
        else:
            col_type = "other"
        
        columns.append({"name": col, "type": col_type})
    
    return {"columns": columns}

@app.route("/format-file", methods=["POST"])
def format_file():
    file = request.files.get("file")
    formatting_roles = request.form.get("formattingRoles")

    if not file:
        return {"error": "No file provided"}, 400
    
    if not is_xlsx_file(file):
        return {"error": "File must be a valid Excel (.xlsx) file"}, 400
    
    df = pd.read_excel(file)
    print("types:", df.dtypes)  # Debugging line to print data types of columns
    if df.empty:
        return {"error": "The uploaded file is empty"}, 400
    
    if formatting_roles:
        roles = json.loads(formatting_roles)
        for i, role in enumerate(roles):
            if role["roundDecimals"]:
                df[role["columnName"]] = df[role["columnName"]].round(role["roundDecimals"])
            if role['trimWhitespace']:
                df[role["columnName"]] = df[role["columnName"]].str.strip()
            if role["textCase"]:
                if role["textCase"] == "uppercase":
                    df[role["columnName"]] = df[role["columnName"]].str.upper()
                elif role["textCase"] == "lowercase":
                    df[role["columnName"]] = df[role["columnName"]].str.lower()
                elif role["textCase"] == "titlecase":
                    df[role["columnName"]] = df[role["columnName"]].str.title()
            if role["dateFormat"]:
                df[role["columnName"]] = pd.to_datetime(df[role["columnName"]], errors='coerce')
                if role["dateFormat"] == "DD/MM/YYYY":
                    df[role["columnName"]] = df[role["columnName"]].dt.strftime('%d/%m/%Y')
                elif role["dateFormat"] == "MM/DD/YYYY":
                    df[role["columnName"]] = df[role["columnName"]].dt.strftime('%m/%d/%Y')
                elif role["dateFormat"] == "DD-MM-YYYY":
                    df[role["columnName"]] = df[role["columnName"]].dt.strftime('%d-%m-%Y')
                elif role["dateFormat"] == "YYYY-MM-DD":
                    df[role["columnName"]] = df[role["columnName"]].dt.strftime('%Y-%m-%d')

    preview_data = df.head().where(pd.notnull(df.head()), None).to_dict(orient='records')
    return {"preview": preview_data}, 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)