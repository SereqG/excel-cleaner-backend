from typing import Dict, Any, Optional, Union, List
from langchain_core.tools import tool
import pandas as pd
from pydantic import BaseModel, Field

# --- Pydantic Schemas (with df: Any) ---

class RenameColumnsSchema(BaseModel):
    df: Any = Field(..., description="Input pandas DataFrame")
    column_mapping: Dict[str, str] = Field(..., description="Dictionary mapping old column names to new column names")
    class Config:
        arbitrary_types_allowed = True

class ReplaceValuesSchema(BaseModel):
    df: Any = Field(..., description="Input pandas DataFrame")
    value_mapping: Dict[Any, Any] = Field(..., description="Dictionary mapping old values to new values")
    columns: Optional[List[str]] = Field(None, description="List of column names to apply replacements to")
    class Config:
        arbitrary_types_allowed = True

class ReplaceNullValuesSchema(BaseModel):
    df: Any = Field(..., description="Input pandas DataFrame")
    replacement_value: Any = Field(..., description="Value to replace null/NaN values with")
    columns: Optional[List[str]] = Field(None, description="List of column names to apply replacements to")
    class Config:
        arbitrary_types_allowed = True

class ChangeStringCaseSchema(BaseModel):
    df: Any = Field(..., description="Input pandas DataFrame")
    columns: Optional[List[str]] = Field(None, description="List of column names to apply case change")
    case_type: str = Field("lower", description='Type of case change to apply ("lower", "upper", "title")')
    class Config:
        arbitrary_types_allowed = True

class SetDateFormatSchema(BaseModel):
    df: Any = Field(..., description="Input pandas DataFrame")
    column: str = Field(..., description="Column name to format as date")
    date_format: str = Field(..., description="Desired date format (strftime syntax)")
    class Config:
        arbitrary_types_allowed = True

@tool(args_schema=RenameColumnsSchema)
def rename_columns(df, column_mapping: Dict[str, str]) -> str:
    """
    Renames columns in a pandas DataFrame (direct DataFrame input).

    Args (all required):
        df: Input pandas DataFrame (leave null for now).
        column_mapping: Dictionary mapping old column names to new column names

    Returns:
        dict: A dictionary with:
            - "message": String describing the operation result and showing the new column names.
            - "df": The DataFrame with renamed columns (if not inplace).
    """
    print("column_mapping:", column_mapping)
    inplace = False
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            return "Error: Input must be a pandas DataFrame"
        
        if not isinstance(column_mapping, dict):
            return "Error: column_mapping must be a dictionary"
        
        # Store original columns for comparison
        original_columns = list(df.columns)
        
        # Check if all columns to rename exist
        missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
        if missing_cols:
            return f"Error: The following columns do not exist in the DataFrame: {missing_cols}"
        
        # Perform the renaming
        if inplace:
            df.rename(columns=column_mapping, inplace=True)
            result_df = df
            action = "modified in-place"
        else:
            result_df = df.rename(columns=column_mapping)
            action = "created new DataFrame with renamed columns"
        
        # Prepare result message
        renamed_columns = {old: new for old, new in column_mapping.items() if old in original_columns}
        new_columns = list(result_df.columns)
        
        result_msg = f"Successfully {action}.\n"
        result_msg += f"Renamed columns: {renamed_columns}\n"
        result_msg += f"New column names: {new_columns}"
        
        if not inplace:
            result_msg += f"\nNote: Original DataFrame unchanged. New DataFrame created."

        return {"message": result_msg, "df": result_df}

    except Exception as e:
        return f"Error renaming columns: {str(e)}"
    

@tool(args_schema=ReplaceValuesSchema)
def replace_values(df, value_mapping: Dict[Any, Any], columns: Optional[List[str]] = None) -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Replace values in a pandas DataFrame (direct DataFrame input).

    Args (all required):
        df: Input pandas DataFrame (leave null for now).
        value_mapping: Dictionary mapping old values to new values (supports any type including None/null)
        columns: List of column names to apply replacements to. If None, applies to all columns
    
    Returns:
        dict: A dictionary with:
            - "message": String describing the operation result.
            - "df": The DataFrame with replaced values.
    """
    inplace = False
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            return {"message": "Error: Input must be a pandas DataFrame", "df": None}
        
        if not isinstance(value_mapping, dict):
            return {"message": "Error: value_mapping must be a dictionary", "df": df}
        
        # Validate columns if provided
        if columns is not None:
            if not isinstance(columns, list):
                return {"message": "Error: columns must be a list", "df": df}
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    "message": f"Error: The following columns do not exist in the DataFrame: {missing_cols}", 
                    "df": df
                }
        
        # Count replacements for reporting
        replacement_count = 0
        affected_columns = []
        
        # Perform the replacement - FIXED to avoid chained assignment warnings
        if columns is None:
            # Apply to all columns
            if inplace:
                # Use df.replace() with entire mapping for better performance
                df.replace(value_mapping, inplace=True)
                result_df = df
                action = "modified in-place"
                # Count after replacement for inplace operations
                for old_val in value_mapping.keys():
                    mask = df == value_mapping[old_val]  # Check for new values
                    affected_columns.extend([col for col in df.columns if mask[col].any()])
                replacement_count = len([col for col in affected_columns])  # Approximate count
            else:
                # Create copy and apply replacement
                result_df = df.replace(value_mapping)
                action = "created new DataFrame with replaced values"
                # Count replacements by comparing original vs result
                for old_val, new_val in value_mapping.items():
                    mask = df == old_val
                    replacement_count += mask.sum().sum()
                    affected_columns.extend([col for col in df.columns if mask[col].any()])
        else:
            # Apply to specific columns - FIXED chained assignment
            if inplace:
                for col in columns:
                    # Count before replacement
                    for old_val, new_val in value_mapping.items():
                        mask = df[col] == old_val
                        count = mask.sum()
                        if count > 0:
                            replacement_count += count
                            affected_columns.append(col)
                    
                    # Use proper assignment instead of chained inplace operation
                    df[col] = df[col].replace(value_mapping)
                
                result_df = df
                action = "modified in-place"
            else:
                result_df = df.copy()
                for col in columns:
                    # Count replacements
                    for old_val, new_val in value_mapping.items():
                        mask = result_df[col] == old_val
                        count = mask.sum()
                        if count > 0:
                            replacement_count += count
                            affected_columns.append(col)
                    
                    # Use proper assignment instead of chained inplace operation
                    result_df[col] = result_df[col].replace(value_mapping)
                
                action = "created new DataFrame with replaced values"
        
        # Remove duplicates from affected columns
        affected_columns = list(set(affected_columns))
        
        # Prepare result message
        result_msg = f"Successfully {action}.\n"
        result_msg += f"Total replacements made: {replacement_count}\n"
        result_msg += f"Value mapping applied: {value_mapping}\n"
        result_msg += f"Affected columns: {affected_columns if affected_columns else 'None'}\n"
        
        if columns:
            result_msg += f"Target columns: {columns}\n"
        else:
            result_msg += "Applied to: All columns\n"
        
        if not inplace:
            result_msg += "Note: Original DataFrame unchanged."
        
        return {"message": result_msg, "df": result_df}
        
    except Exception as e:
        return {"message": f"Error replacing values: {str(e)}", "df": df}


# Additional tool for null value replacement with the same format
@tool(args_schema=ReplaceNullValuesSchema)
def replace_null_values(df, replacement_value: Any, columns: Optional[List[str]] = None) -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Replace null/NaN values in a pandas DataFrame with a specified value.

    Args (all required):
        df: Input pandas DataFrame
        replacement_value: Value to replace null/NaN values with (supports any type)
        columns: List of column names to apply replacements to. If None, applies to all columns
    
    Returns:
        dict: A dictionary with:
            - "message": String describing the operation result.
            - "df": The DataFrame with null values replaced.
    """
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            return {"message": "Error: Input must be a pandas DataFrame", "df": None}
        
        # Validate columns if provided
        if columns is not None:
            if not isinstance(columns, list):
                return {"message": "Error: columns must be a list", "df": df}
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    "message": f"Error: The following columns do not exist in the DataFrame: {missing_cols}",
                    "df": df
                }
        
        # Count null values before replacement
        if columns is None:
            null_count = df.isnull().sum().sum()
            affected_columns = [col for col in df.columns if df[col].isnull().any()]
        else:
            null_count = df[columns].isnull().sum().sum()
            affected_columns = [col for col in columns if df[col].isnull().any()]
        
        # Perform the replacement - FIXED to avoid chained assignment
        if columns is None:
            result_df = df.fillna(replacement_value)
            action = "created new DataFrame with null values replaced"
        else:
            result_df = df.copy()
            # Use .loc for explicit assignment to avoid chained assignment warning
            result_df.loc[:, columns] = result_df[columns].fillna(replacement_value)
            action = "created new DataFrame with null values replaced"
        
        # Prepare result message
        result_msg = f"Successfully {action}.\n"
        result_msg += f"Null values replaced: {null_count}\n"
        result_msg += f"Replacement value: {replacement_value}\n"
        result_msg += f"Affected columns: {affected_columns if affected_columns else 'None'}\n"
        
        if columns:
            result_msg += f"Target columns: {columns}\n"
        else:
            result_msg += "Applied to: All columns\n"
        
        result_msg += "Note: Original DataFrame unchanged."
        
        return {"message": result_msg, "df": result_df}
        
    except Exception as e:
        return {"message": f"Error replacing null values: {str(e)}", "df": df}
    
@tool(args_schema=ChangeStringCaseSchema)
def change_string_case(df, columns: Optional[List[str]] = None, case_type: str = "lower") -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Change the case of string values in specified columns of a pandas DataFrame.

    Args (all required):
        df: Input pandas DataFrame
        columns: List of column names to apply case change. If None, applies to all object columns
        case_type: Type of case change to apply ("lower", "upper", "title")

    Returns:
        dict: A dictionary with:
            - "message": String describing the operation result.
            - "df": The DataFrame with string case changed.
    """
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            return {"message": "Error: Input must be a pandas DataFrame", "df": None}

        if columns is not None:
            if not isinstance(columns, list):
                return {"message": "Error: columns must be a list", "df": df}
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    "message": f"Error: The following columns do not exist in the DataFrame: {missing_cols}",
                    "df": df
                }

        # Perform the case change
        result_df = df.copy()
        if columns is None:
            # Apply to all object columns
            object_cols = result_df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                if case_type == "lower":
                    result_df[col] = result_df[col].str.lower()
                elif case_type == "upper":
                    result_df[col] = result_df[col].str.upper()
                elif case_type == "title":
                    result_df[col] = result_df[col].str.title()
        else:
            for col in columns:
                if case_type == "lower":
                    result_df[col] = result_df[col].str.lower()
                elif case_type == "upper":
                    result_df[col] = result_df[col].str.upper()
                elif case_type == "title":
                    result_df[col] = result_df[col].str.title()

        # Prepare result message
        result_msg = f"Successfully changed string case to {case_type}.\n"
        result_msg += f"Applied to columns: {columns if columns else 'All object columns'}\n"

        return {"message": result_msg, "df": result_df}

    except Exception as e:
        return {"message": f"Error changing string case: {str(e)}", "df": df}
    
@tool  # No schema needed, only df
def do_nothing(df) -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Do nothing and return the original DataFrame.

    Args (all required):
        df: Input pandas DataFrame

    Returns:
        dict: A dictionary with:
            - "message": String describing the operation result.
            - "df": The original DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            return {"message": "Error: Input must be a pandas DataFrame", "df": None}

        return {"message": "No changes made.", "df": df}

    except Exception as e:
        return {"message": f"Error in do_nothing: {str(e)}", "df": df}

@tool(args_schema=SetDateFormatSchema)
def set_date_format(df, column: str, date_format: str) -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Set a specific date format for a column in a pandas DataFrame.

    Args:
        df: Input pandas DataFrame
        column: Column name to format as date
        date_format: Desired date format (strftime syntax)

    Returns:
        dict: {
            "message": Description of the operation,
            "df": DataFrame with formatted date column
        }
    """
    try:
        if not isinstance(df, pd.DataFrame):
            return {"message": "Error: Input must be a pandas DataFrame", "df": None}
        if column not in df.columns:
            return {"message": f"Error: Column '{column}' does not exist in the DataFrame", "df": df}
        # Convert column to datetime, then format
        result_df = df.copy()
        try:
            result_df[column] = pd.to_datetime(result_df[column], errors="coerce").dt.strftime(date_format)
        except Exception as e:
            return {"message": f"Error formatting date: {str(e)}", "df": df}
        msg = f"Successfully formatted column '{column}' to date format '{date_format}'."
        return {"message": msg, "df": result_df}
    except Exception as e:
        return {"message": f"Error in set_date_format: {str(e)}", "df": df}

tools = [rename_columns, replace_values, replace_null_values, change_string_case, do_nothing, set_date_format]