from langchain_core.tools import tool
import pandas as pd

@tool
def rename_column(df, old_name, new_name):
    """
    This tool renames a column name in a DataFrame. 
    Use this tool whenever you need to rename a column name.
    Parameters:
    - df: The DataFrame to modify.
    - old_name: The current name of the column.
    - new_name: The new name for the column.
    """
    df = df.rename(columns={old_name: new_name})
    return df

@tool
def replace_value(df, column, old_value, new_value):
    """
    This tool replaces a value in a DataFrame column.
    Use this tool whenever you need to replace a value in a column.
    Parameters:
    - df: The DataFrame to modify.
    - column: The name of the column to modify.
    - old_value: The value to replace.
    - new_value: The new value to replace the old value.
    """
    df[column] = df[column].replace(old_value, new_value)
    return df

@tool
def replace_empty(df, column, new_value):
    """
    This tool replaces empty values in a DataFrame column.
    Use this tool whenever you need to replace empty values in a column.
    Parameters:
    - df: The DataFrame to modify.
    - column: The name of the column to modify.
    - new_value: The new value to replace empty values.
    """
    df[column] = df[column].replace("", new_value)
    return df

@tool
def format_date(df, column, current_format, new_format):
    """
    This tool formats date values in a DataFrame column.
    Use this tool whenever you need to format date values in a column.
    Parameters:
    - column: The name of the column to modify.
    - current_format: The current date format of the values in the column.
    - new_format: The new date format to apply to the values in the column.
    Skips rows that cannot be formatted.
    """
    formatted_values = []
    for val in df[column]:
        try:
            formatted_values.append(pd.to_datetime(val, format=current_format).strftime(new_format))
        except:
            formatted_values.append(val)
    df[column] = formatted_values
    return df

@tool
def remove_duplicates_from_column(df, column):
    """
    This tool removes duplicate values from a DataFrame column.
    Use this tool whenever you need to remove duplicates from a column.
    Parameters:
    - df: The DataFrame to modify.
    - column: The name of the column to modify.
    """
    df[column] = df[column].drop_duplicates()
    return df

tools = [rename_column, replace_value, replace_empty, format_date, remove_duplicates_from_column]