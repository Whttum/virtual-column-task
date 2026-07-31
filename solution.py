
import re

import pandas as pd


def add_virtual_column(
    df: pd.DataFrame,
    role: str,
    new_column: str,
) -> pd.DataFrame:
    empty_df = pd.DataFrame()
    label_pattern = r"[A-Za-z_]+"

    if not isinstance(df, pd.DataFrame):
        return empty_df

    if not isinstance(role, str) or not isinstance(new_column, str):
        return empty_df

    if re.fullmatch(label_pattern, new_column) is None:
        return empty_df

    for column in df.columns:
        if not isinstance(column, str):
            return empty_df

        if re.fullmatch(label_pattern, column) is None:
            return empty_df

    expression_pattern = (
        rf"\s*({label_pattern})\s*([+\-*])\s*"
        rf"({label_pattern})\s*"
    )
    match = re.fullmatch(expression_pattern, role)

    if match is None:
        return empty_df

    left_column, operation, right_column = match.groups()

    if left_column not in df.columns or right_column not in df.columns:
        return empty_df

    result = df.copy()

    try:
        if operation == "+":
            result[new_column] = result[left_column] + result[right_column]
        elif operation == "-":
            result[new_column] = result[left_column] - result[right_column]
        else:
            result[new_column] = result[left_column] * result[right_column]
    except (TypeError, ValueError):
        return empty_df

    return result
