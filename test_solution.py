import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solution import add_virtual_column


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10, 20, 30],
            "quantity": [2, 3, 4],
            "discount": [1, 2, 3],
        }
    )


@pytest.mark.parametrize(
    ("expression", "expected_values"),
    [
        ("price + quantity", [12, 23, 34]),
        ("price - discount", [9, 18, 27]),
        ("price * quantity", [20, 60, 120]),
    ],
)
def test_add_virtual_column_supported_operations(
    sample_df: pd.DataFrame,
    expression: str,
    expected_values: list[int],
) -> None:
    result = add_virtual_column(sample_df, expression, "result")

    expected = sample_df.copy()
    expected["result"] = expected_values

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "expression",
    [
        "price+quantity",
        " price + quantity ",
        "price   +   quantity",
        "\tprice + quantity\n",
    ],
)
def test_add_virtual_column_accepts_whitespace(
    sample_df: pd.DataFrame,
    expression: str,
) -> None:
    result = add_virtual_column(sample_df, expression, "total")

    expected = sample_df.copy()
    expected["total"] = expected["price"] + expected["quantity"]

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "expression",
    [
        "price / quantity",
        "price ** quantity",
        "price +",
        "+ quantity",
        "price quantity",
        "price + quantity + discount",
        "",
        "   ",
    ],
)
def test_add_virtual_column_rejects_invalid_expression(
    sample_df: pd.DataFrame,
    expression: str,
) -> None:
    result = add_virtual_column(sample_df, expression, "result")

    assert_frame_equal(result, pd.DataFrame())


@pytest.mark.parametrize(
    "expression",
    [
        "missing + quantity",
        "price + missing",
        "left_missing + right_missing",
    ],
)
def test_add_virtual_column_rejects_missing_columns(
    sample_df: pd.DataFrame,
    expression: str,
) -> None:
    result = add_virtual_column(sample_df, expression, "result")

    assert_frame_equal(result, pd.DataFrame())


@pytest.mark.parametrize(
    "new_column",
    [
        "new-column",
        "new column",
        "result1",
        "result!",
        "",
    ],
)
def test_add_virtual_column_rejects_invalid_new_column_name(
    sample_df: pd.DataFrame,
    new_column: str,
) -> None:
    result = add_virtual_column(sample_df, "price + quantity", new_column)

    assert_frame_equal(result, pd.DataFrame())


@pytest.mark.parametrize(
    "invalid_column",
    [
        "invalid-column",
        "invalid column",
        "column1",
        "column!",
        "",
    ],
)
def test_add_virtual_column_rejects_invalid_existing_column_name(
    invalid_column: str,
) -> None:
    df = pd.DataFrame(
        {
            "price": [10, 20],
            "quantity": [2, 3],
            invalid_column: [1, 1],
        }
    )

    result = add_virtual_column(df, "price + quantity", "total")

    assert_frame_equal(result, pd.DataFrame())


def test_add_virtual_column_rejects_non_string_column_name() -> None:
    df = pd.DataFrame(
        {
            "price": [10, 20],
            "quantity": [2, 3],
            123: [1, 1],
        }
    )

    result = add_virtual_column(df, "price + quantity", "total")

    assert_frame_equal(result, pd.DataFrame())


def test_add_virtual_column_does_not_modify_input(
    sample_df: pd.DataFrame,
) -> None:
    original = sample_df.copy(deep=True)

    add_virtual_column(sample_df, "price + quantity", "total")

    assert_frame_equal(sample_df, original)
    assert "total" not in sample_df.columns


@pytest.mark.parametrize(
    ("role", "new_column"),
    [
        (None, "result"),
        (123, "result"),
        ("price + quantity", None),
        ("price + quantity", 123),
    ],
)
def test_add_virtual_column_rejects_non_string_arguments(
    sample_df: pd.DataFrame,
    role: object,
    new_column: object,
) -> None:
    result = add_virtual_column(sample_df, role, new_column)

    assert_frame_equal(result, pd.DataFrame())


def test_add_virtual_column_rejects_non_dataframe_input() -> None:
    result = add_virtual_column(
        {"price": [10], "quantity": [2]},
        "price + quantity",
        "total",
    )

    assert_frame_equal(result, pd.DataFrame())
