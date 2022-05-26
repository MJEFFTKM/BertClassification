from typing import Tuple, Union
import numpy as np
import pandas as pd
import re
from transformer import get_df_after_transformer
from config import TRESHHOLD


# -> Tuple(np.array, np.array)
def task1(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    return get_df_after_transformer(df,
                                    have_dataset=False,
                                    dataset=None,
                                    is_train=False)


def find_phone_numbers(description: str):
    result = re.search(r'((8|\+7|7)[\- ]?)?(\(?\d{3}\)?[\- ]?)?(\d[\- ]?){7}', description)
    return result


def find_email(description: str):
    result = re.search(
        r'((([0-9A-Za-z]{1}[-0-9A-z\.]{1,}[0-9A-Za-z]{1})|([0-9А-Яа-я]{1}[-0-9А-я\.]{1,}[0-9А-Яа-я]{1}))@([-A-Za-z]{1,}\.){1,2}[-A-Za-z]{2,})',
        description)
    return result


def find_start_and_end(description: str, label=True):
    if not label:
        return None, None

    phone_res = find_phone_numbers(description)
    if phone_res:
        return phone_res.start(), phone_res.end()

    mail_res = find_email(description)
    if mail_res:
        return mail_res.start(), mail_res.end()

    return 0, len(description) - 1


def task2(df, indices, y_pred):
    indices_true = indices[y_pred > TRESHHOLD]
    indices_false = indices[y_pred < TRESHHOLD]
    start_true, end_true = zip(*df.loc[indices_true].progress_apply(lambda string: find_start_and_end(string)))
    start_false, end_false = zip(
        *df.loc[indices_false].progress_apply(lambda string: find_start_and_end(string, label=False)))
    return np.concatenate([indices_true, indices_false]), np.concatenate([start_true, start_false]), np.concatenate(
        [end_true, end_false])
