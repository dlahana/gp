"""Plug your activity model in here for exploit mode.

score_candidates is called with
  candidates  DataFrame of chemicals eligible to test (already filtered)
  chemicals   the full master list
  tests       every logged test so far (columns: see store.TEST_COLUMNS; the
              `result` column is where outcomes go once you start recording them)
and must return one score per candidate row, in the same order; higher = more
likely to be effective.
"""


def score_candidates(candidates, chemicals, tests):
    raise NotImplementedError(
        "Exploit mode needs a model. Implement score_candidates in labpicker/model.py."
    )
