# tennispred

Predicts tennis match winners by treating tennis as nested random walks, with
machine learning supplying the step probabilities, and posts a daily thread of
picks to X/Twitter.

## How it works

```
 player history ──► point-in-time features ──► ML: P(server wins a point)
                                                   │  for each player
                                                   ▼
                                     exact random walk:
                              point → game → set → match
                                                   │
                                                   ▼
                     P(match), set-score distribution, hold %, ...
                                                   │
                                                   ▼
                                  daily thread on X (GitHub Actions)
```

1. **Random walk (`markov.py`).** A point is a coin flip that depends on who is
   serving. A game is a walk over the point score (including the deuce loop),
   a tiebreak is a walk with alternating serve, a set is a walk over games
   (tracking who serves first in the next set), and a match is a walk over sets.
   Everything is solved exactly by dynamic programming: no simulation noise.
   It handles best-of-3 and best-of-5, 7-point and 10-point final-set
   tiebreaks (the Grand Slam rule since 2022), and advantage sets. A
   point-by-point Monte Carlo simulator is included as a cross-check.

2. **Learning the step probabilities (`features.py`, `model.py`).** For every
   historical match there are two observations, "A served N points and won K"
   and the same for B. A binomial logistic regression learns

   `logit P(server wins point) = w · x(server, returner, conditions)`

   from the following features. All are computed point-in-time, so nothing
   after the match leaks in.

   | group | features |
   |---|---|
   | strength | overall Elo diff, surface Elo diff, log ranking ratio |
   | serve/return | server's serve rate, returner's return rate (decayed and shrunk to the tour average) |
   | handedness | server lefty, returner lefty, **lefty serving to a righty**, righty serving to a lefty |
   | body | height of each player, height × grass, height × clay |
   | age | age and age² of each player |
   | recent play | form over the last 10 matches, matches in the last 14 days (fatigue), days since the last match (rust), career experience |
   | rivalry | head-to-head record |
   | conditions | surface, best-of-5, tournament level |

   The point model is fit on serve points, not match results, so it uses a
   lot more information per match than a win/loss classifier.

3. **Calibration.** Real point probabilities wobble from match to match, so the
   independent-points chain tends to be over-confident. A single temperature on
   the match log-odds is fit on the most recent 15% of the training window.

4. **Daily pipeline (`pipeline.py`, `cli.py`, `.github/workflows/tennis-daily.yml`).**
   The pipeline downloads the latest data, retrains, fetches today's fixtures,
   predicts, picks the most prominent matches, formats a thread, and posts it.

## Quick start

```bash
cd tennis
pip install -r requirements.txt

# Real data (Jeff Sackmann's ATP/WTA files)
python -m tennispred download --tour atp --start-year 1985
python -m tennispred backtest --years 2022 2023 2024 2025
python -m tennispred train --model models/atp.json   # prints learned effects

# Predict a day from a CSV (dry run: prints the thread, posts nothing)
python -m tennispred predict --fixtures fixtures.example.csv --date 2026-10-10

# Predict from the live fixture API and post
TENNIS_API_KEY=... X_API_KEY=... python -m tennispred predict --fixtures api --post

# Offline demo with synthetic data (known ground truth)
python -m tennispred synth --data-dir /tmp/syn
python -m tennispred backtest --data-dir /tmp/syn --start-year 2000 --train-from 2016-01-01 --years 2020 2021

pytest
```

`--tour wta` works for every command (WTA matches are best of 3).

## Setting up the daily posts

1. **X developer app.** Create an app at developer.x.com with
   **Read and write** permission, then generate an access token and secret
   *after* setting that permission. Check X's current API tiers: posting
   limits and prices change.
2. **Fixtures.** Sign up at api-tennis.com for an API key. The client in
   `fixtures.py` follows their documented `get_fixtures` fields but has not
   been tested against a live response. Run `predict --fixtures api` once by
   hand and check the output. To use a different provider, write a class with
   a `fixtures(day) -> list[Fixture]` method.
3. **GitHub secrets** (Settings → Secrets and variables → Actions):
   `TENNIS_API_KEY`, `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`,
   `X_ACCESS_TOKEN_SECRET`.
4. **Turn on posting.** Add the repository *variable* `POST_TO_X = true`.
   Until then, scheduled runs are dry runs that upload the predictions as a
   build artifact. Manual runs ("Run workflow") have a "Post to X" checkbox.
5. Scheduled workflows only run from the default branch, so merge this branch
   first.

## Caveats

* **Data lag.** Sackmann's files are updated by hand and can trail real
  results by weeks. When the data is stale, the pipeline warns, and computes
  rest and fatigue as of the end of the data rather than treating everyone as
  rusty. Recent form will still be missing. For sharper daily picks, feed
  recent results from your fixture provider into the history too.
* **License.** Sackmann's data is CC BY-NC-SA 4.0: non-commercial use, with
  attribution. Credit "data: Jeff Sackmann / tennis_abstract" in the bot's
  bio.
* **Independent points.** The random walk assumes every point is independent
  given the server. Momentum and big-point effects are not modelled; the
  calibration temperature partly absorbs them.
* These are probabilities, not betting advice.

## Layout

```
tennispred/
  markov.py     exact point→game→tiebreak→set→match DP + Monte Carlo
  features.py   chronological replay, Elo, point-in-time features
  model.py      binomial logistic regression, calibration, save/load
  data.py       Sackmann download/load
  fixtures.py   CSV and api-tennis.com fixture sources
  names.py      fixture name → player id matching
  twitter.py    thread formatting + posting (tweepy, X API v2)
  pipeline.py   build / train / backtest / predict
  synthetic.py  synthetic Sackmann-format data with planted effects
  cli.py        python -m tennispred ...
tests/
```
