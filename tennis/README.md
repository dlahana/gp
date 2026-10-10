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

4. **Daily pipeline (`pipeline.py`, `cli.py`, `.github/workflows/daily.yml`).**
   The pipeline downloads the latest data, retrains, fetches today's fixtures,
   predicts, and submits the most prominent matches to the bot server. The
   server adds dumb nicknames and a corny AI write-up, then **emails you a
   link. Nothing is posted until you approve it.**

## Nicknames: dumb portmanteaus that learn from your friends

```
 surnames ─► enumerate every head+tail splice ─► reward model r(c) ─► sample 2–10 per match
                (al|ca|raz × fe|de|rer:                ▲                 pi(c) ~ exp(r/tau)
                 alcarer, naderer, fedal, ...)         │
                                      friends pick the funniest on /rate/<code>
```

* **Generator (`portmanteau.py`).** Splits surnames into rough syllables
  (handling `ch`, `cz`, `ts`, ... as one sound). It then lists every
  "start of one name + end of the other" in both orders, cutting at
  syllables, inside syllables, or at a shared letter. That gives about
  50–70 candidates per pair, from canonical (sincaraz) to stupid
  (alcaraderer).
* **Reward model (`reward.py`).** Each rating screen shows 4 candidates plus
  "none of these are funny". A pick is a multinomial-logit choice, so the
  model learns `r(c) = w · features(c)`. The features include length, how
  much of each name survives, the kind of cut, silly letters, endings, and
  hashed letter trigrams (so it can learn which *sounds* are funny). It is
  refit automatically after every 20 new votes.
* **Policy.** Every candidate can be listed and scored, so the RLHF objective
  (maximise reward with a KL penalty to uniform) has a closed-form answer,
  `pi(c) ∝ exp(r(c)/tau)`. Posts sample 2–10 nicknames per match from it;
  rating screens mix in uniformly random candidates so the model keeps
  exploring.
* **Name pool.** Rating screens pair real tennis surnames, weighted towards
  big tennis countries (ESP, CZE, ITA, ...). A built-in list of about 110
  players works out of the box; `python -m tennispred name-pool` builds a
  bigger pool from the Sackmann players files (set `BOT_NAME_POOL`).
* **Corny post (`corny.py`).** Claude writes the thread from the
  predictions and nickname options, as structured JSON with a length check.
  If the API key is missing or the call fails, a plain template is used, so
  you still get a draft to approve.

## Bot server (`tennispred/server`)

One small FastAPI app with SQLite:

| URL | who | what |
|---|---|---|
| `/rate/<code>` | friends (one shared link) | tap the funniest nickname, repeat forever |
| `/drafts/<id>?t=<token>` | you (link arrives by email) | edit the tweets, "Rewrite" with a direction, Reject, or **Approve & post** |
| `POST /api/drafts` | daily / insights jobs | submit picks or a match breakdown (admin bearer token) |
| `GET /api/share-link`, `GET /api/admin/stats` | you | the friends' link; votes, raters, what the model has learned |

Run it anywhere that keeps a disk around (Fly.io with a volume, Railway,
Render with a disk, any small VPS):

```bash
docker build -t tennisbot .
docker run -p 8000:8000 -v tennisbot-data:/data \
  -e BOT_ADMIN_TOKEN=... -e BOT_PUBLIC_URL=https://your-host \
  -e ANTHROPIC_API_KEY=... \
  -e SMTP_HOST=smtp.gmail.com -e SMTP_USER=you@gmail.com -e SMTP_PASSWORD=<app password> -e NOTIFY_EMAIL=you@gmail.com \
  -e X_API_KEY=... -e X_API_SECRET=... -e X_ACCESS_TOKEN=... -e X_ACCESS_TOKEN_SECRET=... \
  tennisbot
```

The friends' link (one link for everyone; each browser gets an anonymous id so
votes can be counted per person):

```bash
BOT_SERVER_URL=https://your-host BOT_ADMIN_TOKEN=... python -m tennispred share-link
```

**Notifications.** Email works with any SMTP account. For Gmail, turn on
2-step verification and create an "app password" to use as `SMTP_PASSWORD`.
To add texts later, set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
`TWILIO_FROM`, `NOTIFY_PHONE`; both channels then fire. US carriers require
Twilio numbers to be registered (A2P 10DLC or toll-free verification), which
can take a few days. With no channel configured, the approval link is written
to the server log.

## Match insights posts (on demand)

For a deep dive on one match: GitHub → Actions → **Match insights post** →
Run workflow, then type two players (this also works from the GitHub mobile
app). Or from a terminal:

```bash
python -m tennispred insights "Jannik Sinner" "Carlos Alcaraz" --tournament "Shanghai Masters"           # print it
python -m tennispred insights "Jannik Sinner" "Carlos Alcaraz" --tournament "Shanghai Masters" --submit  # draft + email
```

The breakdown includes:

* the win probability, each player's serve-point and hold rates;
* the factors that moved it (grouped as Elo, serve, return, lefty/righty,
  height, age, workload, ...);
* 500 simulated matches: set scores, deciding sets, tiebreaks, the longest
  match and the longest game;
* a "marathon": the longest of a million simulated service games, point by
  point.

Claude writes it up as a thread from those facts only; you approve it as usual.

**Oddities.** Every daily submission also simulates each match. Anything
strange goes in the approval email for you, never in the post. Examples:
the model disagreeing with Elo, a break-fest, a 30-point game, or a
surprising driver like height or handedness. Any of these can become an
insights post.

## Quick start

```bash
pip install -r requirements.txt

# Real data (Jeff Sackmann's ATP/WTA files)
python -m tennispred download --tour atp --start-year 1985
python -m tennispred backtest --years 2022 2023 2024 2025
python -m tennispred train --model models/atp.json   # prints learned effects

# Predict a day from a CSV (dry run: prints the thread, posts nothing)
python -m tennispred predict --fixtures fixtures.example.csv --date 2026-10-10

# Predict from the live fixture API and send to the bot server for approval
TENNIS_API_KEY=... BOT_SERVER_URL=... BOT_ADMIN_TOKEN=... python -m tennispred predict --fixtures api --submit

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
3. **Deploy the bot server** (above) with the X, email and Anthropic keys.
4. **GitHub secrets** (Settings → Secrets and variables → Actions):
   `TENNIS_API_KEY`, `BOT_SERVER_URL`, `BOT_ADMIN_TOKEN`. The X keys live only
   on the server; GitHub never posts.
5. **Turn on daily drafts.** Add the repository *variable*
   `SUBMIT_DRAFTS = true`. Until then, scheduled runs are dry runs that
   upload the predictions as a build artifact. Manual runs ("Run workflow")
   have a "Send the draft for approval" checkbox.
6. Scheduled workflows only run from the default branch, so merge this branch
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
  portmanteau.py  surname splicing: every head+tail candidate
  reward.py     choice-model reward + exp(r/tau) sampling policy
  name_pool.py  surnames for rating screens (country-weighted)
  corny.py      Claude writes the corny thread (template fallback)
  insights.py   one-match breakdown: drivers, 500 simulations, marathon game, oddities
  server/       FastAPI app: rating page, approval page, email/SMS, posting
  pipeline.py   build / train / backtest / predict
  synthetic.py  synthetic Sackmann-format data with planted effects
  cli.py        python -m tennispred ...
tests/
```
