# labpicker

Shared chemical master list + test log + a picker that suggests what to test next.

## How sharing works

The data lives as CSVs in this git repo, and GitHub is the shared "server". Every
command pulls first and pushes after, so everyone always sees everyone's updates, and
git keeps a full history of who changed what. (GCS would also work, but you'd have to
build your own history and conflict handling; git gives both for free.) Use a
**private** repo and give each lab member write access.

- `data/chemicals.csv` – master list: `chem_id, name, smiles, sensitive, sensitivity_note, available`
- `data/tests/*.csv` – the tested log, one file per logging session (so simultaneous
  logging never conflicts): `chem_id, tested_by, tester_lab_id, tested_at, mode, result, notes`

## Setup

Laptop:
```
git clone <your-repo-url> && cd <repo>/labpicker
pip install -e .
export LABPICKER_NAME="Ada Lovelace" LABPICKER_LAB_ID="AL42"   # optional, else it asks
```

Colab (store a GitHub fine-grained token with Contents read/write on this repo as a
Colab secret named `GITHUB_TOKEN`):
```python
from google.colab import userdata
tok = userdata.get("GITHUB_TOKEN")
!git clone https://{tok}@github.com/<owner>/<repo>.git
%cd <repo>/labpicker
!pip install -q -e .
import os; os.environ["LABPICKER_NAME"]="Ada Lovelace"; os.environ["LABPICKER_LAB_ID"]="AL42"
```
Colab can't answer interactive prompts well; use `suggest --dry-run` and then
`labpicker log C00004 C00007 ...` for what you actually ran.

## Usage

```
labpicker status
labpicker add --name "methanol" --smiles CO [--sensitive --note "store at 4C"]
labpicker add --csv new_chemicals.csv
labpicker suggest --mode explore -n 10
labpicker suggest --mode exploit -n 10
labpicker log C00004 C00007 [--result ... --notes ...]
```

`suggest` prints the list, then asks which you *actually* tested (`1 3 4`, `all`, or
Enter to log nothing now) and only those get logged and pushed. Nothing is logged
unless you say so. Never-tested + `available` compounds are the candidates.

### Modes
- **explore**: greedy max-min diversity. Repeatedly picks the compound whose Morgan
  fingerprint is farthest (Tanimoto) from everything already tested or picked. Sensitive
  compounds are **allowed** by default.
- **exploit**: ranks by `labpicker/model.py: score_candidates` (not implemented yet; it
  errors until you do). Sensitive compounds are **excluded** by default.
- Override either with `--include-sensitive` / `--exclude-sensitive`.

## Tests
`pip install -e .[dev] && pytest`
