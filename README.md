# CADENCE

## Running

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

Install dependencies from the project root:

```powershell
uv sync --locked --dev
```

Set the model API key in the project-root `.env` file:

```dotenv
API_KEY=<your-api-key>
```

Run CADENCE on a standardized candidate-pool JSONL file:

```sh
uv run --env-file .env python -m tools.cadence.run_candidate_pool --candidate-pool data/candidates.jsonl --output runs/cadence/predictions.jsonl
```
