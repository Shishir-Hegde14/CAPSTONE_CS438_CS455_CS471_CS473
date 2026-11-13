# Llama 3.1 8B OS evaluation — results and code

This folder packages the OS-evaluation code used for the local Ollama `llama3.1:8b` experiment together with configuration and aggregate result reports.

- `results/`: three run summaries, aggregate metrics, paired comparison, and replay validity.
- `code/`: launcher, evaluator, task data, supporting tools, and vendor files from the evaluation kit.
- `config.yaml`: experiment configuration.

See `results/metrics.json` for accurate/correct counts and status breakdowns. Replay was 45/100 and baseline 34/100 on the same 100 tasks; paired exact two-sided sign-test p = 0.05224, not significant at 0.05. This is a local OS-task result, not the DB task evaluation or a paper-equivalent result.
