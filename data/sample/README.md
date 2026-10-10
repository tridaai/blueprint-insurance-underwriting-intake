# Sample data

Everything here is **synthetic** — invented insureds, brokers, and loss
histories written for this blueprint. No real submission, carrier, or
broker data is used anywhere in this repository.

- `sample_submissions.json` — the 12 bundled sample submissions (mirror of
  the packaged copy in `src/underwriting_agent/data/`, which is what the
  CLI / demo / API actually load). Includes the two adversarial cases:
  SUB-1011 (a broker relaying a client's "coverage must be in force"
  demand — the bind-language trap) and SUB-1012 (a prompt-injection
  attempt in the broker email).
- `guidelines.json` — the 12 synthetic underwriting guidelines behind
  the retriever (mirror of `src/underwriting_agent/guidelines_data.py`).
