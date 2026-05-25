# Setting up the GitHub repository

This guide walks through publishing `llm-memory-eval` as a public
GitHub repository ready for the journal submission, open-source
contributors, and Zenodo archival.

## 1. Prepare your local copy

Confirm everything is in place:

```bash
cd llm-memory-eval
ls -la                 # README.md, LICENSE, CITATION.cff, src/, tests/, ...
git status             # should be a fresh checkout
```

If this is not yet a git repository:

```bash
git init -b main
git add .
git commit -m "Initial commit: llm-memory-eval v0.1.0"
```

## 2. Create the GitHub repository

In your browser, sign in to GitHub and visit
<https://github.com/new>.

| Field             | Value                                                  |
|-------------------|--------------------------------------------------------|
| Owner             | `okekeag` (or your account / organisation)             |
| Repository name   | `llm-memory-eval`                                      |
| Description       | Reproducible evaluation of Summarization vs RAG memory strategies for LLMs. |
| Visibility        | Public                                                 |
| Initialise with   | *(leave everything unchecked)*                         |

Click **Create repository**.

## 3. Push from the command line

GitHub will show the exact lines; the standard form is:

```bash
git remote add origin https://github.com/okekeag/llm-memory-eval.git
git branch -M main
git push -u origin main
```

If you prefer SSH, replace the URL with
`git@github.com:okekeag/llm-memory-eval.git`.

## 4. Configure repository defaults

Visit *Settings* on the new repository and apply the following.

### Branch protection (Settings → Branches)

- Default branch: `main`
- Require pull-request review before merging.
- Require status checks: select the `CI / Test on Python 3.11`
  workflow once the first CI run completes.
- Require linear history.

### Topics (Settings → General → Topics)

Recommended tags so the repo is discoverable:
`llm`, `rag`, `summarization`, `long-context`, `benchmark`,
`reproducible-research`, `llama`, `dissertation`, `together-ai`.

### Code of Conduct

Already provided as `CODE_OF_CONDUCT.md`. GitHub will surface it on the
*Community Standards* page automatically.

## 5. First-class GitHub features

### Discussions

Settings → Features → Discussions: enable. Useful for questions from
replicators that do not belong in the issue tracker.

### Pages (optional)

If you would like a documentation site rendered from `docs/`, install
mkdocs-material locally and publish to GitHub Pages:

```bash
pip install mkdocs-material
mkdocs gh-deploy
```

Otherwise, the markdown files under `docs/` render natively on
github.com.

### Releases

When the manuscript is accepted:

1. Update `pyproject.toml` and `CITATION.cff` to the release version
   (for example, `1.0.0`).
2. Update `CHANGELOG.md` with the release notes.
3. Tag and push:

   ```bash
   git tag v1.0.0
   git push --tags
   ```

4. Draft the GitHub release at
   <https://github.com/okekeag/llm-memory-eval/releases/new>.

## 6. Archive on Zenodo for a citeable DOI

1. Sign in to <https://zenodo.org/> using your GitHub identity.
2. Navigate to *Settings → GitHub* and toggle the
   `okekeag/llm-memory-eval` repository to *On*.
3. Back on GitHub, publish the first release (Step 5 above). Zenodo
   automatically archives the source tarball and mints a versioned DOI.
4. Replace the placeholder in `CITATION.cff`:

   ```yaml
   doi: 10.5281/zenodo.<real-id>
   ```

   Commit and push the change so the displayed citation reflects the
   real DOI.

## 7. Add a Together AI secret for CI (optional)

CI does **not** require a real API key by default; tests use a stub
client. If you later add integration tests that call Together AI, store
the key as a GitHub Actions secret:

- Settings → Secrets and variables → Actions → New repository secret
- Name: `TOGETHER_API_KEY`
- Value: your key

Reference it in a workflow with `${{ secrets.TOGETHER_API_KEY }}`.

## 8. Connect the manuscript

In the published article, link to the exact commit you used:

> *Reproduction artefacts:* `https://github.com/okekeag/llm-memory-eval`
> at commit `<sha>` (Zenodo DOI: `10.5281/zenodo.<id>`).

The commit SHA is captured automatically in
`results/logs/run_metadata.json` for every reproduction run.

## 9. Encourage replication

Add a "Reproduce in one command" badge to the README using
[GitHub Codespaces](https://github.com/features/codespaces) or
[Zenodo](https://zenodo.org/) so replicators can launch a known-good
environment without local setup.
