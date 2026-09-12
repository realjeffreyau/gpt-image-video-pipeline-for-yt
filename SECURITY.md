# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow from the repository Security tab. Do not open a public issue for credentials, suspected secrets, or an exploitable vulnerability.

Include the affected file or workflow, the smallest reproducible example, the impact, and a suggested mitigation when you have one. Remove credentials and personal data from reports.

## Local credential boundary

The pipeline uses local ChatGPT OAuth for Codex stages and a Google AI Studio key for Gemini TTS. Keep both outside commits. `.env`, `.codex/`, generated provider logs, and media are excluded by default.

## Supported security surface

The supported public surface is the current `main` branch and the release files it contains. Dependency and workflow updates are reviewed through GitHub Actions and Dependabot.
