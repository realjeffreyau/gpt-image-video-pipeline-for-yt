# Contributing

## Before opening a change

- Keep provider keys, auth files, generated media, raw model responses, and local paths out of commits.
- Keep stage contracts explicit and preserve resumability when changing a worker.
- Keep subprocess calls as argument arrays with `shell=False`.
- Add or update a meaningful test when behavior changes.
- Update the README or relevant documentation when commands, dependencies, or output locations change.

## Validation

Run:

~~~bash
./setup.sh --dev
.venv/bin/python -m pytest -q
~~~

For dependency changes, review the CI dependency-audit result. For workflow changes, inspect permissions and keep actions limited to the access they need.

## Pull requests

Describe the user-visible behavior, the files changed, and the validation run. Keep unrelated generated project state out of the pull request. A maintainer may request a fresh clone or a clean-room run before merging.
