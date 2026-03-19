<div align="center">

<img src="docs/assets/pystamps-logo.svg" alt="pySTAMPS" style="width: 124px; height: auto; max-width: 100%;" />

# pySTAMPS

Python-first StaMPS migration runtime for structured InSAR processing, verification, and reproducible parity workflows.

Run pipeline stages, inspect dataset state, and verify outputs against reference datasets.

<p align="center">
  <a href="docs/index.html"><img src="https://img.shields.io/badge/-Documentation-0f172a?style=for-the-badge&logo=readme&logoColor=white&labelColor=0f172a" alt="Documentation" style="height: 34px;" /></a>
  <a href="docs/quickstart.html"><img src="https://img.shields.io/badge/-Quick%20Start-0f172a?style=for-the-badge&logo=firefoxbrowser&logoColor=white&labelColor=0f172a" alt="Quick Start" style="height: 34px;" /></a>
  <a href="docs/api/pystamps.html"><img src="https://img.shields.io/badge/-API%20Reference-0f172a?style=for-the-badge&logo=python&logoColor=white&labelColor=0f172a" alt="API Reference" style="height: 34px;" /></a>
  <a href="examples/00_pystamps_beginner_walkthrough.ipynb"><img src="https://img.shields.io/badge/-Beginner%20Notebook-0f172a?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=0f172a" alt="Beginner Notebook" style="height: 34px;" /></a>
</p>

</div>

**Author:** Roberto Del Prete

pySTAMPS works with StaMPS-style dataset folders by orchestrating stage execution and optional parity verification in a reproducible way.

## Install

```bash
git clone <repo-url>
cd <repo>
python -m pip install -e .
```

For local development (including docs and tests), use:

```bash
python -m pip install -e ".[dev]"
```

## What pySTAMPS does

- Inspect dataset status and stage progress (`status`)
- Dry-run and execute targeted stage ranges (`run`)
- Validate outputs with explicit comparison flows (`verify`)
- Track compatibility/replay mode for controlled reproducibility

## Read the full docs

- [Introduction (docs index)](docs/index.html)
- [Quick Start](docs/quickstart.html)
- [Getting Started](docs/getting_started.md)
- [Usage and command patterns](docs/usage.html)
- [Configuration](docs/configuration.html)
- [Architecture](docs/architecture.md)
- [Verification](docs/verification.html)
- [API Reference](docs/api/pystamps.html)
- [Release workflow](docs/release.md)

## External notebooks

- `examples/00_pystamps_beginner_walkthrough.ipynb`
- `howtorun.md`

## Governance

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License](LICENSE) (Apache 2.0)
