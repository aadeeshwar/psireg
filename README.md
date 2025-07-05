# PSIREG - Predictive Swarm Intelligence for Renewable Energy Grids

Predictive Swarm Intelligence for Renewable Energy Grids

A hybrid AI approach combining reinforcement learning predictions with swarm intelligence for optimizing renewable energy grid management.

## Development Setup

This project uses Poetry for dependency management and Gradle for build automation.

### Prerequisites

- Python 3.12
- Poetry
- Gradle

### Installation

```bash
poetry install
```

## Available Gradle Tasks

### Primary Tasks

- **`fmt`** - Format code using black
- **`lint`** - Run linting using ruff
- **`type`** - Run type checking using mypy
- **`test`** - Run all tests using pytest
- **`cov`** - Run tests with coverage reporting
- **`build`** - Build the project (runs fmt, lint, type, test)
- **`docs`** - Generate documentation using Sphinx

### Additional Tasks

- **`clean`** - Clean build artifacts
- **`check`** - Run all quality checks (fmt, lint, type, test)
- **`dev`** - Set up development environment
- **`ci`** - Run CI pipeline

### Test-Specific Tasks

- **`testUnit`** - Run unit tests only
- **`testIntegration`** - Run integration tests only
- **`testAcceptance`** - Run acceptance tests only

### Usage Examples

```bash
# Format code
gradle fmt

# Run linting
gradle lint

# Run type checking
gradle type

# Run all tests
gradle test

# Run tests with coverage
gradle cov

# Build the project (runs all quality checks)
gradle build

# Generate documentation
gradle docs

# Clean build artifacts
gradle clean

# Run CI pipeline
gradle ci
```

## Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

All tools are configured in `pyproject.toml`.
