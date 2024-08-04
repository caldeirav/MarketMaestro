# MarketMaestro: An AI Agent for Stock Recommendation

This project contains a stock recommendation agent and an evaluator to assess its performance.

## Installation

1. Ensure you have Python 3.8+ installed on your MacBook.
2. Clone this repository:

   ```
   git clone https://github.com/yourusername/stock-recommendation-project.git
   cd stock-recommendation-project
   ```
3. Replace the placeholder MODEL_SERVICE and API_KEY in src/config.py with your actual values if they differ.
4. Add company reports in PDF format under `data/annual_reports/`. A good source for such reports is the [US Securities and Exchange Commission Filing Search](https://www.sec.gov/search-filings) 10-K and 10-Q reports viewed as HTML and saved in PDF.
5. Run the setup script to create a virtual environment and install dependencies:

   ```
   ./setup.sh
   ```

## Usage example


1. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

2. Run the agent:

   ```
   python run_agent.py
   ```

3. Run the evaluator:

   ```
   python run_evaluator.py
   ```

## Project Structure

- `src/agent.py`: Contains the stock recommendation agent
- `src/evaluator.py`: Contains the evaluation logic
- `src/config.py`: Configuration settings
- `data/annual_reports/`: Directory to store company annual or quarterly reports (PDF format)
- `tests/`: Contains unit tests

## Running Tests

To run tests, activate the virtual environment and run:
```
python -m unittest discover tests
```

## Contributing

For any questions, bugs or feature requests please open an [issue](<https://github.com/finos/MarketMaestro/issues)
For anything else please send an email to {project mailing list}.

To submit a contribution:

1. Fork it (<<https://github.com/finos/MarketMaestro/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Read our [contribution guidelines](.github/CONTRIBUTING.md) and [Community Code of Conduct](https://www.finos.org/code-of-conduct)
4. Commit your changes (`git commit -am 'Add some fooBar'`)
5. Push to the branch (`git push origin feature/fooBar`)
6. Create a new Pull Request

_NOTE:_ Commits and pull requests to FINOS repositories will only be accepted from those contributors with an active, executed Individual Contributor License Agreement (ICLA) with FINOS OR who are covered under an existing and active Corporate Contribution License Agreement (CCLA) executed with FINOS. Commits from individuals not covered under an ICLA or CCLA will be flagged and blocked by the FINOS Clabot tool (or [EasyCLA](https://community.finos.org/docs/governance/Software-Projects/easycla)). Please note that some CCLAs require individuals/employees to be explicitly named on the CCLA.

_Need an ICLA? Unsure if you are covered under an existing CCLA? Email [help@finos.org](mailto:help@finos.org)_

## License

Copyright 2024 Vincent Caldeira

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)
