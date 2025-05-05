# Job Insight Analyzer

A project providing comprehensive analysis of job postings, resumes, and job-resume matching using AI-powered insights. It includes both a Streamlit web application and a powerful command-line interface (CLI).

## Features

### Common Features
- AI-powered analysis using configurable models via OpenRouter.
- Extracts key requirements, skills, and qualifications from job postings.
- Analyzes resume content for skills, experience, and education.
- Compares resumes against job postings with detailed match scoring and recommendations.

### Streamlit Web Application (`app.py`)
- Interactive web interface for single job analysis, resume analysis, and matching.
- Visualizations like word clouds and sentiment analysis.
- Batch job analysis from XML files.
- Interactive chat to discuss analysis results.
- Downloadable results (JSON, Markdown, PDF).

### Command-Line Interface (`analyze_job_cli.py`)
- **Interactive Menu System**: Easy navigation through analysis options.
- **Rich Terminal UI**: Enhanced user experience with colors, tables, progress bars, and layouts using `rich`.
- **Single Job Analysis**: Analyze postings via URL or pasted text.
- **Job History**: View, manage, and recall previously analyzed jobs.
- **Job Comparison**: Compare multiple job postings side-by-side.
- **Resume Analysis**: Analyze resume text pasted into the CLI.
- **Job-Resume Matching**: Match analyzed resumes against jobs in history.
- **Batch Processing**: Analyze multiple URLs from a file or text files from a directory.
- **Configurable Settings**: Customize API key, model, output directory, history size, etc.
- **Export Results**: Save analysis in JSON, Markdown, or PDF formats.
- **Comprehensive Help System**: Built-in documentation for all features.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd job-insight-analyzer
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs dependencies for both the Streamlit app and the CLI.*

4.  **Create a `.env` file** with your OpenRouter API key:
    ```
    OPENROUTER_API_KEY=your_api_key_here
    ```
    *Alternatively, you can set the API key within the CLI's settings menu.*

## Usage

### Running the Streamlit Application
```bash
streamlit run app.py
```
Access the web interface through your browser. Follow the on-screen instructions for analysis.

### Running the Command-Line Interface (CLI)
```bash
python analyze_job_cli.py
```
This will launch the interactive menu system.

**CLI Menu Options:**

1.  **Analyze Job Posting**:
    -   Choose to input a URL or paste job description text.
    -   The tool fetches/processes the text and performs AI analysis.
    -   View summary, comprehensive report, skills, or raw JSON.
    -   Export results to JSON, Markdown, or PDF.
    -   Analysis is automatically saved to history.

2.  **View Job History**:
    -   Lists previously analyzed jobs with ID, Date, Title, and Company.
    -   Select an ID to view detailed analysis results.
    -   Option to delete entries from history (this also removes the associated result file).

3.  **Compare Jobs**:
    -   Select two or more jobs from the history list by their IDs.
    -   View a side-by-side comparison of basic details and required skills.
    -   See a list of common skills required by all selected jobs.

4.  **Resume Analysis & Matching**:
    -   **Analyze Resume**: Paste resume text to extract skills, experience, etc.
    -   **Match Resume to Job**: Select an analyzed resume and a job from history to calculate a match score, view strengths/gaps, and get recommendations.

5.  **Batch Process Jobs**:
    -   Process multiple job postings from a file containing URLs (one per line).
    -   Process multiple job descriptions from text files within a specified directory.
    -   View a summary of the batch processing results.

6.  **Settings**:
    -   View and modify configuration options like API key, default model, output directory, history size, UI preferences, and default export format.
    -   Settings are saved to `job_analyzer_config.yaml`.

7.  **Help**:
    -   Access detailed documentation on different features and usage.

8.  **Exit**:
    -   Quit the CLI application.

## File Structure

```
.
├── app.py                     # Streamlit application code
├── analyze_job_cli.py         # Command-line interface code
├── prompts.json               # AI analysis prompt templates
├── requirements.txt           # Python dependencies
├── test_app.py                # Test suite for Streamlit app (needs update for CLI)
├── .env                       # Environment variables (API key, not in repo)
├── README.md                  # This documentation file
├── job_analyzer_config.yaml   # CLI configuration file (created on first run/save)
├── job_analysis_history.json  # CLI job analysis history log (created on first analysis)
└── job_analysis_results/      # Default directory for saved analysis files (created on first analysis)
    └── *.json                 # Individual job analysis JSON files
```

## Dependencies

Major dependencies include:
- `streamlit` (for web app)
- `openai` (for OpenRouter API interaction)
- `requests`, `beautifulsoup4` (for web scraping)
- `pandas`, `plotly`, `altair`, `wordcloud`, `textblob` (for data analysis/visualization in web app)
- `python-dotenv` (for environment variables)
- `reportlab` (for PDF generation)
- `rich`, `colorama`, `blessed`, `inquirer`, `pyfiglet`, `tabulate` (for CLI UI)
- `typer`, `tqdm` (for CLI functionality)
- `PyYAML` (for CLI configuration)

See `requirements.txt` for the complete list and versions.

## Development

### Testing
The project includes unit tests primarily for the Streamlit app (`test_app.py`). Tests for the CLI tool need to be added.

To run existing tests:
```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run tests
pytest test_app.py

# Run tests with coverage report
pytest test_app.py --cov=app --cov-report=term-missing
```

### Code Quality
The codebase aims to follow best practices:
- Type hints
- Comprehensive error handling
- Detailed logging (`job_analyzer_cli.log` for CLI)
- Modular design (Classes for Config, History, UI, CLI)

### Contributing
1.  Fork the repository
2.  Create a feature branch
3.  Add tests for new functionality
4.  Ensure all tests pass
5.  Commit your changes
6.  Push to the branch
7.  Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
