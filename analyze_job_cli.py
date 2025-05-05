#!/usr/bin/env python3
"""
Job Insight Analyzer CLI

A powerful command-line interface for analyzing job postings, managing job history,
comparing multiple jobs, and matching resumes to job postings.

This enhanced CLI tool provides a comprehensive, user-friendly interface with
rich terminal UI elements, interactive menus, and advanced functionality.
"""

import sys
import os
import json
import logging
import argparse
import datetime
import re
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Rich terminal UI libraries
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
import inquirer
from pyfiglet import Figlet
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
from tabulate import tabulate

# Import functions from app.py
from app import (
    load_prompts,
    get_api_key,
    fetch_and_clean_url_content,
    analyze_job_posting,
    generate_report_markdown,
    generate_pdf_report,
    analyze_resume_text,
    analyze_match
)

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("job_analyzer_cli.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("job_analyzer_cli")

# Constants
CONFIG_FILE = "job_analyzer_config.yaml"
HISTORY_FILE = "job_analysis_history.json"
DEFAULT_OUTPUT_DIR = "job_analysis_results"
VERSION = "1.0.0"

# ASCII Art for the application header
def get_ascii_header():
    """Generate ASCII art header for the application."""
    figlet = Figlet(font="slant")
    header = figlet.renderText("Job Insight Analyzer")
    return f"{Fore.CYAN}{header}{Style.RESET_ALL}"

class JobAnalyzerConfig:
    """Configuration manager for the Job Analyzer CLI."""

    def __init__(self):
        """Initialize configuration with default values."""
        self.config = {
            "api_key": "",
            "default_model": "google/gemini-2.5-flash-preview:online",
            "output_directory": DEFAULT_OUTPUT_DIR,
            "max_history_items": 50,
            "color_theme": "default",
            "show_progress_bars": True,
            "default_format": "all",  # Options: json, markdown, pdf, all
            "auto_open_results": False,
        }
        self.load_config()

    def load_config(self):
        """
        Loads configuration settings from the specified YAML file.

        If the configuration file exists and is valid, its contents update
        the default configuration. Errors during loading are logged.
        """
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config and isinstance(loaded_config, dict):
                        self.config.update(loaded_config)
                        logger.info(f"Configuration loaded from {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {CONFIG_FILE}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def get(self, key, default=None):
        """
        Retrieves a configuration value by key.

        Args:
            key: The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        return self.config.get(key, default)

    def set(self, key, value):
        """
        Sets a configuration value and saves the configuration.

        Args:
            key: The configuration key.
            value: The value to set.

        Returns:
            True if saving was successful after setting the value, False otherwise.
        """
        self.config[key] = value
        return self.save_config()

    def update(self, config_dict):
        """
        Updates multiple configuration values from a dictionary and saves the configuration.

        Args:
            config_dict: A dictionary containing key-value pairs to update.

        Returns:
            True if saving was successful after updating, False otherwise.
        """
        self.config.update(config_dict)
        return self.save_config()


class JobHistoryManager:
    """Manages job analysis history."""

    def __init__(self, config: JobAnalyzerConfig):
        """Initialize the history manager."""
        self.config = config
        self.history = []
        self.load_history()

    def load_history(self):
        """
        Loads job analysis history from the JSON history file.

        If the history file exists and is valid, its contents populate the
        history list. Errors during loading are logged.
        """
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    loaded_history = json.load(f)
                    if isinstance(loaded_history, list):
                        self.history = loaded_history
                        logger.info(f"Loaded {len(self.history)} history items")
        except Exception as e:
            logger.error(f"Error loading history: {e}")

    def save_history(self):
        """
        Saves the current job analysis history to the JSON history file.

        Ensures the history size does not exceed the configured maximum.

        Returns:
            True if saving was successful, False otherwise. Errors are logged.
        """
        try:
            # Ensure we don't exceed the maximum history items
            max_items = self.config.get("max_history_items", 50)
            if len(self.history) > max_items:
                self.history = self.history[-max_items:]

            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Saved {len(self.history)} history items")
            return True
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return False

    def add_job_analysis(self, job_data: Dict):
        """
        Adds a new job analysis result to the history.

        Creates a unique ID, saves the full analysis data to a file in the
        output directory, and adds a summary entry to the history list.

        Args:
            job_data: A dictionary containing the full job analysis results.

        Returns:
            The history entry dictionary for the added analysis.
        """
        # Extract key information for the history entry
        timestamp = datetime.datetime.now().isoformat()
        title = job_data.get("extraction", {}).get("title", "Unknown Position")
        company = job_data.get("extraction", {}).get("company", "Unknown Company")
        url = job_data.get("url", "")

        # Create a unique ID for the analysis
        analysis_id = f"{len(self.history) + 1}_{timestamp.replace(':', '-')}"

        # Create the history entry
        history_entry = {
            "id": analysis_id,
            "timestamp": timestamp,
            "title": title,
            "company": company,
            "url": url,
            "file_path": f"{self.config.get('output_directory')}/{analysis_id}.json"
        }

        # Save the full analysis to a file
        os.makedirs(self.config.get("output_directory"), exist_ok=True)
        with open(history_entry["file_path"], 'w') as f:
            json.dump(job_data, f, indent=2)

        # Add to history and save
        self.history.append(history_entry)
        self.save_history()
        return history_entry

    def get_job_analysis(self, analysis_id: str) -> Optional[Dict]:
        """
        Retrieves a full job analysis result from file based on its history ID.

        Args:
            analysis_id: The unique ID of the analysis entry in the history.

        Returns:
            A dictionary containing the full analysis data, or None if the
            ID is not found or the file cannot be loaded.
        """
        # Find the history entry
        entry = next((item for item in self.history if item["id"] == analysis_id), None)
        if not entry:
            return None

        # Load the full analysis from file
        try:
            with open(entry["file_path"], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis {analysis_id}: {e}")
            return None

    def delete_job_analysis(self, analysis_id: str) -> bool:
        """
        Deletes a job analysis entry from history and its corresponding file.

        Args:
            analysis_id: The unique ID of the analysis entry to delete.

        Returns:
            True if the entry was found and deleted, False otherwise. Errors
            during file removal are logged but do not prevent history update.
        """
        entry = next((item for item in self.history if item["id"] == analysis_id), None)
        if not entry:
            return False

        # Remove the file if it exists
        try:
            if os.path.exists(entry["file_path"]):
                os.remove(entry["file_path"])
        except Exception as e:
            logger.error(f"Error removing analysis file: {e}")

        # Remove from history and save
        self.history = [item for item in self.history if item["id"] != analysis_id]
        self.save_history()
        return True

    def get_history_table(self) -> List[Dict]:
        """
        Retrieves the job analysis history formatted for display in a table.

        Returns:
            A list of history entry dictionaries, sorted by timestamp in
            descending order (most recent first).
        """
        return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)

class UIManager:
    """Manages the terminal UI elements."""

    def __init__(self, config: JobAnalyzerConfig):
        """Initialize the UI manager."""
        self.config = config
        self.console = Console()

    def clear_screen(self):
        """
        Clears the terminal screen using the appropriate command for the OS.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """
        Prints the application's ASCII art header and title panel.
        """
        self.clear_screen()
        self.console.print(get_ascii_header())
        self.console.print(Panel(f"[bold cyan]Job Insight Analyzer CLI v{VERSION}[/bold cyan]",
                                 subtitle="[italic]Powerful job analysis at your fingertips[/italic]"))
        self.console.print()

    def print_section_header(self, title: str):
        """Print a section header."""
        self.console.print()
        self.console.print(f"[bold yellow]{'=' * 20} {title} {'=' * 20}[/bold yellow]")
        self.console.print()

    def print_success(self, message: str):
        """
        Prints a success message with a green checkmark.

        Args:
            message: The success message text.
        """
        self.console.print(f"[bold green]✓ {message}[/bold green]")

    def print_error(self, message: str):
        """
        Prints an error message with a red cross mark.

        Args:
            message: The error message text.
        """
        self.console.print(f"[bold red]✗ {message}[/bold red]")

    def print_warning(self, message: str):
        """
        Prints a warning message with a yellow warning sign.

        Args:
            message: The warning message text.
        """
        self.console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    def print_info(self, message: str):
        """
        Prints an informational message with a blue info sign.

        Args:
            message: The info message text.
        """
        self.console.print(f"[bold blue]ℹ {message}[/bold blue]")

    def print_markdown(self, markdown_text: str):
        """
        Prints formatted markdown text to the console using Rich's Markdown renderer.

        Args:
            markdown_text: The markdown string to print.
        """
        self.console.print(Markdown(markdown_text))

    def print_json(self, json_data: Dict):
        """
        Prints formatted JSON data with syntax highlighting.

        Args:
            json_data: The dictionary to print as JSON.
        """
        json_str = json.dumps(json_data, indent=2)
        self.console.print(Syntax(json_str, "json", theme="monokai", line_numbers=True))

    def create_table(self, title: str, columns: List[str]) -> Table:
        """
        Creates a Rich Table object with the specified title and columns.

        Args:
            title: The title of the table.
            columns: A list of strings for the table column headers.

        Returns:
            A Rich Table instance.
        """
        table = Table(title=title, box=box.ROUNDED)
        for column in columns:
            table.add_column(column, style="cyan")
        return table

    def progress_context(self, description: str, total: int = 100):
        """
        Creates a Rich Progress context manager for displaying progress bars.

        Args:
            description: The text description for the progress bar task.
            total: The total number of steps for the progress bar.

        Returns:
            A Rich Progress context manager.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )

    def get_menu_choice(self, title: str, options: List[str]) -> str:
        """
        Displays a menu using Inquirer and gets the user's selection.

        Args:
            title: The title of the menu.
            options: A list of strings representing the menu options.

        Returns:
            The selected option string, or None if no selection was made.
        """
        questions = [
            inquirer.List('choice',
                          message=title,
                          choices=options,
                          carousel=True)
        ]
        answers = inquirer.prompt(questions)
        return answers['choice'] if answers else None

    def get_input(self, prompt: str, default: str = "") -> str:
        """
        Gets text input from the user using Rich's Prompt.

        Args:
            prompt: The prompt message to display.
            default: A default value for the input (optional).

        Returns:
            The user's input string.
        """
        return Prompt.ask(prompt, default=default)

    def get_confirmation(self, prompt: str, default: bool = False) -> bool:
        """
        Gets a yes/no confirmation from the user using Rich's Confirm.

        Args:
            prompt: The confirmation prompt message.
            default: The default answer (True for yes, False for no).

        Returns:
            True if the user confirms (yes), False otherwise (no).
        """
        return Confirm.ask(prompt, default=default)

    def display_job_summary(self, job_data: Dict):
        """
        Displays a formatted summary of a job analysis using Rich Layouts and Panels.

        Shows key details (title, company, location, etc.) and a summary of skills.

        Args:
            job_data: A dictionary containing the job analysis results.
        """
        extraction = job_data.get("extraction", {})
        skills = job_data.get("skills", {}).get("technical", {})

        # Create a layout for the summary
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Header content
        title = extraction.get("title", "Unknown Position")
        company = extraction.get("company", "Unknown Company")
        header_content = Panel(
            f"[bold cyan]{title}[/bold cyan]\n[yellow]{company}[/yellow]",
            title="Job Summary",
            border_style="blue"
        )

        # Left panel content - Job details
        left_content = Panel(
            "\n".join([
                f"[bold]Level:[/bold] {extraction.get('level', 'Not specified')}",
                f"[bold]Location:[/bold] {extraction.get('location', 'Not specified')}",
                f"[bold]Work Model:[/bold] {extraction.get('work_model', 'Not specified')}",
                f"[bold]Experience:[/bold] {extraction.get('experience', 'Not specified')}",
                f"[bold]Education:[/bold] {extraction.get('education', 'Not specified')}"
            ]),
            title="Details",
            border_style="green"
        )

        # Right panel content - Skills
        required_skills = skills.get("required", [])
        preferred_skills = skills.get("preferred", [])

        skills_content = []
        if required_skills:
            skills_content.append("[bold]Required Skills:[/bold]")
            for skill in required_skills[:5]:  # Show top 5 skills
                skills_content.append(f"• {skill}")

        if preferred_skills:
            skills_content.append("\n[bold]Preferred Skills:[/bold]")
            for skill in preferred_skills[:3]:  # Show top 3 preferred skills
                skills_content.append(f"• {skill}")

        right_content = Panel(
            "\n".join(skills_content),
            title="Skills",
            border_style="yellow"
        )

        # Assign content to layout
        layout["header"].update(header_content)
        layout["main"]["left"].update(left_content)
        layout["main"]["right"].update(right_content)

        # Print the layout
        self.console.print(layout)

class JobAnalyzerCLI:
    """Main CLI class for the Job Analyzer application."""

    def __init__(self):
        """Initialize the CLI application."""
        self.config = JobAnalyzerConfig()
        self.ui = UIManager(self.config)
        self.history_manager = JobHistoryManager(self.config)
        self.prompts_data = None
        self.api_key = None
        self.current_job_analysis = None
        self.current_resume_analysis = None

    def initialize(self):
        """
        Initializes the CLI application by loading prompts and retrieving the API key.

        Also ensures the output directory exists.

        Returns:
            True if initialization was successful, False otherwise.
        """
        # Load prompts
        self.prompts_data = load_prompts()
        if not self.prompts_data:
            self.ui.print_error("Could not load prompts.json")
            return False

        # Get API key
        self.api_key = get_api_key()
        if not self.api_key:
            self.api_key = self.config.get("api_key", "")

        if not self.api_key:
            self.ui.print_warning("API key not found. You'll need to provide it in the settings.")

        # Create output directory if it doesn't exist
        output_dir = self.config.get("output_directory")
        os.makedirs(output_dir, exist_ok=True)

        return True

    def run(self):
        """
        Runs the main loop of the CLI application.

        Displays the main menu and handles user interaction to navigate
        to different functionalities.

        Returns:
            An exit code (0 for success, 1 for initialization failure).
        """
        if not self.initialize():
            return 1

        while True:
            self.ui.print_header()

            # Show API key status
            if self.api_key:
                self.ui.print_info("API Key: Configured ✓")
            else:
                self.ui.print_warning("API Key: Not configured ✗")

            # Main menu
            choice = self.ui.get_menu_choice("Select an option:", [
                "1. Analyze Job Posting",
                "2. View Job History",
                "3. Compare Jobs",
                "4. Resume Analysis & Matching",
                "5. Batch Process Jobs",
                "6. Settings",
                "7. Help",
                "8. Exit"
            ])

            if not choice:
                continue

            # Process menu choice
            if choice.startswith("1"):
                self.analyze_job_menu()
            elif choice.startswith("2"):
                self.view_history_menu()
            elif choice.startswith("3"):
                self.compare_jobs_menu()
            elif choice.startswith("4"):
                self.resume_analysis_menu()
            elif choice.startswith("5"):
                self.batch_process_menu()
            elif choice.startswith("6"):
                self.settings_menu()
            elif choice.startswith("7"):
                self.help_menu()
            elif choice.startswith("8"):
                self.ui.print_info("Exiting Job Insight Analyzer. Goodbye!")
                break

        return 0

    def analyze_job_menu(self):
        """Menu for analyzing a job posting."""
        self.ui.print_header()
        self.ui.print_section_header("Analyze Job Posting")

        # Check API key
        if not self.api_key:
            self.ui.print_error("API key not configured. Please set it in the settings menu.")
            self.ui.get_input("Press Enter to continue...")
            return

        # Get job URL or text
        input_type = self.ui.get_menu_choice("Select input method:", [
            "1. Enter job posting URL",
            "2. Paste job description text",
            "3. Back to main menu"
        ])

        if not input_type or input_type.startswith("3"):
            return

        job_text = None
        job_url = None

        if input_type.startswith("1"):
            job_url = self.ui.get_input("Enter job posting URL:")
            if not job_url:
                self.ui.print_warning("No URL provided.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Fetch and clean job content
            with self.ui.progress_context("Fetching job content...") as progress:
                task = progress.add_task("Downloading...", total=100)
                progress.update(task, advance=30)

                job_text = fetch_and_clean_url_content(job_url)
                progress.update(task, advance=70)

            if not job_text:
                self.ui.print_error("Could not fetch or clean job content.")
                self.ui.get_input("Press Enter to continue...")
                return

        elif input_type.startswith("2"):
            self.ui.print_info("Paste job description text below (press Ctrl+D or Ctrl+Z on a new line when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                job_text = "\n".join(lines)

            if not job_text:
                self.ui.print_warning("No text provided.")
                self.ui.get_input("Press Enter to continue...")
                return

        # Select model
        model = self.config.get("default_model")

        # Analyze job posting
        self.ui.print_info(f"Analyzing job posting using {model}...")

        with self.ui.progress_context("Analyzing job posting...") as progress:
            task = progress.add_task("Processing...", total=100)
            progress.update(task, advance=20)

            analysis_result = analyze_job_posting(self.api_key, model, job_text, self.prompts_data)
            progress.update(task, advance=50)

            if not analysis_result:
                self.ui.print_error("Job analysis failed.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Add URL to the result if provided
            if job_url:
                analysis_result["url"] = job_url

            # Generate comprehensive markdown report
            progress.update(task, description="Generating comprehensive report...")
            comprehensive_md = generate_report_markdown(
                self.api_key,
                model,
                analysis_result,
                "comprehensive_report_markdown",
                self.prompts_data
            )
            progress.update(task, advance=30)

            # Add the report to the analysis result
            analysis_result["comprehensive_report"] = comprehensive_md

        # Add to history
        history_entry = self.history_manager.add_job_analysis(analysis_result)

        # Set as current job analysis
        self.current_job_analysis = analysis_result

        # Display results
        self.display_job_analysis(analysis_result, history_entry)

    def display_job_analysis(self, analysis_result: Dict, history_entry: Dict):
        """Display the results of a job analysis."""
        self.ui.print_header()
        self.ui.print_section_header("Job Analysis Results")

        # Display job summary
        self.ui.display_job_summary(analysis_result)

        # Options for viewing details
        while True:
            choice = self.ui.get_menu_choice("Select an option:", [
                "1. View Comprehensive Report",
                "2. View Technical Skills",
                "3. View JSON Data",
                "4. Export Results",
                "5. Back to Main Menu"
            ])

            if not choice or choice.startswith("5"):
                break

            if choice.startswith("1"):
                # View comprehensive report
                self.ui.print_header()
                self.ui.print_section_header("Comprehensive Report")
                comprehensive_report = analysis_result.get("comprehensive_report", "No report available.")
                self.ui.print_markdown(comprehensive_report)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("2"):
                # View technical skills
                self.ui.print_header()
                self.ui.print_section_header("Technical Skills")

                skills = analysis_result.get("skills", {}).get("technical", {})
                required = skills.get("required", [])
                preferred = skills.get("preferred", [])

                table = self.ui.create_table("Technical Skills", ["Type", "Skill"])

                for skill in required:
                    table.add_row("Required", skill)

                for skill in preferred:
                    table.add_row("Preferred", skill)

                self.ui.console.print(table)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("3"):
                # View JSON data
                self.ui.print_header()
                self.ui.print_section_header("JSON Data")
                self.ui.print_json(analysis_result)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("4"):
                # Export results
                self.export_results_menu(analysis_result, history_entry)

    def export_results_menu(self, analysis_result: Dict, history_entry: Dict):
        """Menu for exporting analysis results."""
        self.ui.print_header()
        self.ui.print_section_header("Export Results")

        # Get export format
        format_choice = self.ui.get_menu_choice("Select export format:", [
            "1. JSON",
            "2. Markdown",
            "3. PDF",
            "4. All Formats",
            "5. Back"
        ])

        if not format_choice or format_choice.startswith("5"):
            return

        # Get output directory
        output_dir = self.config.get("output_directory")
        custom_dir = self.ui.get_input(f"Output directory (default: {output_dir}):", output_dir)

        if custom_dir:
            output_dir = custom_dir
            os.makedirs(output_dir, exist_ok=True)

        # Get base filename
        extraction = analysis_result.get("extraction", {})
        default_filename = f"{extraction.get('company', 'company')}_{extraction.get('title', 'job')}"
        default_filename = re.sub(r'[^\w\-_\.]', '_', default_filename)

        filename = self.ui.get_input(f"Base filename (default: {default_filename}):", default_filename)
        if not filename:
            filename = default_filename

        # Export based on format choice
        if format_choice.startswith("1") or format_choice.startswith("4"):  # JSON
            json_path = os.path.join(output_dir, f"{filename}.json")
            with open(json_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            self.ui.print_success(f"JSON exported to: {json_path}")

        if format_choice.startswith("2") or format_choice.startswith("4"):  # Markdown
            md_path = os.path.join(output_dir, f"{filename}.md")
            with open(md_path, 'w') as f:
                f.write(analysis_result.get("comprehensive_report", "No report available."))
            self.ui.print_success(f"Markdown exported to: {md_path}")

        if format_choice.startswith("3") or format_choice.startswith("4"):  # PDF
            pdf_path = os.path.join(output_dir, f"{filename}.pdf")
            pdf_bytes = generate_pdf_report(analysis_result, "job")
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            self.ui.print_success(f"PDF exported to: {pdf_path}")

        self.ui.get_input("Press Enter to continue...")

    def view_history_menu(self):
        """Menu for viewing job analysis history."""
        while True:
            self.ui.print_header()
            self.ui.print_section_header("Job Analysis History")

            history_items = self.history_manager.get_history_table()

            if not history_items:
                self.ui.print_info("No job analyses in history.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Create a table to display history
            table = self.ui.create_table("Analysis History", ["ID", "Date", "Title", "Company"])

            for i, item in enumerate(history_items):
                # Format the date
                timestamp = datetime.datetime.fromisoformat(item["timestamp"])
                date_str = timestamp.strftime("%Y-%m-%d %H:%M")

                table.add_row(
                    str(i + 1),
                    date_str,
                    item["title"],
                    item["company"]
                )

            self.ui.console.print(table)

            # Options
            options = ["Back to Main Menu"]
            if history_items:
                options.insert(0, "View Analysis Details")
                options.insert(1, "Delete Analysis")

            choice = self.ui.get_menu_choice("Select an option:", options)

            if choice == "Back to Main Menu":
                break

            if choice == "View Analysis Details":
                # Get the analysis ID to view
                index = self.ui.get_input("Enter the ID number to view (or 0 to cancel):")
                try:
                    index = int(index) - 1
                    if index < 0:
                        continue

                    if index < len(history_items):
                        analysis_id = history_items[index]["id"]
                        analysis_result = self.history_manager.get_job_analysis(analysis_id)

                        if analysis_result:
                            # Set as current job analysis
                            self.current_job_analysis = analysis_result

                            # Display the analysis
                            self.display_job_analysis(analysis_result, history_items[index])
                        else:
                            self.ui.print_error(f"Could not load analysis with ID {analysis_id}")
                            self.ui.get_input("Press Enter to continue...")
                    else:
                        self.ui.print_error("Invalid ID number.")
                        self.ui.get_input("Press Enter to continue...")
                except ValueError:
                    self.ui.print_error("Please enter a valid number.")
                    self.ui.get_input("Press Enter to continue...")

            elif choice == "Delete Analysis":
                # Get the analysis ID to delete
                index = self.ui.get_input("Enter the ID number to delete (or 0 to cancel):")
                try:
                    index = int(index) - 1
                    if index < 0:
                        continue

                    if index < len(history_items):
                        analysis_id = history_items[index]["id"]

                        # Confirm deletion
                        if self.ui.get_confirmation(f"Are you sure you want to delete the analysis for '{history_items[index]['title']}'?"):
                            if self.history_manager.delete_job_analysis(analysis_id):
                                self.ui.print_success("Analysis deleted successfully.")
                            else:
                                self.ui.print_error("Failed to delete analysis.")

                        self.ui.get_input("Press Enter to continue...")
                    else:
                        self.ui.print_error("Invalid ID number.")
                        self.ui.get_input("Press Enter to continue...")
                except ValueError:
                    self.ui.print_error("Please enter a valid number.")
                    self.ui.get_input("Press Enter to continue...")

    def compare_jobs_menu(self):
        """Menu for comparing multiple job postings."""
        self.ui.print_header()
        self.ui.print_section_header("Compare Jobs")

        history_items = self.history_manager.get_history_table()

        if len(history_items) < 2:
            self.ui.print_warning("You need at least 2 job analyses to compare. Please analyze more jobs first.")
            self.ui.get_input("Press Enter to continue...")
            return

        # Display available jobs
        table = self.ui.create_table("Available Jobs", ["ID", "Title", "Company"])

        for i, item in enumerate(history_items):
            table.add_row(
                str(i + 1),
                item["title"],
                item["company"]
            )

        self.ui.console.print(table)

        # Select jobs to compare
        self.ui.print_info("Select two or more jobs to compare (comma-separated IDs, e.g., '1,3'):")
        selection = self.ui.get_input("Job IDs:")

        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(",")]

            # Validate selection
            if len(selected_indices) < 2:
                self.ui.print_error("Please select at least 2 jobs to compare.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Check if indices are valid
            if any(idx < 0 or idx >= len(history_items) for idx in selected_indices):
                self.ui.print_error("One or more selected IDs are invalid.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Load selected job analyses
            selected_jobs = []
            for idx in selected_indices:
                analysis_id = history_items[idx]["id"]
                analysis = self.history_manager.get_job_analysis(analysis_id)
                if analysis:
                    selected_jobs.append({
                        "id": analysis_id,
                        "title": history_items[idx]["title"],
                        "company": history_items[idx]["company"],
                        "data": analysis
                    })
                else:
                    self.ui.print_error(f"Could not load analysis for {history_items[idx]['title']}")

            if len(selected_jobs) < 2:
                self.ui.print_error("Could not load enough job analyses for comparison.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Perform comparison
            self.display_job_comparison(selected_jobs)

        except ValueError:
            self.ui.print_error("Please enter valid job IDs.")
            self.ui.get_input("Press Enter to continue...")

    def display_job_comparison(self, jobs: List[Dict]):
        """Display a comparison of multiple job analyses."""
        self.ui.print_header()
        self.ui.print_section_header("Job Comparison")

        # Create comparison table for basic details
        basic_table = self.ui.create_table("Basic Details", ["Attribute"] + [f"{job['company']}: {job['title']}" for job in jobs])

        # Add rows for basic attributes
        attributes = [
            ("Level", lambda j: j["data"].get("extraction", {}).get("level", "N/A")),
            ("Location", lambda j: j["data"].get("extraction", {}).get("location", "N/A")),
            ("Work Model", lambda j: j["data"].get("extraction", {}).get("work_model", "N/A")),
            ("Experience", lambda j: j["data"].get("extraction", {}).get("experience", "N/A")),
            ("Education", lambda j: j["data"].get("extraction", {}).get("education", "N/A"))
        ]

        for attr_name, attr_func in attributes:
            basic_table.add_row(attr_name, *[attr_func(job) for job in jobs])

        self.ui.console.print(basic_table)

        # Create comparison table for skills
        skills_table = self.ui.create_table("Required Skills Comparison", ["Job"] + ["Required Skills"])

        for job in jobs:
            required_skills = job["data"].get("skills", {}).get("technical", {}).get("required", [])
            skills_str = "\n".join([f"• {skill}" for skill in required_skills[:10]])
            skills_table.add_row(f"{job['company']}: {job['title']}", skills_str)

        self.ui.console.print(skills_table)

        # Create a table for common skills
        all_required_skills = []
        for job in jobs:
            all_required_skills.append(set(job["data"].get("skills", {}).get("technical", {}).get("required", [])))

        common_skills = set.intersection(*all_required_skills) if all_required_skills else set()

        if common_skills:
            self.ui.print_section_header("Common Required Skills")
            common_table = self.ui.create_table("Skills Required by All Jobs", ["Skill"])

            for skill in common_skills:
                common_table.add_row(skill)

            self.ui.console.print(common_table)

        # Options
        self.ui.get_input("Press Enter to continue...")

    def resume_analysis_menu(self):
        """Menu for resume analysis and job matching."""
        self.ui.print_header()
        self.ui.print_section_header("Resume Analysis & Matching")

        # Check API key
        if not self.api_key:
            self.ui.print_error("API key not configured. Please set it in the settings menu.")
            self.ui.get_input("Press Enter to continue...")
            return

        # Options
        choice = self.ui.get_menu_choice("Select an option:", [
            "1. Analyze Resume (Paste Text)",
            "2. Analyze Resume (From File)",
            "3. Match Resume to Job",
            "4. Back to Main Menu"
        ])

        if not choice or choice.startswith("4"):
            return

        if choice.startswith("1"):
            self.analyze_resume_from_paste()
        elif choice.startswith("2"):
            self.analyze_resume_from_file()
        elif choice.startswith("3"):
            self.match_resume_to_job()

    def analyze_resume_from_paste(self):
        """Analyze a resume from pasted text."""
        self.ui.print_header()
        self.ui.print_section_header("Resume Analysis (Paste Text)")

        # Get resume text
        self.ui.print_info("Paste resume text below (press Ctrl+D or Ctrl+Z on a new line when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            resume_text = "\n".join(lines)

        if not resume_text:
            self.ui.print_warning("No resume text provided.")
            self.ui.get_input("Press Enter to continue...")
            return

        self._perform_resume_analysis(resume_text)


    def analyze_resume_from_file(self):
        """Analyze a resume from a file."""
        self.ui.print_header()
        self.ui.print_section_header("Resume Analysis (From File)")

        file_path = self.ui.get_input("Enter path to resume file:")
        if not file_path or not os.path.exists(file_path):
            self.ui.print_error(f"File not found: {file_path}")
            self.ui.get_input("Press Enter to continue...")
            return

        try:
            with open(file_path, 'r') as f:
                resume_text = f.read()

            if not resume_text:
                self.ui.print_warning(f"Empty file: {file_path}")
                self.ui.get_input("Press Enter to continue...")
                return

            self._perform_resume_analysis(resume_text)

        except Exception as e:
            self.ui.print_error(f"Error reading file {file_path}: {e}")
            self.ui.get_input("Press Enter to continue...")


    def _perform_resume_analysis(self, resume_text: str):
        """Performs the actual resume analysis using the provided text."""
        # Select model
        model = self.config.get("default_model")

        # Analyze resume
        self.ui.print_info(f"Analyzing resume using {model}...")

        with self.ui.progress_context("Analyzing resume...") as progress:
            task = progress.add_task("Processing...", total=100)
            progress.update(task, advance=50)

            resume_analysis = analyze_resume_text(self.api_key, model, resume_text, self.prompts_data)
            progress.update(task, advance=50)

            if not resume_analysis:
                self.ui.print_error("Resume analysis failed.")
                self.ui.get_input("Press Enter to continue...")
                return

        # Store the resume analysis
        self.current_resume_analysis = resume_analysis

        # Display results
        self.display_resume_analysis(resume_analysis)


    def display_resume_analysis(self, resume_analysis: Dict):
        """Display the results of a resume analysis."""
        self.ui.print_header()
        self.ui.print_section_header("Resume Analysis Results")

        # Extract key information
        contact = resume_analysis.get("contact", {})
        skills = resume_analysis.get("skills", {})

        # Create a layout for the summary
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Header content
        name = contact.get("name", "Unknown")
        summary = resume_analysis.get("professional_summary", "No summary available.")
        header_content = Panel(
            f"[bold cyan]{name}[/bold cyan]\n[yellow]{summary}[/yellow]",
            title="Resume Summary",
            border_style="blue"
        )

        # Left panel content - Contact details
        left_content = Panel(
            "\n".join([
                f"[bold]Email:[/bold] {contact.get('email', 'Not specified')}",
                f"[bold]Phone:[/bold] {contact.get('phone', 'Not specified')}",
                f"[bold]Location:[/bold] {contact.get('location', 'Not specified')}",
                f"[bold]Clearance:[/bold] {resume_analysis.get('clearance', 'Not specified')}"
            ]),
            title="Contact Details",
            border_style="green"
        )

        # Right panel content - Skills
        technical_skills = skills.get("technical", [])
        soft_skills = skills.get("soft_skills", [])

        skills_content = []
        if technical_skills:
            skills_content.append("[bold]Technical Skills:[/bold]")
            for skill in technical_skills[:7]:  # Show top 7 skills
                skills_content.append(f"• {skill}")

        if soft_skills:
            skills_content.append("\n[bold]Soft Skills:[/bold]")
            for skill in soft_skills[:3]:  # Show top 3 soft skills
                skills_content.append(f"• {skill}")

        right_content = Panel(
            "\n".join(skills_content),
            title="Skills",
            border_style="yellow"
        )

        # Assign content to layout
        layout["header"].update(header_content)
        layout["main"]["left"].update(left_content)
        layout["main"]["right"].update(right_content)

        # Print the layout
        self.console.print(layout)

        # Display experience
        experience = resume_analysis.get("experience", [])
        if experience:
            self.ui.print_section_header("Experience")
            for exp in experience:
                exp_panel = Panel(
                    f"[bold]{exp.get('title', 'Unknown Position')}[/bold] at {exp.get('company', 'Unknown Company')}\n"
                    f"[italic]{exp.get('location', '')} • {exp.get('dates', '')}[/italic]\n\n"
                    + "\n".join([f"• {highlight}" for highlight in exp.get("highlights", [])[:3]]),
                    border_style="cyan"
                )
                self.ui.console.print(exp_panel)

        # Options
        self.ui.get_input("Press Enter to continue...")

    def match_resume_to_job(self):
        """Match a resume to a job posting."""
        self.ui.print_header()
        self.ui.print_section_header("Resume-Job Matching")

        # Check if we have both a resume and job analysis
        if not self.current_resume_analysis:
            self.ui.print_warning("No resume analysis available. Please analyze a resume first.")
            self.ui.get_input("Press Enter to continue...")
            return

        # If no current job analysis, let the user select one from history
        if not self.current_job_analysis:
            history_items = self.history_manager.get_history_table()

            if not history_items:
                self.ui.print_warning("No job analyses in history. Please analyze a job posting first.")
                self.ui.get_input("Press Enter to continue...")
                return

            # Display available jobs
            table = self.ui.create_table("Select a Job", ["ID", "Title", "Company"])

            for i, item in enumerate(history_items):
                table.add_row(
                    str(i + 1),
                    item["title"],
                    item["company"]
                )

            self.ui.console.print(table)

            # Select a job
            index = self.ui.get_input("Enter the ID number of the job to match with (or 0 to cancel):")
            try:
                index = int(index) - 1
                if index < 0:
                    return

                if index < len(history_items):
                    analysis_id = history_items[index]["id"]
                    job_analysis = self.history_manager.get_job_analysis(analysis_id)

                    if job_analysis:
                        self.current_job_analysis = job_analysis
                    else:
                        self.ui.print_error(f"Could not load job analysis with ID {analysis_id}")
                        self.ui.get_input("Press Enter to continue...")
                        return
                else:
                    self.ui.print_error("Invalid ID number.")
                    self.ui.get_input("Press Enter to continue...")
                    return
            except ValueError:
                self.ui.print_error("Please enter a valid number.")
                self.ui.get_input("Press Enter to continue...")
                return

        # Perform the match analysis
        self.ui.print_info("Matching resume to job posting...")

        # Select model
        model = self.config.get("default_model")

        with self.ui.progress_context("Analyzing match...") as progress:
            task = progress.add_task("Processing...", total=100)
            progress.update(task, advance=50)

            match_result = analyze_match(
                self.api_key,
                model,
                self.current_job_analysis,
                self.current_resume_analysis,
                self.prompts_data
            )
            progress.update(task, advance=50)

            if not match_result:
                self.ui.print_error("Match analysis failed.")
                self.ui.get_input("Press Enter to continue...")
                return

        # Display match results
        self.display_match_results(match_result)

    def display_match_results(self, match_result: Dict):
        """Display the results of a resume-job match analysis."""
        self.ui.print_header()
        self.ui.print_section_header("Match Results")

        # Extract key information
        overall_score = match_result.get("overall_match_score", 0)
        detailed_scores = match_result.get("detailed_scores", {})
        strengths = match_result.get("strengths", [])
        gaps = match_result.get("gaps", [])

        # Display overall score
        score_panel = Panel(
            f"[bold]{overall_score}%[/bold]",
            title="Overall Match Score",
            border_style="green" if overall_score >= 80 else "yellow" if overall_score >= 60 else "red"
        )
        self.ui.console.print(score_panel)

        # Display detailed scores
        if detailed_scores:
            self.ui.print_section_header("Detailed Scores")
            scores_table = self.ui.create_table("Category Scores", ["Category", "Score"])

            for category, score in detailed_scores.items():
                category_name = category.replace("_", " ").title()
                scores_table.add_row(category_name, f"{score}%")

            self.ui.console.print(scores_table)

        # Display strengths and gaps
        columns = Columns([
            Panel(
                "\n".join([f"• {strength}" for strength in strengths]),
                title="Strengths",
                border_style="green"
            ),
            Panel(
                "\n".join([f"• {gap}" for gap in gaps]),
                title="Gaps",
                border_style="red"
            )
        ])
        self.ui.console.print(columns)

        # Display recommendations
        recommendations = match_result.get("recommendations", "No recommendations available.")
        self.ui.print_section_header("Recommendations")
        self.ui.print_markdown(recommendations)

        # Options
        self.ui.get_input("Press Enter to continue...")

    def batch_process_menu(self):
        """Menu for batch processing multiple job postings."""
        self.ui.print_header()
        self.ui.print_section_header("Batch Process Jobs")

        # Check API key
        if not self.api_key:
            self.ui.print_error("API key not configured. Please set it in the settings menu.")
            self.ui.get_input("Press Enter to continue...")
            return

        # Options for batch processing
        choice = self.ui.get_menu_choice("Select input method:", [
            "1. Process multiple URLs (from file)",
            "2. Process multiple job descriptions (from directory)",
            "3. Process jobs from XML file",
            "4. Back to main menu"
        ])

        if not choice or choice.startswith("4"):
            return

        if choice.startswith("1"):
            # Process URLs from file
            file_path = self.ui.get_input("Enter path to file containing URLs (one per line):")
            if not file_path or not os.path.exists(file_path):
                self.ui.print_error(f"File not found: {file_path}")
                self.ui.get_input("Press Enter to continue...")
                return

            try:
                with open(file_path, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]

                if not urls:
                    self.ui.print_warning("No URLs found in the file.")
                    self.ui.get_input("Press Enter to continue...")
                    return

                self.ui.print_info(f"Found {len(urls)} URLs to process.")

                # Confirm processing
                if not self.ui.get_confirmation(f"Process {len(urls)} job postings?"):
                    return

                # Process each URL
                model = self.config.get("default_model")
                results = []

                with self.ui.progress_context(f"Processing {len(urls)} job postings...") as progress:
                    task = progress.add_task("Processing...", total=len(urls))

                    for i, url in enumerate(urls):
                        progress.update(task, description=f"Processing URL {i+1}/{len(urls)}")

                        # Fetch and clean job content
                        job_text = fetch_and_clean_url_content(url)
                        if not job_text:
                            self.ui.print_error(f"Could not fetch or clean content from URL: {url}")
                            continue

                        # Analyze job posting
                        analysis_result = analyze_job_posting(self.api_key, model, job_text, self.prompts_data)
                        if not analysis_result:
                            self.ui.print_error(f"Analysis failed for URL: {url}")
                            continue

                        # Add URL to the result
                        analysis_result["url"] = url

                        # Add to history
                        history_entry = self.history_manager.add_job_analysis(analysis_result)
                        results.append({
                            "url": url,
                            "analysis": analysis_result,
                            "history_entry": history_entry
                        })

                        progress.update(task, advance=1)

                # Display summary
                self.ui.print_header()
                self.ui.print_section_header("Batch Processing Results")

                table = self.ui.create_table("Processing Summary", ["URL", "Title", "Company", "Status"])

                for result in results:
                    analysis = result["analysis"]
                    extraction = analysis.get("extraction", {})
                    title = extraction.get("title", "Unknown")
                    company = extraction.get("company", "Unknown")

                    table.add_row(
                        result["url"],
                        title,
                        company,
                        "[green]Success[/green]"
                    )

                self.ui.console.print(table)
                self.ui.get_input("Press Enter to continue...")

            except Exception as e:
                self.ui.print_error(f"Error processing batch: {e}")
                self.ui.get_input("Press Enter to continue...")

        elif choice.startswith("2"):
            # Process job descriptions from directory
            dir_path = self.ui.get_input("Enter path to directory containing job description files:")
            if not dir_path or not os.path.isdir(dir_path):
                self.ui.print_error(f"Directory not found: {dir_path}")
                self.ui.get_input("Press Enter to continue...")
                return

            try:
                # Get all text files in the directory
                files = [f for f in os.listdir(dir_path) if f.endswith('.txt') and os.path.isfile(os.path.join(dir_path, f))]

                if not files:
                    self.ui.print_warning("No text files found in the directory.")
                    self.ui.get_input("Press Enter to continue...")
                    return

                self.ui.print_info(f"Found {len(files)} job description files to process.")

                # Confirm processing
                if not self.ui.get_confirmation(f"Process {len(files)} job descriptions?"):
                    return

                # Process each file
                model = self.config.get("default_model")
                results = []

                with self.ui.progress_context(f"Processing {len(files)} job descriptions...") as progress:
                    task = progress.add_task("Processing...", total=len(files))

                    for i, file_name in enumerate(files):
                        file_path = os.path.join(dir_path, file_name)
                        progress.update(task, description=f"Processing file {i+1}/{len(files)}")

                        # Read job description from file
                        try:
                            with open(file_path, 'r') as f:
                                job_text = f.read()

                            if not job_text:
                                self.ui.print_error(f"Empty file: {file_path}")
                                continue

                            # Analyze job posting
                            analysis_result = analyze_job_posting(self.api_key, model, job_text, self.prompts_data)
                            if not analysis_result:
                                self.ui.print_error(f"Analysis failed for file: {file_path}")
                                continue

                            # Add file path to the result
                            analysis_result["file_path"] = file_path

                            # Add to history
                            history_entry = self.history_manager.add_job_analysis(analysis_result)
                            results.append({
                                "file_path": file_path,
                                "analysis": analysis_result,
                                "history_entry": history_entry
                            })

                        except Exception as e:
                            self.ui.print_error(f"Error processing file {file_path}: {e}")

                        progress.update(task, advance=1)

                # Display summary
                self.ui.print_header()
                self.ui.print_section_header("Batch Processing Results")

                table = self.ui.create_table("Processing Summary", ["File", "Title", "Company", "Status"])

                for result in results:
                    analysis = result["analysis"]
                    extraction = analysis.get("extraction", {})
                    title = extraction.get("title", "Unknown")
                    company = extraction.get("company", "Unknown")

                    table.add_row(
                        os.path.basename(result["file_path"]),
                        title,
                        company,
                        "[green]Success[/green]"
                    )

                self.ui.console.print(table)
                self.ui.get_input("Press Enter to continue...")

            except Exception as e:
                self.ui.print_error(f"Error processing batch: {e}")
                self.ui.get_input("Press Enter to continue...")

        elif choice.startswith("3"):
            # Process jobs from XML file
            xml_file_path = self.ui.get_input("Enter path to XML file containing job postings:")
            if not xml_file_path or not os.path.exists(xml_file_path):
                self.ui.print_error(f"File not found: {xml_file_path}")
                self.ui.get_input("Press Enter to continue...")
                return

            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(xml_file_path)
                root = tree.getroot()

                # Assuming XML structure: <jobs><job><description>...</description></job></jobs>
                job_elements = root.findall('.//job')

                if not job_elements:
                    self.ui.print_warning("No <job> elements found in the XML file.")
                    self.ui.get_input("Press Enter to continue...")
                    return

                self.ui.print_info(f"Found {len(job_elements)} job postings in the XML file.")

                # Confirm processing
                if not self.ui.get_confirmation(f"Process {len(job_elements)} job postings?"):
                    return

                # Process each job element
                model = self.config.get("default_model")
                results = []

                with self.ui.progress_context(f"Processing {len(job_elements)} job postings from XML...") as progress:
                    task = progress.add_task("Processing...", total=len(job_elements))

                    for i, job_element in enumerate(job_elements):
                        progress.update(task, description=f"Processing job {i+1}/{len(job_elements)}")

                        description_element = job_element.find('description')
                        job_text = description_element.text.strip() if description_element is not None and description_element.text else ""

                        if not job_text:
                            self.ui.print_warning(f"Empty description for job {i+1} in XML file.")
                            continue

                        # Analyze job posting
                        analysis_result = analyze_job_posting(self.api_key, model, job_text, self.prompts_data)
                        if not analysis_result:
                            self.ui.print_error(f"Analysis failed for job {i+1} in XML file.")
                            continue

                        # Add XML file path and job index to the result
                        analysis_result["source_file"] = xml_file_path
                        analysis_result["source_index"] = i

                        # Add to history
                        history_entry = self.history_manager.add_job_analysis(analysis_result)
                        results.append({
                            "source_file": xml_file_path,
                            "source_index": i,
                            "analysis": analysis_result,
                            "history_entry": history_entry
                        })

                        progress.update(task, advance=1)

                # Display summary
                self.ui.print_header()
                self.ui.print_section_header("Batch Processing Results (XML)")

                table = self.ui.create_table("Processing Summary", ["Source File", "Index", "Title", "Company", "Status"])

                for result in results:
                    analysis = result["analysis"]
                    extraction = analysis.get("extraction", {})
                    title = extraction.get("title", "Unknown")
                    company = extraction.get("company", "Unknown")

                    table.add_row(
                        os.path.basename(result["source_file"]),
                        str(result["source_index"]),
                        title,
                        company,
                        "[green]Success[/green]"
                    )

                self.ui.console.print(table)
                self.ui.get_input("Press Enter to continue...")

            except FileNotFoundError:
                 self.ui.print_error(f"XML file not found: {xml_file_path}")
                 self.ui.get_input("Press Enter to continue...")
            except ET.ParseError as e:
                 self.ui.print_error(f"Error parsing XML file {xml_file_path}: {e}")
                 self.ui.get_input("Press Enter to continue...")
            except Exception as e:
                self.ui.print_error(f"Error processing batch from XML: {e}")
                self.ui.get_input("Press Enter to continue...")


    def settings_menu(self):
        """Menu for configuring application settings."""
        while True:
            self.ui.print_header()
            self.ui.print_section_header("Settings")

            # Display current settings
            table = self.ui.create_table("Current Settings", ["Setting", "Value"])

            for key, value in self.config.config.items():
                # Don't show API key directly
                if key == "api_key":
                    if value:
                        value = "********" + value[-4:] if len(value) > 4 else "********"
                    else:
                        value = "Not set"

                table.add_row(key, str(value))

            self.ui.console.print(table)

            # Options
            choice = self.ui.get_menu_choice("Select an option:", [
                "1. Set API Key",
                "2. Change Default Model",
                "3. Change Output Directory",
                "4. Set Maximum History Items",
                "5. Toggle Progress Bars",
                "6. Set Default Export Format",
                "7. Back to Main Menu"
            ])

            if not choice or choice.startswith("7"):
                break

            if choice.startswith("1"):
                # Set API key
                api_key = self.ui.get_input("Enter OpenRouter API Key:")
                if api_key:
                    self.config.set("api_key", api_key)
                    self.api_key = api_key
                    self.ui.print_success("API key updated.")
                else:
                    self.ui.print_warning("API key not changed.")

            elif choice.startswith("2"):
                # Change default model
                model = self.ui.get_input("Enter default model (e.g., google/gemini-2.5-flash-preview:online):",
                                         self.config.get("default_model"))
                if model:
                    self.config.set("default_model", model)
                    self.ui.print_success(f"Default model changed to: {model}")

            elif choice.startswith("3"):
                # Change output directory
                output_dir = self.ui.get_input("Enter output directory:", self.config.get("output_directory"))
                if output_dir:
                    self.config.set("output_directory", output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    self.ui.print_success(f"Output directory changed to: {output_dir}")

            elif choice.startswith("4"):
                # Set maximum history items
                try:
                    max_items = int(self.ui.get_input("Enter maximum history items:", str(self.config.get("max_history_items"))))
                    if max_items > 0:
                        self.config.set("max_history_items", max_items)
                        self.ui.print_success(f"Maximum history items set to: {max_items}")
                    else:
                        self.ui.print_error("Please enter a positive number.")
                except ValueError:
                    self.ui.print_error("Please enter a valid number.")

            elif choice.startswith("5"):
                # Toggle progress bars
                current = self.config.get("show_progress_bars", True)
                new_value = not current
                self.config.set("show_progress_bars", new_value)
                self.ui.print_success(f"Progress bars {'enabled' if new_value else 'disabled'}.")

            elif choice.startswith("6"):
                # Set default export format
                format_choice = self.ui.get_menu_choice("Select default export format:", [
                    "1. JSON",
                    "2. Markdown",
                    "3. PDF",
                    "4. All Formats"
                ])

                if format_choice:
                    format_map = {
                        "1": "json",
                        "2": "markdown",
                        "3": "pdf",
                        "4": "all"
                    }

                    for prefix, format_value in format_map.items():
                        if format_choice.startswith(prefix):
                            self.config.set("default_format", format_value)
                            self.ui.print_success(f"Default export format set to: {format_value}")
                            break

            self.ui.get_input("Press Enter to continue...")

    def help_menu(self):
        """Display help information and documentation."""
        while True:
            self.ui.print_header()
            self.ui.print_section_header("Help & Documentation")

            # Help menu options
            choice = self.ui.get_menu_choice("Select a topic:", [
                "1. Getting Started",
                "2. Analyzing Job Postings",
                "3. Resume Analysis & Matching",
                "4. Batch Processing",
                "5. Command Reference",
                "6. About",
                "7. Back to Main Menu"
            ])

            if not choice or choice.startswith("7"):
                break

            if choice.startswith("1"):
                # Getting Started
                self.ui.print_header()
                self.ui.print_section_header("Getting Started")

                help_text = """
# Getting Started with Job Insight Analyzer

Job Insight Analyzer is a powerful tool for analyzing job postings, resumes, and matching candidates to positions.

## Initial Setup

1. **API Key**: You need an OpenRouter API key to use the analysis features. Set this in the Settings menu.
2. **Output Directory**: By default, analysis results are saved to the `job_analysis_results` directory.
3. **Configuration**: You can customize various settings through the Settings menu.

## Basic Workflow

1. **Analyze a Job**: Start by analyzing a job posting via URL or pasted text.
2. **Review Analysis**: Examine the extracted information, skills, and comprehensive report.
3. **Export Results**: Save the analysis in various formats (JSON, Markdown, PDF).
4. **Analyze Resume**: Analyze a resume to extract skills and experience.
5. **Match Resume to Job**: Compare a resume against a job posting for compatibility.

## Tips

- Use the history feature to recall previously analyzed jobs.
- Compare multiple jobs to identify common requirements.
- Batch process multiple job postings for efficient analysis.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("2"):
                # Analyzing Job Postings
                self.ui.print_header()
                self.ui.print_section_header("Analyzing Job Postings")

                help_text = """
# Analyzing Job Postings

Job Insight Analyzer provides comprehensive analysis of job postings, extracting key information and generating detailed reports.

## Analysis Methods

1. **URL Analysis**: Enter a job posting URL, and the tool will fetch and analyze the content.
2. **Text Analysis**: Paste job description text directly for analysis.

## Analysis Components

- **Basic Extraction**: Job title, company, location, work model, experience requirements, etc.
- **Skills Analysis**: Technical skills (required and preferred) and soft skills.
- **Tools & Technologies**: Specific tools and platforms mentioned in the posting.
- **Comprehensive Report**: Detailed analysis with sections on company profile, position analysis, compensation, location, and opportunity assessment.

## Viewing Results

- **Summary View**: Quick overview of the job posting.
- **Technical Skills**: Detailed breakdown of required and preferred skills.
- **JSON Data**: Raw structured data from the analysis.
- **Comprehensive Report**: Formatted markdown report with detailed insights.

## Exporting Results

- **JSON**: Structured data format for programmatic use.
- **Markdown**: Formatted text report for human reading.
- **PDF**: Professional document format for sharing and printing.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("3"):
                # Resume Analysis & Matching
                self.ui.print_header()
                self.ui.print_section_header("Resume Analysis & Matching")

                help_text = """
# Resume Analysis & Matching

Job Insight Analyzer can analyze resumes and match them against job postings to assess compatibility.

## Resume Analysis

1. **Input**: Paste resume text for analysis.
2. **Extraction**: The tool extracts contact information, skills, experience, education, and certifications.
3. **Results**: View a structured breakdown of the resume content.

## Job-Resume Matching

1. **Select Job**: Choose a previously analyzed job posting or analyze a new one.
2. **Select Resume**: Use a previously analyzed resume or analyze a new one.
3. **Match Analysis**: The tool compares the resume against the job requirements.
4. **Match Score**: Receive an overall match score and detailed category scores.
5. **Strengths & Gaps**: Identify areas where the candidate excels and areas for improvement.
6. **Recommendations**: Get actionable suggestions for improving the match.

## Tips for Better Matching

- Ensure the resume contains relevant keywords from the job posting.
- Include specific technical skills mentioned in the job requirements.
- Quantify achievements and experience to better match job expectations.
- Tailor the resume to highlight experience relevant to the specific job.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("4"):
                # Batch Processing
                self.ui.print_header()
                self.ui.print_section_header("Batch Processing")

                help_text = """
# Batch Processing

Job Insight Analyzer supports batch processing of multiple job postings, saving you time and effort.

## Input Methods

1. **URLs from File**: Provide a text file containing a list of job posting URLs, one per line.
2. **Job Descriptions from Directory**: Provide a directory containing text files, where each file contains a job description.

## Process

- The tool will iterate through each URL or file in the batch.
- Each job posting will be analyzed individually using the configured model.
- Analysis results will be saved to the history and the output directory.
- A summary table will be displayed upon completion of the batch.

## Tips

- Ensure your input files/directories are correctly formatted.
- Batch processing can take time depending on the number of jobs and the model used.
- Review the batch summary to quickly see which jobs were processed successfully.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("5"):
                # Command Reference
                self.ui.print_header()
                self.ui.print_section_header("Command Reference")

                help_text = """
# Command Reference

The Job Insight Analyzer CLI is primarily menu-driven, but here's a quick reference to the main sections:

- **Analyze Job Posting**: Start a new analysis of a single job.
- **View Job History**: Browse and manage previously analyzed jobs.
- **Compare Jobs**: Compare multiple jobs from your history.
- **Resume Analysis & Matching**: Analyze a resume and match it against a job.
- **Batch Process Jobs**: Analyze multiple jobs from a file or directory.
- **Settings**: Configure API key, default model, output directory, etc.
- **Help**: Access this help system.
- **Exit**: Close the application.

Each menu option will guide you through the necessary steps.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

            elif choice.startswith("6"):
                # About
                self.ui.print_header()
                self.ui.print_section_header("About Job Insight Analyzer")

                help_text = f"""
# About Job Insight Analyzer

**Version:** {VERSION}

Job Insight Analyzer is a command-line tool designed to assist job seekers and recruiters by providing AI-powered analysis of job postings and resumes.

It leverages large language models (via OpenRouter) to extract key information, identify skills, compare jobs, and assess resume compatibility.

**Developed by:** Your Name/Organization (Optional)
**License:** MIT License (See README.md)

This tool is intended to provide insights and should not be used as the sole basis for job application or hiring decisions.
                """

                self.ui.print_markdown(help_text)
                self.ui.get_input("Press Enter to continue...")

# Main execution block
if __name__ == "__main__":
    cli = JobAnalyzerCLI()
    sys.exit(cli.run())