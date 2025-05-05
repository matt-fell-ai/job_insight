import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import altair as alt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import json
import os
import re
import logging
from datetime import datetime
import io
from pathlib import Path
import uuid
import base64
import pyperclip
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem
from reportlab.lib.units import inch
from typing import List, Dict, Optional, Tuple, Union, Any
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Setup ---

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions ---

def load_prompts(filepath: str = "prompts.json") -> Dict[str, Dict]:
    """
    Loads prompts from the specified JSON file (expected to be a list of dicts)
    and converts it into a dictionary keyed by the 'key' field.

    Args:
        filepath: The path to the JSON file containing prompts.

    Returns:
        A dictionary where keys are prompt identifiers (from the 'key' field)
        and values are the corresponding prompt dictionaries.
        Returns an empty dictionary if the file is not found, invalid, or not a list.
    """
    prompts_dict = {}
    try:
        with open(filepath, 'r') as f:
            prompts_list = json.load(f)

        if not isinstance(prompts_list, list):
            logger.error(f"Error: Prompts file at {filepath} does not contain a JSON list.")
            return {}

        for prompt_item in prompts_list:
            if isinstance(prompt_item, dict) and "key" in prompt_item:
                prompts_dict[prompt_item["key"]] = prompt_item
            else:
                logger.warning(f"Skipping invalid prompt item in {filepath}: {prompt_item}")

        logger.info(f"Successfully loaded and processed {len(prompts_dict)} prompts from {filepath}")
        return prompts_dict
    except FileNotFoundError:
        logger.error(f"Error: Prompts file not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading prompts: {e}")
        return {}

def get_api_key() -> Optional[str]:
    """
    Retrieves the OpenRouter API key from the environment variable.

    Returns:
        The API key as a string, or None if not found.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY environment variable not set.")
        # In a real app, you might prompt the user here or raise an error
    return api_key

def _configure_openai_client(api_key: str) -> openai.OpenAI:
    """
    Configures and returns an OpenAI client instance pointed at the OpenRouter API.

    This is a helper function used internally by other functions that interact
    with the OpenRouter API.

    Args:
        api_key: The OpenRouter API key.

    Returns:
        An configured instance of openai.OpenAI.
    """
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def fetch_available_models(api_key: str) -> List[str]:
    """
    Queries OpenRouter for available models using the openai client.

    Args:
        api_key: The OpenRouter API key.

    Returns:
        A list of available model IDs, or an empty list on error.
    """
    if not api_key:
        logger.error("API key is required to fetch models.")
        return []
    try:
        client = _configure_openai_client(api_key)
        models_response = client.models.list()
        model_ids = [model.id for model in models_response.data]
        logger.info(f"Fetched {len(model_ids)} available models from OpenRouter.")
        return model_ids
    except openai.APIConnectionError as e:
        logger.error(f"OpenRouter API Connection Error: {e}")
        return []
    except openai.RateLimitError as e:
        logger.error(f"OpenRouter Rate Limit Error: {e}")
        return []
    except openai.AuthenticationError as e:
        logger.error(f"OpenRouter Authentication Error: Invalid API Key? {e}")
        return []
    except openai.APIStatusError as e:
        logger.error(f"OpenRouter API Status Error: {e.status_code} - {e.response}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching models: {e}")
        return []

def fetch_and_clean_url_content(url: str) -> Optional[str]:
    """
    Fetches and cleans text content from a URL using requests and BeautifulSoup.

    Args:
        url: The URL to fetch content from.

    Returns:
        The cleaned text content as a string, or None on error.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Use BeautifulSoup to parse and extract text
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, strip leading/trailing whitespace, and reduce multiple newlines
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

        logger.info(f"Successfully fetched and cleaned content from {url}")
        return cleaned_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred cleaning content from {url}: {e}")
        return None

def _clean_html(raw_html: Optional[str]) -> str:
    """
    Helper function to clean HTML content using BeautifulSoup.

    Removes HTML tags and returns plain text.

    Args:
        raw_html: The raw HTML string, or None.

    Returns:
        The cleaned plain text string. Returns an empty string if input is None
        or cleaning fails.
    """
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.warning(f"Could not clean HTML snippet: {e}")
        # Fallback: return the raw text if cleaning fails
        return str(raw_html)


def process_xml_batch(xml_source: Union[str, io.BytesIO]) -> List[Dict[str, str]]:
    """
    Parses job entries from an XML file or file-like object using ElementTree.

    Assumes a structure like <jobs><job><title>...</title><description>...</description>...</job></jobs>
    Adjust tags ('job', 'title', 'description', 'url') as needed for the actual XML structure.

    Args:
        xml_source: The path to the XML file (str) or a file-like object (io.BytesIO).

    Returns:
        A list of dictionaries, each representing a job posting with cleaned fields.
        Returns an empty list on error or if the source is invalid.
    """
    jobs = []
    try:
        tree = ET.parse(xml_source)
        root = tree.getroot()

        # Adjust these tags based on the actual XML structure
        job_tag = 'job'
        title_tag = 'title'
        description_tag = 'description'
        url_tag = 'url' # Optional: Add other relevant tags like 'company', 'location', etc.
        # company_tag = 'company'
        # location_tag = 'location'

        for job_elem in root.findall(f'.//{job_tag}'):
            job_data = {}
            title = job_elem.findtext(title_tag, default='').strip()
            description_html = job_elem.findtext(description_tag, default='')
            url = job_elem.findtext(url_tag, default='').strip()
            # company = job_elem.findtext(company_tag, default='').strip()
            # location = job_elem.findtext(location_tag, default='').strip()

            # Clean HTML from description
            description_clean = _clean_html(description_html)

            if title and description_clean: # Only add if essential fields are present
                job_data['title'] = title
                job_data['description'] = description_clean
                if url: job_data['url'] = url
                # if company: job_data['company'] = company
                # if location: job_data['location'] = location
                jobs.append(job_data)
            else:
                logger.warning(f"Skipping job entry due to missing title or description in {xml_source}")

        logger.info(f"Successfully processed {len(jobs)} job entries from {xml_source}")
        return jobs

    except FileNotFoundError:
        logger.error(f"Error: XML file not found at {xml_source}")
        return []
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {xml_source}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred processing XML {xml_source}: {e}")
        return []

def _extract_json_from_markdown(markdown_text: str) -> Optional[Dict | str]:
    """
    Extracts a JSON object from a markdown code block within the given text.

    Looks for content within ```json ... ``` fences. If not found, attempts
    to parse the entire text as JSON as a fallback.

    Args:
        markdown_text: The text potentially containing a JSON markdown block.

    Returns:
        The parsed JSON dictionary if successful.
        The raw JSON string if extraction succeeds but parsing fails.
        None if no JSON markdown block is found and the entire text is not valid JSON.
    """
    # Regex to find JSON within ```json ... ``` blocks
    match = re.search(r'```json\s*(\{.*?\})\s*```', markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON extracted from markdown: {e}. Raw string: {json_str}")
            return json_str # Return raw string if parsing fails

    # Fallback: Try to find JSON without markdown fences if the whole response might be JSON
    try:
        # Attempt to parse the entire string if no markdown block found
        potential_json = json.loads(markdown_text)
        if isinstance(potential_json, dict):
            logger.info("Parsed response directly as JSON (no markdown fences found).")
            return potential_json
    except json.JSONDecodeError:
        # If direct parsing fails, it's likely not just JSON
        pass

    logger.warning("Could not find or parse JSON object within markdown code block.")
    return None # Return None if no JSON found or parsed


def run_ai_analysis(
    api_key: str,
    model: str,
    prompt_key: str,
    context_text: str,
    prompts_data: Dict[str, str],
    expected_format: str = "json" # "json" or "text"
) -> Optional[Union[Dict, str]]:
    """
    Core LLM interaction function using the OpenAI client pointed at OpenRouter.

    Args:
        api_key: The OpenRouter API key.
        model: The model ID to use (e.g., 'openai/gpt-4o').
        prompt_key: The key for the desired prompt in prompts_data.
        context_text: The text context (job description, resume, etc.) to analyze.
        prompts_data: Dictionary containing the loaded prompts.
        expected_format: The expected format of the response ('json' or 'text').

    Returns:
        A dictionary if expected_format is 'json' and parsing succeeds.
        A string if expected_format is 'text'.
        None on error or if the prompt key is invalid.
    """
    if not api_key:
        logger.error("API key is required for AI analysis.")
        return None
    if not model:
        logger.error("Model is required for AI analysis.")
        return None
    if prompt_key not in prompts_data:
        logger.error(f"Prompt key '{prompt_key}' not found in prompts data.")
        return None

    prompt_details = prompts_data.get(prompt_key)
    if not prompt_details or "prompt_text" not in prompt_details:
        logger.error(f"Prompt details or prompt_text not found for key '{prompt_key}'.")
        return None

    prompt_text = prompt_details["prompt_text"]

    try:
        client = _configure_openai_client(api_key)
        logger.info(f"Running AI analysis with model '{model}' and prompt key '{prompt_key}'...")

        # Construct messages: Use the prompt text as the primary instruction,
        # and the context text as the data to analyze.
        messages = [
            {"role": "system", "content": "You are an expert HR analyst. Provide responses in the requested format."},
            {"role": "user", "content": f"{prompt_text}\n\nData to analyze:\n{context_text}"}
        ]

        # If the prompt itself is very long, consider putting it in the system message
        # or splitting it, but for now, combining in the user message is simpler.

        completion = client.chat.completions.create(
            model=model, # Use the model parameter
            messages=messages,
            # max_tokens=... # Consider setting a max_tokens based on expected output size
            temperature=0.5, # Adjust for creativity vs. determinism
            # response_format={"type": "json_object"} if expected_format == "json" else None # Use if model supports it reliably
        )

        response_content = completion.choices[0].message.content

        if not response_content:
            logger.warning("Received empty response content from AI model.")
            return None

        logger.info(f"Successfully received response for prompt '{prompt_key}'.")

        if expected_format == "json":
            # Try extracting JSON from markdown or direct parsing
            parsed_json = _extract_json_from_markdown(response_content)
            if isinstance(parsed_json, dict):
                return parsed_json
            else:
                # If parsing failed but we expected JSON, log warning and return raw content
                logger.warning(f"Expected JSON but failed to parse. Returning raw response for prompt '{prompt_key}'.")
                return response_content # Return raw string as fallback
        else: # expected_format == "text"
            return response_content.strip()

    except openai.APIConnectionError as e:
        logger.error(f"OpenRouter API Connection Error during analysis: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"OpenRouter Rate Limit Error during analysis: {e}")
        return None
    except openai.AuthenticationError as e:
        logger.error(f"OpenRouter Authentication Error during analysis: {e}")
        return None
    except openai.APITimeoutError as e:
        logger.error(f"OpenRouter API Timeout Error during analysis: {e}")
        return None
    except openai.APIStatusError as e:
        logger.error(f"OpenRouter API Status Error during analysis: {e.status_code} - {e.response}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during AI analysis: {e}")
        return None


def analyze_job_posting(api_key: str, model: str, job_text: str, prompts_data: Dict[str, str]) -> Optional[Dict]:
    """
    Orchestrates job analysis using multiple relevant prompts via run_ai_analysis
    and combines the results.

    Args:
        api_key: OpenRouter API key.
        model: Model ID to use.
        job_text: The text of the job posting.
        prompts_data: Dictionary of loaded prompts.

    Returns:
        A dictionary containing the combined structured analysis results,
        or a dictionary with an 'error' key if critical parts fail.
    """
    logger.info("Starting comprehensive job posting analysis...")
    combined_results = {}
    errors = []

    # Define the keys for the different analysis aspects
    analysis_keys = {
        "extraction": "initial_extraction_json",
        "skills": "skills_analysis_json",
        "tools": "tools_technologies_json",
        "summary": "job_summary_markdown", # Markdown format
        # Add other relevant keys like 'company_analysis_json', 'location_analysis_json' if desired
    }

    for aspect, prompt_key in analysis_keys.items():
        # Find the expected format from the prompts_data based on the key
        prompt_details = next((p for p in prompts_data.values() if p.get("key") == prompt_key), None)
        expected_format = "text" # Default to text
        if prompt_details and "expected_format" in prompt_details:
            expected_format = prompt_details["expected_format"]

        logger.info(f"Running analysis for: {aspect} (using key: {prompt_key}, expecting: {expected_format})")
        result = run_ai_analysis(api_key, model, prompt_key, job_text, prompts_data, expected_format=expected_format)

        if result is not None:
            # Store the result under the aspect key
            combined_results[aspect] = result
            logger.info(f"Successfully completed analysis for: {aspect}")
        else:
            error_msg = f"Analysis failed for aspect: {aspect} (prompt key: {prompt_key})"
            logger.error(error_msg)
            errors.append(error_msg)
            combined_results[aspect] = {"error": f"Failed to get result for {aspect}"} # Add error marker

    if not combined_results or errors:
         # If there were errors, include them in the final dict
         if errors:
             combined_results["analysis_errors"] = errors
         logger.warning(f"Job posting analysis completed with errors: {errors}")
         # Decide if returning partial results is okay or if it should be None
         # Returning partial results for now
         return combined_results

    logger.info("Successfully completed comprehensive job posting analysis.")
    return combined_results


def analyze_resume_text(api_key: str, model: str, resume_text: str, prompts_data: Dict[str, str]) -> Optional[Dict]:
    """
    Orchestrates resume analysis using run_ai_analysis.

    Args:
        api_key: OpenRouter API key.
        model: Model ID to use.
        resume_text: The text content of the resume.
        prompts_data: Dictionary of loaded prompts.

    Returns:
        A dictionary containing the structured analysis results, or None on error.
    """
    logger.info("Starting resume analysis...")
    # Adjust prompt_key based on your prompts.json structure
    resume_analysis_prompt_key = "resume_analysis_json"  # Corrected key based on prompts.json
    result = run_ai_analysis(api_key, model, resume_analysis_prompt_key, resume_text, prompts_data, expected_format="json")

    if isinstance(result, dict):
        logger.info("Successfully analyzed resume.")
        return result
    else:
        logger.error("Failed to get structured JSON result for resume analysis.")
        return {"error": "Failed to parse resume analysis", "raw_response": result} if result else None


def analyze_match(
    api_key: str,
    model: str,
    job_analysis_result: Dict,
    resume_analysis_result: Dict,
    prompts_data: Dict[str, str]
) -> Optional[Dict]:
    """
    Orchestrates match analysis between job and resume using run_ai_analysis.

    Args:
        api_key: OpenRouter API key.
        model: Model ID to use.
        job_analysis_result: The structured analysis result of the job posting.
        resume_analysis_result: The structured analysis result of the resume.
        prompts_data: Dictionary of loaded prompts.

    Returns:
        A dictionary containing the structured match analysis results, or None on error.
    """
    logger.info("Starting job-resume match analysis...")
    # Combine job and resume analysis into a context string for the match prompt
    # Ensure sensitive data is handled appropriately if needed
    try:
        context_dict = {
            "job_details": job_analysis_result,
            "resume_details": resume_analysis_result
        }
        context_text = json.dumps(context_dict, indent=2)
    except Exception as e:
        logger.error(f"Error formatting context for match analysis: {e}")
        return None

    # Adjust prompt_key based on your prompts.json structure
    match_analysis_prompt_key = "analyze_match"
    result = run_ai_analysis(api_key, model, match_analysis_prompt_key, context_text, prompts_data, expected_format="json")

    if isinstance(result, dict):
        logger.info("Successfully performed match analysis.")
        return result
    else:
        logger.error("Failed to get structured JSON result for match analysis.")
        return {"error": "Failed to parse match analysis", "raw_response": result} if result else None


def generate_report_markdown(
    api_key: str,
    model: str,
    context: Union[str, Dict],
    report_prompt_key: str,
    prompts_data: Dict[str, str]
) -> str:
    """
    Generates markdown reports via run_ai_analysis.

    Args:
        api_key: OpenRouter API key.
        model: Model ID to use.
        context: The data or text to base the report on (can be string or dict).
        report_prompt_key: The key for the report generation prompt.
        prompts_data: Dictionary of loaded prompts.

    Returns:
        The generated markdown report as a string, or an error message string.
    """
    logger.info(f"Generating markdown report using prompt '{report_prompt_key}'...")
    if isinstance(context, dict):
        try:
            context_text = json.dumps(context, indent=2)
        except Exception as e:
            logger.error(f"Error formatting dictionary context for report generation: {e}")
            return f"# Report Generation Error\n\nCould not format input context: {e}"
    else:
        context_text = str(context)

    result = run_ai_analysis(api_key, model, report_prompt_key, context_text, prompts_data, expected_format="text")

    if isinstance(result, str):
        logger.info("Successfully generated markdown report.")
        return result
    else:
        logger.error("Failed to generate markdown report.")
        return "# Report Generation Error\n\nFailed to get text response from AI model."


def generate_word_cloud(text: str, stopwords: Optional[List[str]] = None) -> Optional[bytes]:
    """
    Generates a word cloud image bytes using the wordcloud library.

    Args:
        text: The input text for the word cloud.
        stopwords: An optional list of stopwords to exclude.

    Returns:
        The word cloud image as bytes, or None on error.
    """
    if not text:
        logger.warning("Cannot generate word cloud from empty text.")
        return None

    custom_stopwords = set(STOPWORDS)
    if stopwords:
        custom_stopwords.update(stopwords)

    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=custom_stopwords,
            min_font_size=10
        ).generate(text)

        # Save image to a bytes buffer
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        logger.info("Successfully generated word cloud image.")
        return img_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        return None


def analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """
    Analyzes sentiment (polarity and subjectivity) using TextBlob.

    Args:
        text: The input text.

    Returns:
        A tuple containing (polarity, subjectivity). Polarity is in [-1, 1], Subjectivity is in [0, 1].
        Returns (0.0, 0.0) if text is empty or analysis fails.
    """
    if not text:
        logger.warning("Cannot analyze sentiment of empty text.")
        return (0.0, 0.0)
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        logger.info(f"Sentiment analysis complete: Polarity={polarity:.2f}, Subjectivity={subjectivity:.2f}")
        return (polarity, subjectivity)
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return (0.0, 0.0)


def save_analysis_results(results: Union[Dict, List], output_path: str):
    """
    Saves analysis results (dictionary or list) to a JSON file.

    Args:
        results: The data to save.
        output_path: The path to the output JSON file.
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved analysis results to {output_path}")
    except TypeError as e:
        logger.error(f"Error: Data is not JSON serializable. Could not save to {output_path}. Error: {e}")
    except IOError as e:
        logger.error(f"Error writing results to file {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred saving results to {output_path}: {e}")


def generate_batch_summary_markdown(batch_results: List[Dict]) -> str:
    """
    Creates a simple markdown summary for batch processing results.

    Args:
        batch_results: A list of dictionaries, where each dict is an analysis result for one item.
                       Assumes each dict might have 'title', 'url', 'match_score', 'summary', etc.

    Returns:
        A markdown string summarizing the batch results.
    """
    if not batch_results:
        return "# Batch Summary\n\nNo results to summarize."

    summary_md = "# Batch Analysis Summary\n\n"
    summary_md += f"Processed {len(batch_results)} items.\n\n"
    summary_md += "| Title | Match Score | Key Insights |\n"
    summary_md += "|-------|-------------|--------------|\n"

    for i, result in enumerate(batch_results):
        title = result.get('title', f'Item {i+1}')
        # Example: Extracting a hypothetical match score and summary
        match_info = result.get('match_analysis', {})
        score = match_info.get('overall_match_score', 'N/A')
        insights = match_info.get('key_takeaways', 'No summary available.')

        # Truncate long insights for the table
        insights_short = (insights[:70] + '...') if len(insights) > 73 else insights
        insights_short = insights_short.replace('\n', ' ') # Remove newlines for table

        summary_md += f"| {title} | {score} | {insights_short} |\n"

    # Add more sections as needed, e.g., overall statistics
    # avg_score = ... calculate average score ...
    # summary_md += f"\n**Overall Average Match Score:** {avg_score:.2f}\n"

    logger.info("Generated batch summary markdown.")
    return summary_md


# --- Placeholder Functions ---

# --- Streamlit UI Implementation ---

# Helper to check if essential config is ready
def is_config_ready():
    return st.session_state.api_key and st.session_state.selected_model

# Helper to reset dependent states when context changes
def reset_dependent_states(job_changed=False, resume_changed=False, match_changed=False):
    """
    Resets Streamlit session states that depend on job, resume, or match analysis.

    This is crucial to ensure that analysis results are re-generated when
    the underlying input data (job or resume) changes.

    Args:
        job_changed: Boolean indicating if the job input has changed.
        resume_changed: Boolean indicating if the resume input has changed.
        match_changed: Boolean indicating if the match analysis needs resetting
                       (e.g., if model or prompts change).
    """
    if job_changed or resume_changed:
        if 'match_analysis_results' in st.session_state:
            del st.session_state.match_analysis_results
        if 'chat_history' in st.session_state:
            del st.session_state.chat_history
            st.info("Analysis context changed. Chat history cleared.") # Notify user
    if match_changed:
         if 'chat_history' in st.session_state:
            del st.session_state.chat_history
            st.info("Match analysis updated. Chat history cleared.") # Notify user

# Use cache_data for functions that don't depend on session state and whose results are reusable
@st.cache_data
def cached_fetch_available_models(api_key: str) -> List[str]:
    """Cached version of fetch_available_models."""
    return fetch_available_models(api_key)

@st.cache_data
def cached_fetch_and_clean_url_content(url: str) -> Optional[str]:
    """Cached version of fetch_and_clean_url_content."""
    return fetch_and_clean_url_content(url)

@st.cache_data
def cached_process_xml_batch(uploaded_file_content: bytes) -> List[Dict[str, str]]:
    """Cached version of process_xml_batch, taking bytes content."""
    # Need to write bytes to a temporary file-like object for ET.parse
    try:
        xml_file = io.BytesIO(uploaded_file_content)
        return process_xml_batch(xml_file) # Pass the file-like object
    except Exception as e:
        logger.error(f"Error processing uploaded XML content in cache: {e}")
        return []

@st.cache_data
def cached_generate_word_cloud(text: str) -> Optional[bytes]:
    """Cached version of generate_word_cloud."""
    return generate_word_cloud(text)

@st.cache_data
def cached_analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """Cached version of analyze_text_sentiment."""
    return analyze_text_sentiment(text)

def generate_pdf_report(analysis_data: Dict, report_type: str = "job") -> bytes:
    """
    Generates a professionally formatted PDF report from analysis data.
    
    Args:
        analysis_data: Dictionary containing analysis results
        report_type: Type of report ('job', 'resume', or 'match')
    
    Returns:
        PDF document as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()

    # Create custom styles with different names to avoid conflicts
    title_style = ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        name='ReportHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        name='ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        spaceBefore=6,
        spaceAfter=6
    )

    # Create a specific style for list items with indentation
    list_item_style = ParagraphStyle(
        name='ReportListItem',
        parent=normal_style, # Inherit from normal_style
        leftIndent=20,       # Add left indentation
        spaceBefore=3,       # Adjust spacing for list items
        spaceAfter=3
    )

    # Create the content elements
    elements = []

    if not isinstance(analysis_data, dict):
        logger.error(f"Invalid analysis_data type for PDF generation: {type(analysis_data)}. Expected dict.")
        # Generate a simple error report PDF
        elements.append(Paragraph("<b>PDF Generation Error</b>", title_style))
        elements.append(Paragraph(f"Could not generate the report due to invalid data format.", normal_style))
        elements.append(Paragraph(f"Expected dictionary, but received {type(analysis_data)}.", normal_style))
        elements.append(Paragraph(f"Please ensure the analysis completed successfully before attempting to generate the PDF.", normal_style))
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    # Add title and date
    current_date = datetime.now().strftime("%B %d, %Y")

    if report_type == "job":
        title = "Job Analysis Report"
        job_title = analysis_data.get("extraction", {}).get("job_title", "Job Position")
        company = analysis_data.get("extraction", {}).get("company_name", "Company")

        # Fallback: Try to extract from summary or comprehensive report if not found in extraction
        if job_title == "Job Position" or company == "Company":
            report_text = analysis_data.get("comprehensive_report", "") or analysis_data.get("summary", "")
            if report_text:
                # Attempt to find "the [Job Title] position at [Company Name]"
                match = re.search(r'the (.*?) position at (.*?) in', report_text, re.IGNORECASE)
                if match:
                    extracted_job_title = match.group(1).strip()
                    extracted_company = match.group(2).strip()
                    if job_title == "Job Position" and extracted_job_title:
                        job_title = extracted_job_title
                    if company == "Company" and extracted_company:
                        company = extracted_company
                else:
                    # Attempt to find "[Job Title] at [Company Name]"
                    match = re.search(r'(.*?) at (.*)', report_text, re.IGNORECASE)
                    if match:
                        extracted_job_title = match.group(1).strip()
                        extracted_company = match.group(2).strip()
                        if job_title == "Job Position" and extracted_job_title:
                            job_title = extracted_job_title
                        if company == "Company" and extracted_company:
                            company = extracted_company

        elements.append(Paragraph(f"<b>{title}</b>", title_style))
        elements.append(Paragraph(f"<b>Position:</b> {job_title}", normal_style))
        elements.append(Paragraph(f"<b>Company:</b> {company}", normal_style))
        elements.append(Paragraph(f"<b>Date:</b> {current_date}", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        # Add comprehensive report if available, otherwise use summary
        comprehensive_report = analysis_data.get("comprehensive_report", "")
        if not comprehensive_report:
            summary = analysis_data.get("summary", "No summary available.")
            elements.append(Paragraph("<b>Summary</b>", heading_style))
            elements.append(Paragraph(summary, normal_style))
        else:
            # Use a simple markdown-like parsing for PDF
            # This is a basic implementation and might not support all markdown features

            # Split into lines and process each line
            lines = comprehensive_report.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                # Handle headings (basic # and ##)
                if line.startswith('## '):
                    elements.append(Paragraph(f"<b>{line[3:].strip()}</b>", heading_style))
                elif line.startswith('# '):
                     elements.append(Paragraph(f"<b>{line[2:].strip()}</b>", title_style))
                # Handle list items (basic - )
                elif line.startswith('- '):
                    # Manually format list items with indentation and bullet using list_item_style
                    item_text = line[2:].strip()
                    # Basic handling for bold and italics within list items
                    formatted_item_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item_text)
                    formatted_item_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_item_text)
                    elements.append(Paragraph(f"•   {formatted_item_text}", list_item_style)) # Use list_item_style
                # Handle paragraphs (default)
                else:
                    # Basic handling for bold and italics within paragraphs
                    # Replace **text** with <b>text</b> and *text* with <i>text</i>
                    formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    formatted_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_line)
                    elements.append(Paragraph(formatted_line, normal_style))

                # Add a small space after each element
                elements.append(Spacer(1, 0.05*inch))

            # Add a larger space at the end of the comprehensive report section
            elements.append(Spacer(1, 0.25*inch))

        # Add skills section - Manually format list items using list_item_style
        skills = analysis_data.get("skills", {}).get("required_skills", [])
        if skills:
            elements.append(Paragraph("<b>Required Skills</b>", heading_style))
            for skill in skills:
                # Manually format list items with indentation and bullet using list_item_style
                elements.append(Paragraph(f"•   {skill}", list_item_style)) # Use list_item_style
            elements.append(Spacer(1, 0.1*inch))

        # Add tools and technologies - Manually format list items using list_item_style
        tools = analysis_data.get("tools", {}).get("technologies", [])
        if tools:
            elements.append(Paragraph("<b>Tools & Technologies</b>", heading_style))
            for tool in tools:
                # Manually format list items with indentation and bullet using list_item_style
                elements.append(Paragraph(f"•   {tool}", list_item_style)) # Use list_item_style
            elements.append(Spacer(1, 0.1*inch))

    elif report_type == "match":
        title = "Job-Resume Match Analysis"
        job_title = analysis_data.get("job_title", "Job Position")

        elements.append(Paragraph(f"<b>{title}</b>", title_style))
        elements.append(Paragraph(f"<b>Position:</b> {job_title}", normal_style))
        elements.append(Paragraph(f"<b>Date:</b> {current_date}", normal_style))
        elements.append(Spacer(1, 0.25*inch))

        # Add overall score
        overall_score = analysis_data.get("overall_match_score", "N/A")
        elements.append(Paragraph(f"<b>Overall Match Score:</b> {overall_score}/10", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        # Add strengths - Manually format list items using list_item_style
        strengths = analysis_data.get("strengths", [])
        if strengths:
            elements.append(Paragraph("<b>Strengths</b>", heading_style))
            for strength in strengths:
                # Manually format list items with indentation and bullet using list_item_style
                elements.append(Paragraph(f"•   {strength}", list_item_style)) # Use list_item_style
            elements.append(Spacer(1, 0.1*inch))

        # Add gaps - Manually format list items using list_item_style
        gaps = analysis_data.get("gaps", [])
        if gaps:
            elements.append(Paragraph("<b>Areas for Improvement</b>", heading_style))
            for gap in gaps:
                # Manually format list items with indentation and bullet using list_item_style
                elements.append(Paragraph(f"•   {gap}", list_item_style)) # Use list_item_style
            elements.append(Spacer(1, 0.1*inch))

        # Add recommendations
        recommendations = analysis_data.get("recommendations", "No recommendations available.")
        elements.append(Paragraph("<b>Recommendations</b>", heading_style))
        # Process recommendations with basic markdown formatting
        formatted_recommendations = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', recommendations)
        formatted_recommendations = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_recommendations)
        elements.append(Paragraph(formatted_recommendations, normal_style))

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
# PDF generation function is defined above

def run_streamlit_app():
    """Runs the Streamlit web interface for the Job Insight Analyzer."""
    st.set_page_config(page_title="Job Insight Analyzer", layout="wide")
# --- Custom CSS for UI improvements ---
    st.markdown("""
        <style>
        /* General body styling */
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
            background-color: #f4f4f4;
        }
        /* Title styling */
        .stApp > header {
            background-color: #007bff;
            padding: 1rem;
            color: white;
            text-align: center;
        }
        .stApp > header h1 {
            color: white;
        }
        /* Sidebar styling (if any) */
        .stSidebar {
            background-color: #e9ecef;
        }
        /* Main content area padding */
        .stApp {
            padding-top: 1rem;
        }
        /* Button styling */
        .stButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #218838;
            color: white;
        }
         .stButton>button:disabled {
            background-color: #cccccc;
            color: #666666;
            cursor: not-allowed;
        }
        /* Text area and input styling */
        .stTextArea textarea, .stTextInput input {
            border-radius: 0.5rem;
            border: 1px solid #ced4da;
            padding: 0.5rem;
        }
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #007bff;
        }
        /* Info, Warning, Error boxes */
        .stAlert {
            border-radius: 0.5rem;
        }
        /* Metrics */
        [data-testid="stMetric"] label {
            font-weight: bold;
            color: #555;
        }
        [data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
            display: none; /* Hide delta arrow */
        }
        </style>
        """, unsafe_allow_html=True)

    # --- Initialization ---
    prompts_data = load_prompts()
    if not prompts_data:
        st.error("Fatal Error: Could not load `prompts.json`. Please ensure the file exists and is valid.")
        return # Stop execution if prompts are missing

    # Initialize session state keys if they don't exist
    # Explicitly check and initialize each key for robustness
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "use_env_key" not in st.session_state:
        st.session_state.use_env_key = True
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "job_analysis_results" not in st.session_state:
        st.session_state.job_analysis_results = None
    if "resume_analysis_results" not in st.session_state:
        st.session_state.resume_analysis_results = None
    if "match_analysis_results" not in st.session_state:
        st.session_state.match_analysis_results = None
    if "batch_jobs" not in st.session_state:
        st.session_state.batch_jobs = None
    if "batch_analysis_results" not in st.session_state:
        st.session_state.batch_analysis_results = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "job_input_type" not in st.session_state:
        st.session_state.job_input_type = "URL"
    if "job_url_input" not in st.session_state:
        st.session_state.job_url_input = ""
    if "job_text_input" not in st.session_state:
        st.session_state.job_text_input = ""
    if "resume_text_input" not in st.session_state:
        st.session_state.resume_text_input = ""
    if "uploaded_batch_file" not in st.session_state:
        st.session_state.uploaded_batch_file = None

    # Attempt to load API key from env if toggle is set
    if st.session_state.use_env_key:
        env_key = get_api_key()
        if env_key:
            st.session_state.api_key = env_key
        # If env key not found, api_key remains None until manually entered

    st.title("🚀 Job Insight Analyzer")

    # --- Main Navigation Tabs ---
    tab_config, tab_single, tab_batch, tab_resume, tab_chat = st.tabs([
        "⚙️ Model Config",
        "📄 Single Job Analysis",
        "📊 Batch Job Analysis",
        "🤝 Resume & Match",
        "💬 Chat with Results"
    ])

    # --- Model Config Tab ---
    with tab_config:
        st.header("Model Configuration")
        st.markdown("Configure your OpenRouter API Key and select the analysis model.")

        # Store the toggle state change immediately
        use_env = st.toggle(
            "Use OPENROUTER_API_KEY environment variable",
            value=st.session_state.use_env_key,
            key="use_env_key_toggle", # Add a key for stability
            help="If enabled, tries to load the key from the environment. If disabled, use the manual input below."
        )
        # Update session state based on toggle interaction BEFORE evaluating logic
        if use_env != st.session_state.use_env_key:
            st.session_state.use_env_key = use_env
            # Clear the key when toggling to force re-evaluation
            st.session_state.api_key = None
            st.rerun() # Rerun to update UI elements based on new toggle state

        # Determine API key and display inputs/status based on toggle state
        current_api_key = None
        if st.session_state.use_env_key:
            # --- Environment Variable Mode ---
            env_key = get_api_key()
            if env_key:
                current_api_key = env_key
                st.info("API Key loaded successfully from environment variable.")
                # Display disabled manual input for clarity
                st.text_input(
                    "Enter OpenRouter API Key manually",
                    type="password",
                    placeholder="Using environment variable",
                    disabled=True,
                    key="manual_api_key_input_disabled"
                )
            else:
                st.warning("Environment variable `OPENROUTER_API_KEY` not found. Please set it or disable this toggle and enter manually.")
                # Display disabled manual input
                st.text_input(
                    "Enter OpenRouter API Key manually",
                    type="password",
                    placeholder="Environment variable not found",
                    disabled=True,
                    key="manual_api_key_input_disabled_warn"
                )
        else:
            # --- Manual Input Mode ---
            manual_key_input = st.text_input(
                "Enter OpenRouter API Key manually",
                type="password",
                placeholder="sk-or-v1-...",
                key="manual_api_key_input_active", # Use a different key when active
                help="Enter your key here as the environment variable toggle is off."
            )
            if manual_key_input:
                current_api_key = manual_key_input

        # Update the session state *after* determining the key for this run
        st.session_state.api_key = current_api_key

        # Display API Key Status based on the final key value for this run
        if st.session_state.api_key:
            st.success(f"API Key is configured (ending in ...{st.session_state.api_key[-4:]}).")
        else:
            st.error("API Key is not configured. Please set the environment variable or enter it manually.")

        # Fetch and display available models
        if st.session_state.api_key:
            if not st.session_state.available_models:
                with st.spinner("Fetching available models..."):
                    st.session_state.available_models = cached_fetch_available_models(st.session_state.api_key)
            
            if st.session_state.available_models:
                default_model = "google/gemini-2.5-flash-preview:online"
                default_index = 0 # Default to the first item if preferred model not found

                if default_model in st.session_state.available_models:
                    default_index = st.session_state.available_models.index(default_model)
                elif st.session_state.available_models:
                     default_index = 0 # Fallback to the first available model

                selected_model = st.selectbox(
                    "Select Model:",
                    st.session_state.available_models,
                    index=default_index,
                    format_func=lambda x: x.split("/")[-1] if "/" in x else x,  # Display only model name, not provider
                    help="Select the model to use for analysis."
                )
                st.session_state.selected_model = selected_model
                st.success(f"Using model: `{selected_model}`")
            else:
                # If no models available, use a default
                default_model = "google/gemini-2.5-flash-preview:online"
                st.session_state.selected_model = default_model
                st.warning(f"Could not fetch models from OpenRouter. Using default model: `{default_model}`")
        else:
            st.session_state.selected_model = None  # Clear model if no API key

    # --- Single Job Analysis Tab ---
    with tab_single:
        st.header("Single Job Posting Analysis")

        # Input Section
        st.session_state.job_input_type = st.radio(
            "Select Input Method:",
            ("URL", "Paste Text"),
            key="single_job_input_type_radio",
            horizontal=True
        )

        job_input_value = None
        if st.session_state.job_input_type == "URL":
            st.session_state.job_url_input = st.text_input("Job Posting URL:", value=st.session_state.job_url_input)
            job_input_value = st.session_state.job_url_input
        else:
            st.session_state.job_text_input = st.text_area("Paste Job Description Text:", value=st.session_state.job_text_input, height=250)
            job_input_value = st.session_state.job_text_input

        analyze_button_disabled = not is_config_ready() or not job_input_value
        if st.button("Analyze Job", disabled=analyze_button_disabled, type="primary"):
            if not is_config_ready():
                st.warning("Please configure API Key and select a model first.")
            elif not job_input_value:
                 st.warning("Please provide a URL or paste job text.")
            else:
                job_text_to_analyze = None
                with st.spinner("Processing job input..."):
                    if st.session_state.job_input_type == "URL":
                        try:
                            job_text_to_analyze = cached_fetch_and_clean_url_content(job_input_value)
                            if not job_text_to_analyze:
                                st.error(f"Failed to fetch or clean content from URL: {job_input_value}. The URL might be invalid or the content could not be extracted.")
                                job_text_to_analyze = None # Ensure it's None on error
                        except Exception as e:
                            st.error(f"An error occurred while fetching the URL: {e}")
                            job_text_to_analyze = None # Ensure it's None on error
                    else:
                        job_text_to_analyze = job_input_value # Already have the text

                if job_text_to_analyze:
                    with st.spinner(f"Analyzing job posting using {st.session_state.selected_model}..."):
                        analysis_result = analyze_job_posting(
                            st.session_state.api_key,
                            st.session_state.selected_model,
                            job_text_to_analyze,
                            prompts_data
                        )
                        if analysis_result:
                            st.session_state.job_analysis_results = analysis_result
                            reset_dependent_states(job_changed=True) # Clear match/chat

                            # Check for analysis errors within the result
                            if "analysis_errors" in analysis_result:
                                st.warning("Job analysis completed with some errors:")
                                for err in analysis_result["analysis_errors"]:
                                    st.markdown(f"- {err}")
                                # Optionally remove the error key from the displayed results JSON
                                display_results = analysis_result.copy()
                                del display_results["analysis_errors"]
                                st.session_state.job_analysis_results = display_results # Store cleaned version for display/download
                            else:
                                st.success("Job analysis complete!")
                        else:
                            st.error("Job analysis failed. Check logs or model configuration.")
                            st.session_state.job_analysis_results = None
                else:
                   st.error("Job analysis failed. Check logs or model configuration.")
                   st.session_state.job_analysis_results = None

        # Display Single Job Results
        if "job_analysis_results" in st.session_state and st.session_state.job_analysis_results:
            st.divider()
            st.subheader("Analysis Results")

            # Display source URL if input method was URL
            if st.session_state.job_input_type == "URL" and st.session_state.job_url_input:
                st.markdown(f"**Source URL:** [{st.session_state.job_url_input}]({st.session_state.job_url_input})")
                st.markdown("---")  # Add a separator

            # Use columns for better layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # Display structured data using expander
                with st.expander("View Full Analysis JSON", expanded=False):
                    st.json(st.session_state.job_analysis_results)

                # Display key sections using markdown if available in results
                summary = st.session_state.job_analysis_results.get("summary", "Summary not available.")
                key_reqs = st.session_state.job_analysis_results.get("key_requirements", [])
                nice_to_haves = st.session_state.job_analysis_results.get("nice_to_haves", [])

                st.markdown("#### Summary")
                st.markdown(summary)

                if key_reqs:
                    st.markdown("#### Key Requirements")
                    for req in key_reqs: st.markdown(f"- {req}")

                if nice_to_haves:
                    st.markdown("#### Nice-to-Haves")
                    for nth in nice_to_haves: st.markdown(f"- {nth}")

                # Generate and display comprehensive markdown report
                with st.spinner("Generating comprehensive markdown report..."):
                    comprehensive_md = generate_report_markdown(
                        st.session_state.api_key,
                        st.session_state.selected_model,
                        st.session_state.job_analysis_results,
                        "comprehensive_report_markdown", # Use the specific prompt key
                        prompts_data
                    )
                with st.expander("View Comprehensive Markdown Report", expanded=True):
                    st.markdown(comprehensive_md)

            with col2:
                # Sentiment Analysis
                job_text_for_sentiment = st.session_state.job_text_input if st.session_state.job_input_type == "Paste Text" else cached_fetch_and_clean_url_content(st.session_state.job_url_input)
                if job_text_for_sentiment:
                    polarity, subjectivity = cached_analyze_text_sentiment(job_text_for_sentiment)
                    st.metric("Sentiment Polarity", f"{polarity:.2f}", help="Range [-1 (Negative) to 1 (Positive)]")
                    st.metric("Sentiment Subjectivity", f"{subjectivity:.2f}", help="Range [0 (Objective) to 1 (Subjective)]")

                # Word Cloud
                if job_text_for_sentiment:
                    with st.spinner("Generating word cloud..."):
                        wordcloud_bytes = cached_generate_word_cloud(job_text_for_sentiment)
                    if wordcloud_bytes:
                        st.image(wordcloud_bytes, caption="Job Description Word Cloud", use_container_width=True)
                    else:
                        st.warning("Could not generate word cloud.")

            # Download Buttons
            st.divider()
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                try:
                    json_str = json.dumps(st.session_state.job_analysis_results, indent=4)
                    st.download_button(
                        label="Download Full JSON Result",
                        data=json_str,
                        file_name=f"job_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Could not prepare JSON for download: {e}")
            with col_dl2:
                st.download_button(
                    label="Download Markdown Report",
                    data=comprehensive_md,
                    file_name=f"job_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
            with col_dl3:
                try:
                    # Generate PDF report with comprehensive report included
                    job_data_with_report = st.session_state.job_analysis_results.copy()
                    job_data_with_report["comprehensive_report"] = comprehensive_md
                    
                    pdf_bytes = generate_pdf_report(job_data_with_report, "job")
                    st.download_button(
                        label="Export as PDF",
                        data=pdf_bytes,
                        file_name=f"job_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"Could not generate PDF: {e}")

    # --- Batch Job Analysis Tab ---
    with tab_batch:
        st.header("Batch Job Analysis (from XML)")

        uploaded_file = st.file_uploader(
            "Upload XML file containing job postings",
            type="xml",
            key="batch_xml_uploader"
        )

        # Process uploaded file only if it's different from the stored one
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_batch_file:
            st.session_state.uploaded_batch_file = uploaded_file # Store the new file object
            with st.spinner("Parsing XML file..."):
                # Read content as bytes for caching
                uploaded_file_content = uploaded_file.getvalue()
                try:
                    st.session_state.batch_jobs = cached_process_xml_batch(uploaded_file_content)

                    if st.session_state.batch_jobs:
                        st.success(f"Successfully parsed {len(st.session_state.batch_jobs)} jobs from '{uploaded_file.name}'.")
                        st.session_state.batch_analysis_results = None # Clear previous batch results
                    else:
                        st.warning(f"No valid jobs found in '{uploaded_file.name}'. Check file format and tags.")
                        st.session_state.batch_jobs = None
                except Exception as e:
                    st.error(f"An error occurred while parsing the XML file '{uploaded_file.name}': {e}")
                    st.session_state.batch_jobs = None
        elif uploaded_file is None and st.session_state.uploaded_batch_file is not None:
            # If user removes the file, clear the state
            st.session_state.uploaded_batch_file = None
            st.session_state.batch_jobs = None
            st.session_state.batch_analysis_results = None
            st.info("File removed.")


        batch_analyze_disabled = not is_config_ready() or not st.session_state.batch_jobs
        num_jobs = len(st.session_state.batch_jobs) if st.session_state.batch_jobs else 0

        if st.button(f"Analyze {num_jobs} Jobs", disabled=batch_analyze_disabled, type="primary"):
            if not is_config_ready():
                st.warning("Please configure API Key and select a model first.")
            elif not st.session_state.batch_jobs:
                st.warning("Please upload and parse a valid XML file first.")
            else:
                st.session_state.batch_analysis_results = []
                progress_bar = st.progress(0, text="Starting batch analysis...")
                total_jobs = len(st.session_state.batch_jobs)

                for i, job_data in enumerate(st.session_state.batch_jobs):
                    job_text = job_data.get('description', '')
                    job_title = job_data.get('title', f'Job {i+1}')
                    progress_text = f"Analyzing job {i+1}/{total_jobs}: '{job_title[:30]}...'"
                    progress_bar.progress((i + 1) / total_jobs, text=progress_text)

                    if not job_text:
                        logger.warning(f"Skipping job {i+1} ('{job_title}') due to empty description.")
                        st.session_state.batch_analysis_results.append({"title": job_title, "error": "Empty description"})
                        continue

                    analysis_result = analyze_job_posting(
                        st.session_state.api_key,
                        st.session_state.selected_model,
                        job_text,
                        prompts_data
                    )

                    if analysis_result:
                        # Add original title/url back for reference if available
                        analysis_result['original_title'] = job_title
                        if 'url' in job_data: analysis_result['original_url'] = job_data['url']
                        st.session_state.batch_analysis_results.append(analysis_result)
                    else:
                        logger.error(f"Failed analysis for job {i+1} ('{job_title}').")
                        st.session_state.batch_analysis_results.append({"title": job_title, "error": "Analysis failed"})

                progress_bar.progress(1.0, text="Batch analysis complete!")
                st.success(f"Finished analyzing {total_jobs} jobs.")

        # Display Batch Results
        if "batch_analysis_results" in st.session_state and st.session_state.batch_analysis_results:
            st.divider()
            st.subheader("Batch Analysis Results")

            # Generate and display summary markdown
            batch_summary_md = generate_batch_summary_markdown(st.session_state.batch_analysis_results)
            with st.expander("View Batch Summary Report", expanded=True):
                st.markdown(batch_summary_md)

            # Display individual results in expanders
            for i, result in enumerate(st.session_state.batch_analysis_results):
                title = result.get('original_title', result.get('title', f'Item {i+1}'))
                with st.expander(f"Result for: {title}", expanded=False):
                    if "error" in result:
                        st.error(f"Analysis Error for '{title}': {result['error']}")
                        if "raw_response" in result:
                            st.text("Raw response (if available):")
                            st.text(result["raw_response"])
                    else:
                        st.json(result)
                        # Optionally add sentiment/word cloud like in single analysis
                        # Find the original job text for visualization if available
                        original_job_data = next((job for job in (st.session_state.batch_jobs or []) if job.get('title') == title), None)
                        if original_job_data and 'description' in original_job_data:
                            job_text_for_viz = original_job_data['description']
                            # Sentiment
                            pol, subj = cached_analyze_text_sentiment(job_text_for_viz)
                            st.metric(f"Sentiment Polarity", f"{pol:.2f}")
                            st.metric(f"Sentiment Subjectivity", f"{subj:.2f}")
                            # Word Cloud (consider performance for many jobs)
                            # wc_bytes = cached_generate_word_cloud(job_text_for_viz)
                            # if wc_bytes: st.image(wc_bytes, caption=f"Word Cloud for {title}", use_column_width=True)
                        elif 'original_url' in result:
                            st.info(f"Original job text for visualization not available (fetched from URL: {result['original_url']}).")
                        else:
                            st.info("Original job text for visualization not available.")


            # Download Buttons for Batch
            st.divider()
            col_bdl1, col_bdl2, col_bdl3 = st.columns(3)
            with col_bdl1:
                try:
                    batch_json_str = json.dumps(st.session_state.batch_analysis_results, indent=4)
                    st.download_button(
                        label="Download Full Batch JSON Results",
                        data=batch_json_str,
                        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Could not prepare batch JSON for download: {e}")
            with col_bdl2:
                st.download_button(
                    label="Download Batch Summary Markdown",
                    data=batch_summary_md,
                    file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
            with col_bdl3:
                try:
                    # For batch analysis, we'll create a PDF with the comprehensive report
                    # Generate a more comprehensive report for the batch analysis
                    batch_report_md = batch_summary_md
                    
                    # Create a data structure with the comprehensive report
                    batch_summary_data = {
                        "comprehensive_report": batch_report_md,
                        "job_count": len(st.session_state.batch_analysis_results),
                        "extraction": {"job_title": "Batch Analysis", "company_name": "Multiple Companies"}
                    }
                    batch_pdf_bytes = generate_pdf_report(batch_summary_data, "job")
                    st.download_button(
                        label="Export as PDF",
                        data=batch_pdf_bytes,
                        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"Could not generate PDF: {e}")

    # --- Resume & Match Tab ---
    with tab_resume:
        st.header("Resume Analysis & Job Match")

        st.session_state.resume_text_input = st.text_area(
            "Paste Resume Text:",
            value=st.session_state.resume_text_input,
            height=300,
            key="resume_input_area"
        )

        analyze_resume_disabled = not is_config_ready() or not st.session_state.resume_text_input
        if st.button("Analyze Resume", disabled=analyze_resume_disabled):
            if not is_config_ready():
                st.warning("Please configure API Key and select a model first.")
            elif not st.session_state.resume_text_input:
                st.warning("Please paste resume text first.")
            else:
                with st.spinner(f"Analyzing resume using {st.session_state.selected_model}..."):
                    analysis_result = analyze_resume_text(
                        st.session_state.api_key,
                        st.session_state.selected_model,
                        st.session_state.resume_text_input,
                        prompts_data
                    )
                    if analysis_result:
                        st.session_state.resume_analysis_results = analysis_result
                        reset_dependent_states(resume_changed=True) # Clear match/chat
                        st.success("Resume analysis complete!")
                    else:
                        error_msg = "Resume analysis failed."
                        if analysis_result and isinstance(analysis_result, dict) and "error" in analysis_result:
                            error_msg += f" Error: {analysis_result['error']}"
                            if "raw_response" in analysis_result:
                                error_msg += f" Raw response: {analysis_result['raw_response'][:100]}..." # Truncate raw response
                        st.error(error_msg)
                        st.session_state.resume_analysis_results = None

        # Display Resume Analysis Results
        if "resume_analysis_results" in st.session_state and st.session_state.resume_analysis_results:
            st.divider()
            st.subheader("Resume Analysis Results")
            col_res1, col_res2 = st.columns([2,1])
            with col_res1:
                with st.expander("View Resume Analysis JSON", expanded=False):
                    st.json(st.session_state.resume_analysis_results)
                # Display key parts if available
                # Extract key sections from the analysis results
                resume_summary = st.session_state.resume_analysis_results.get("professional_summary",
                    st.session_state.resume_analysis_results.get("summary", "Summary not available."))
                skills = (
                    st.session_state.resume_analysis_results.get("skills", {}).get("technical", []) +
                    st.session_state.resume_analysis_results.get("skills", {}).get("soft_skills", [])
                )
                st.markdown("#### Summary")
                st.markdown(resume_summary)
                if skills:
                    st.markdown("#### Extracted Skills")
                    st.markdown(", ".join(skills))

            with col_res2:
                # Sentiment
                res_pol, res_subj = cached_analyze_text_sentiment(st.session_state.resume_text_input)
                st.metric("Resume Sentiment Polarity", f"{res_pol:.2f}")
                st.metric("Resume Sentiment Subjectivity", f"{res_subj:.2f}")

        # Match Calculation
        st.divider()
        st.subheader("Job-Resume Match Calculation")

        calculate_match_disabled = not is_config_ready() or not st.session_state.job_analysis_results or not st.session_state.resume_analysis_results
        if not st.session_state.job_analysis_results:
            st.info("Analyze a job posting first (in the 'Single Job Analysis' tab) to enable matching.")
        if not st.session_state.resume_analysis_results:
            st.info("Analyze a resume first (above) to enable matching.")

        if st.button("Calculate Match Score", disabled=calculate_match_disabled, type="primary"):
            if not is_config_ready():
                st.warning("Please configure API Key and select a model first.")
            elif not st.session_state.job_analysis_results:
                st.warning("Job analysis results are missing.")
            elif not st.session_state.resume_analysis_results:
                st.warning("Resume analysis results are missing.")
            else:
                with st.spinner(f"Calculating match score using {st.session_state.selected_model}..."):
                    match_result = analyze_match(
                        st.session_state.api_key,
                        st.session_state.selected_model,
                        st.session_state.job_analysis_results,
                        st.session_state.resume_analysis_results,
                        prompts_data
                    )
                    if match_result:
                        st.session_state.match_analysis_results = match_result
                        reset_dependent_states(match_changed=True) # Clear chat
                        st.success("Match analysis complete!")
                    else:
                        error_msg = "Match analysis failed."
                        if match_result and isinstance(match_result, dict) and "error" in match_result:
                            error_msg += f" Error: {match_result['error']}"
                            if "raw_response" in match_result:
                                error_msg += f" Raw response: {match_result['raw_response'][:100]}..." # Truncate raw response
                        st.error(error_msg)
                        st.session_state.match_analysis_results = None

        # Display Match Results
        # Safely check if match_analysis_results exists and is not None
        if 'match_analysis_results' in st.session_state and st.session_state.match_analysis_results:
            st.divider()
            st.subheader("Match Analysis Results")

            # Extract key match details (adjust keys based on actual prompt output)
            overall_score = st.session_state.match_analysis_results.get("overall_match_score", 0)
            strengths = st.session_state.match_analysis_results.get("strengths", [])
            gaps = st.session_state.match_analysis_results.get("gaps", [])
            recommendations = st.session_state.match_analysis_results.get("recommendations", "No recommendations provided.")
            detailed_scores = st.session_state.match_analysis_results.get("detailed_scores", {}) # e.g., {"skills": 8, "experience": 7}

            # Display using metrics, charts, and markdown
            st.metric("Overall Match Score", f"{overall_score}/10")

            if detailed_scores and isinstance(detailed_scores, dict):
                st.markdown("#### Detailed Scores")
                # Enhanced bar chart using Altair with better formatting
                try:
                    score_data = pd.DataFrame(list(detailed_scores.items()), columns=['Category', 'Score'])
                    
                    # Create a more visually appealing chart with labels and colors
                    chart = alt.Chart(score_data).mark_bar(
                        cornerRadiusEnd=3,  # Rounded corners
                        color='#4682B4'     # Steel blue color
                    ).encode(
                        x=alt.X('Score:Q',
                              scale=alt.Scale(domain=[0, 10]),
                              axis=alt.Axis(title='Score (0-10)', grid=True)),
                        y=alt.Y('Category:N',
                              sort='-x',
                              axis=alt.Axis(title='Skill Category', labelLimit=150)),
                        tooltip=['Category', 'Score']
                    ).properties(
                        title='Match Score Breakdown',
                        height=min(40 * len(detailed_scores), 300)  # Dynamic height based on number of categories
                    )
                    
                    # Add text labels to the bars
                    text = chart.mark_text(
                        align='left',
                        baseline='middle',
                        dx=3,  # Offset from the end of the bar
                        color='white'
                    ).encode(
                        text=alt.Text('Score:Q', format='.1f')
                    )
                    
                    # Combine the chart and text
                    final_chart = (chart + text)
                    st.altair_chart(final_chart, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Could not generate detailed score chart: {e}")
                    # Fallback to a simple but effective table visualization
                    st.write("Score Breakdown:")
                    score_df = pd.DataFrame(list(detailed_scores.items()), columns=['Category', 'Score'])
                    score_df = score_df.sort_values('Score', ascending=False)
                    st.dataframe(score_df, use_container_width=True, hide_index=True)

            col_match1, col_match2 = st.columns(2)
            with col_match1:
                st.markdown("#### Strengths")
                if strengths:
                    for strength in strengths: st.markdown(f"- {strength}")
                else:
                    st.markdown("No specific strengths identified.")
            with col_match2:
                st.markdown("#### Gaps / Areas for Improvement")
                if gaps:
                    for gap in gaps: st.markdown(f"- {gap}")
                else:
                    st.markdown("No specific gaps identified.")

            with st.expander("Tailoring Recommendations", expanded=True):
                st.markdown(recommendations)

            with st.expander("View Full Match JSON", expanded=False):
                st.json(st.session_state.match_analysis_results)

            # Download and Export Options
            st.divider()
            col_match_dl1, col_match_dl2 = st.columns(2)
            
            with col_match_dl1:
                try:
                    match_json_str = json.dumps(st.session_state.match_analysis_results, indent=4)
                    st.download_button(
                        label="Download Match Analysis JSON",
                        data=match_json_str,
                        file_name=f"match_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Could not prepare match JSON for download: {e}")
            
            with col_match_dl2:
                try:
                    # Generate a comprehensive report for the match analysis
                    match_report_md = generate_report_markdown(
                        st.session_state.api_key,
                        st.session_state.selected_model,
                        st.session_state.match_analysis_results,
                        "comprehensive_report_markdown",
                        prompts_data
                    )
                    
                    # Include the comprehensive report in the match data for PDF generation
                    match_data_with_report = st.session_state.match_analysis_results.copy()
                    match_data_with_report["comprehensive_report"] = match_report_md
                    
                    # Generate PDF with comprehensive report
                    match_pdf_bytes = generate_pdf_report(match_data_with_report, "match")
                    st.download_button(
                        label="Export as PDF",
                        data=match_pdf_bytes,
                        file_name=f"match_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"Could not generate PDF: {e}")


    # --- Chat with Results Tab ---
    with tab_chat:
        st.header("Chat with Analysis Results")

        # Chat with Results Tab
        # Disable chat if no job analysis results are available
        chat_disabled = not st.session_state.job_analysis_results
        if chat_disabled:
            st.info("Please analyze a job posting first to enable chat.")

        # Display chat history, safely accessing session state
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the analysis results...", disabled=chat_disabled):
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Construct context for the LLM
            # Construct context for the LLM, safely accessing session state
            chat_context = {
                "job_analysis": st.session_state.get("job_analysis_results"),
                "resume_analysis": st.session_state.get("resume_analysis_results"),
                "match_analysis": st.session_state.get("match_analysis_results"),
                "recent_chat_history": st.session_state.get("chat_history", [])[-5:] # Include recent history
            }
            try:
                context_text = json.dumps(chat_context, indent=2)
            except Exception as e:
                st.error(f"Error formatting context for chat: {e}")
                context_text = "{'error': 'Could not format context'}"

            # Define the chat prompt (using a generic or specific key)
            # Option 1: Use a dedicated chat prompt key
            chat_prompt_key = "chat_with_results"
            # Option 2: Construct prompt dynamically (less ideal if complex)
            # system_message = "You are a helpful assistant analyzing job/resume data..."
            # user_message = f"Based on the following data:\n{context_text}\n\nAnd the recent chat history, answer this question: {prompt}"

            # Display assistant response placeholder and stream
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    client = _configure_openai_client(st.session_state.api_key)
                    # Prepare messages using the prompt key
                    prompt_template = prompts_data.get(chat_prompt_key, "Based on the context: {context}\n\nAnswer the user question: {user_question}")
                    final_user_prompt = prompt_template.format(context=context_text, user_question=prompt)

                    messages = [
                        {"role": "system", "content": "You are an expert HR analyst assistant. Answer questions based on the provided analysis data and chat history."},
                        {"role": "user", "content": final_user_prompt}
                    ]

                    # Use streaming if possible
                    stream = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=messages,
                        stream=True,
                        temperature=0.7,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌") # Simulate typing cursor
                    message_placeholder.markdown(full_response) # Final response

                except openai.APIConnectionError as e:
                    full_response = f"Chat Error: Could not connect to the API. Please check your internet connection and API key configuration. Details: {e}"
                    message_placeholder.error(full_response)
                except openai.RateLimitError as e:
                    full_response = f"Chat Error: Rate limit exceeded. Please wait a moment before trying again. Details: {e}"
                    message_placeholder.error(full_response)
                except openai.AuthenticationError as e:
                    full_response = f"Chat Error: Authentication failed. Please check your API key. Details: {e}"
                    message_placeholder.error(full_response)
                except openai.APITimeoutError as e:
                    full_response = f"Chat Error: Request timed out. Please try again. Details: {e}"
                    message_placeholder.error(full_response)
                except openai.APIStatusError as e:
                    full_response = f"Chat Error: API returned an error. Status: {e.status_code}. Details: {e.response}"
                    message_placeholder.error(full_response)
                except Exception as e:
                    full_response = f"An unexpected error occurred during chat: {e}"
                    message_placeholder.error(full_response)

            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            # No st.rerun needed here, chat updates automatically

def run_cli_analysis():
    """Placeholder for the Command Line Interface logic."""
    print("CLI not implemented yet.")
    # Example: You might parse arguments and call analysis functions here later
    # parser = argparse.ArgumentParser(description="Job Insight Analyzer CLI")
    # parser.add_argument("--url", help="URL of the job posting")
    # parser.add_argument("--resume", help="Path to the resume file")
    # ... other arguments ...
    # args = parser.parse_args()
    # prompts = load_prompts()
    # api_key = get_api_key()
    # ... call analysis functions based on args ...
    pass

# --- Main Execution Block ---

if __name__ == "__main__":
    # Assume the script is run via Streamlit if executed directly
    logger.info("Starting Streamlit app...")
    run_streamlit_app()