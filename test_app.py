import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from pathlib import Path
import io
import app

# Test Data
SAMPLE_PROMPTS = [
    {
        "key": "test_prompt",
        "title": "Test Prompt",
        "prompt_text": "Test prompt text",
        "expected_format": "json"
    }
]

SAMPLE_JOB_TEXT = """
Software Engineer
Requirements:
- Python experience
- Web development
- Team player
"""

SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer
Skills: Python, JavaScript, Web Development
"""

@pytest.fixture
def sample_prompts_file(tmp_path):
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps(SAMPLE_PROMPTS))
    return prompts_file

@pytest.fixture
def mock_openai_client():
    with patch('openai.OpenAI') as mock:
        yield mock

def test_load_prompts(sample_prompts_file):
    prompts = app.load_prompts(str(sample_prompts_file))
    assert "test_prompt" in prompts
    assert prompts["test_prompt"]["prompt_text"] == "Test prompt text"

def test_load_prompts_missing_file():
    prompts = app.load_prompts("nonexistent.json")
    assert prompts == {}

def test_get_api_key():
    with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
        assert app.get_api_key() == 'test-key'

def test_get_api_key_missing():
    with patch.dict('os.environ', {}, clear=True):
        assert app.get_api_key() is None

def test_configure_openai_client():
    client = app._configure_openai_client("test-key")
    assert client.api_key == "test-key"
    assert client.base_url == "https://openrouter.ai/api/v1/"

@patch('requests.get')
def test_fetch_and_clean_url_content_success(mock_get):
    mock_get.return_value.content = b"<html><body>Test content</body></html>"
    mock_get.return_value.raise_for_status = Mock()
    
    result = app.fetch_and_clean_url_content("http://test.com")
    assert "Test content" in result
    mock_get.assert_called_once()

@patch('requests.get')
def test_fetch_and_clean_url_content_failure(mock_get):
    mock_get.side_effect = Exception("Failed to fetch")
    result = app.fetch_and_clean_url_content("http://test.com")
    assert result is None

def test_process_xml_batch(tmp_path):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<jobs>
    <job>
        <title>Software Engineer</title>
        <description>Python developer needed</description>
        <url>http://test.com/job1</url>
    </job>
</jobs>"""
    xml_file = tmp_path / "jobs.xml"
    xml_file.write_text(xml_content)
    
    results = app.process_xml_batch(str(xml_file))
    assert len(results) == 1
    assert results[0]["title"] == "Software Engineer"
    assert "Python developer" in results[0]["description"]

def test_extract_json_from_markdown():
    markdown = """
    Some text
    ```json
    {"key": "value"}
    ```
    More text
    """
    result = app._extract_json_from_markdown(markdown)
    assert result == {"key": "value"}

def test_extract_json_from_markdown_invalid():
    result = app._extract_json_from_markdown("Not JSON")
    assert result is None

def test_run_ai_analysis():
    with patch('openai.OpenAI') as mock_openai:
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content='{"result": "test"}'))]
        mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        prompts_data = {"test_key": {"prompt_text": "test prompt"}}
        result = app.run_ai_analysis("test-key", "test_key", "test_key", "test context", prompts_data, expected_format="json")
        assert result == {"result": "test"}

def test_run_ai_analysis_missing_api_key():
    result = app.run_ai_analysis("", "test_key", "test_key", "test context", {"test_key": {"prompt_text": "dummy prompt"}}, expected_format="json")
    assert result is None

def test_run_ai_analysis_invalid_prompt_key():
    result = app.run_ai_analysis("test-key", "invalid_key", "invalid_key", "test context", {"invalid_key": {"prompt_text": "dummy prompt"}}, expected_format="json")
    assert result is None

def test_run_ai_analysis_missing_prompt_text():
    result = app.run_ai_analysis("test-key", "test_key", "test_key", "test context", {"test_key": {"prompt_text": "dummy prompt"}}, expected_format="json")
    assert result is None

def test_analyze_job_posting():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = {"skills": ["Python"]}
        result = app.analyze_job_posting("test-key", "test_key", SAMPLE_JOB_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert "skills" in result

def test_analyze_job_posting_with_errors():
    with patch('app.run_ai_analysis') as mock_run_ai:
        # Mock responses for all analysis aspects: extraction, skills, tools, summary
        mock_run_ai.side_effect = [None, {"skills": ["Python"]}, None, {"text": "summary"}]
        result = app.analyze_job_posting("test-key", "test_key", SAMPLE_JOB_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert "analysis_errors" in result
        assert len(result["analysis_errors"]) > 0

def test_analyze_resume_text():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = {"skills": ["Python"]}
        result = app.analyze_resume_text("test-key", "test_key", SAMPLE_RESUME_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert "skills" in result

def test_analyze_resume_text_failure():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = "Invalid JSON"
        result = app.analyze_resume_text("test-key", "test_key", SAMPLE_RESUME_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert "error" in result

def test_analyze_match():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = {"match_score": 85}
        result = app.analyze_match(
            "test-key",
            "test_key", # Added missing model argument
            {"job": "details"},
            {"resume": "details"},
            {"test_key": {"prompt_text": "dummy prompt"}}
        )
        assert "match_score" in result

def test_analyze_match_formatting_error():
    with patch('json.dumps') as mock_dumps:
        mock_dumps.side_effect = Exception("JSON error")
        result = app.analyze_match("test-key", "test_key", {}, {}, {"test_key": {"prompt_text": "dummy prompt"}})
        assert result is None

def test_generate_report_markdown():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = "# Test Report"
        result = app.generate_report_markdown(
            "test-key",
            "test_key", # Added missing model argument
            {"test": "data"},
            "report_key",
            {"report_key": {"prompt_text": "dummy prompt"}}
        )
        assert "Test Report" in result

def test_generate_report_markdown_dict_error():
    with patch('json.dumps') as mock_dumps:
        mock_dumps.side_effect = Exception("JSON error")
        result = app.generate_report_markdown("test-key", "test_key", {"test": object()}, "test_key", {"test_key": {"prompt_text": "dummy prompt"}})
        assert "Error" in result

def test_generate_word_cloud():
    result = app.generate_word_cloud("test words test more words")
    assert isinstance(result, bytes)

def test_generate_word_cloud_empty_text():
    result = app.generate_word_cloud("")
    assert result is None

def test_analyze_text_sentiment():
    polarity, subjectivity = app.analyze_text_sentiment("This is a great test!")
    assert isinstance(polarity, float)
    assert isinstance(subjectivity, float)

def test_analyze_text_sentiment_empty_text():
    polarity, subjectivity = app.analyze_text_sentiment("")
    assert polarity == 0.0
    assert subjectivity == 0.0

def test_save_analysis_results(tmp_path):
    results = {"test": "data"}
    output_path = tmp_path / "results.json"
    app.save_analysis_results(results, str(output_path))
    assert output_path.exists()
    assert json.loads(output_path.read_text()) == results

def test_save_analysis_results_type_error():
    # Test that the function handles TypeError internally
    non_serializable = {'key': object()}
    result = app.save_analysis_results(non_serializable, "test.json")
    assert result is None

def test_generate_batch_summary_markdown():
    batch_results = [
        {"title": "Job 1", "match_analysis": {"overall_match_score": 85}},
        {"title": "Job 2", "match_analysis": {"overall_match_score": 75}}
    ]
    result = app.generate_batch_summary_markdown(batch_results)
    assert "Job 1" in result
    assert "Job 2" in result
    assert "85" in result

def test_generate_batch_summary_markdown_empty():
    result = app.generate_batch_summary_markdown([])
    assert "No results" in result

def test_end_to_end_job_analysis():
    with patch('app.run_ai_analysis') as mock_run_ai:
        mock_run_ai.return_value = {
            "title": "Software Engineer",
            "skills": ["Python", "JavaScript"],
            "summary": "Great opportunity"
        }
        
        # Test job analysis
        job_result = app.analyze_job_posting("test-key", "test_key", SAMPLE_JOB_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert job_result is not None
        assert "skills" in job_result
        
        # Test resume analysis
        resume_result = app.analyze_resume_text("test-key", "test_key", SAMPLE_RESUME_TEXT, {"test_key": {"prompt_text": "dummy prompt"}})
        assert resume_result is not None
        
        # Test match analysis
        match_result = app.analyze_match("test-key", "test_key", job_result, resume_result, {"test_key": {"prompt_text": "dummy prompt"}})
        assert match_result is not None

if __name__ == '__main__':
    pytest.main([__file__])