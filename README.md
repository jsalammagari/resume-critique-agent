# Resume Critique Agent

## Overview
An AI-powered Resume Critique Agent that analyzes resumes against job descriptions to provide tailored feedback, improvement suggestions, and a similarity score. Built with a modern web interface using Flask and Groq's AI API, the system features real-time progress tracking, detailed analysis, actionable improvement suggestions, and resume optimization.

## Technology Stack

### Backend
- **Python 3.9+**: Core programming language
- **Flask**: Web framework for serving the application
- **LangChain**: Framework for building LLM-powered applications
- **LangGraph**: Tool for creating agent workflows with state management
- **Groq API**: High-speed LLM provider powering all AI functionality
- **Pydantic**: Data validation and settings management
- **Threading**: For background task processing and cleanup

### Frontend
- **HTML5/CSS3/JavaScript**: Core web technologies
- **Bootstrap 5**: UI framework for responsive design
- **Font Awesome**: Icon library
- **AOS**: Animation library for scroll effects
- **Marked.js**: Markdown parsing for dynamic content rendering
- **AJAX**: For asynchronous progress polling

### Development & Deployment
- **dotenv**: Environment variable management
- **Logging**: Comprehensive backend logging system

## Features & Tools

### AI Analysis Tools
- **Resume Comparison Tool**: Evaluates user resume against ideal candidate profile with detailed scoring and feedback
- **Ideal Resume Generator**: Creates a tailored ideal resume based on job description
- **Improvement Suggestions Tool**: Provides actionable, specific recommendations to enhance resume effectiveness
- **Resume Optimization Tool**: Restructures and enhances user resume to match ideal resume without adding false information

### User Experience Features
- **Real-time Progress Tracking**: Shows detailed progress during analysis with timestamp logs
- **Tab-based Results Interface**: Organizes results in intuitive, accessible tabs
- **ATS-Compatibility Analysis**: Evaluates how well resumes will perform with ATS systems
- **Similarity Scoring**: Provides percentage-based scoring with color-coded visualization
- **Detailed Logging**: Backend captures timing and performance metrics for transparency
- **Responsive Design**: Works across desktop and mobile devices

## Project Structure

```
resume-critique-agent/
├── app.py                  # Main Flask application with endpoints and progress tracking
├── backend/
│   ├── __init__.py         # Backend module initialization
│   ├── config.py           # Configuration settings for the agent
│   ├── graph.py            # Agent graph structure defining the workflow
│   ├── prompts.py          # System prompts for the agent
│   ├── tools.py            # Tools for resume analysis (generate ideal, compare, improve)
│   └── utils.py            # Utility functions for the backend
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css  # Custom CSS styles for the UI
│   │   └── js/
│   │       └── main.js     # Frontend JavaScript with progress polling and UI logic
│   └── templates/
│       └── index.html      # Main HTML template for the web interface
└── .env.example            # Example environment variable configuration
```

## Getting Started

### Prerequisites
- Python 3.9 or higher
- A Groq API key (required for AI functionality)

### Setup

1. Clone the repository

```bash
git clone https://github.com/jsalammagari/resume-critique-agent.git
cd resume-critique-agent
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

```bash
cp .env.example .env
```

5. Edit the `.env` file and add your Groq API key

```
GROQ_API_KEY=your-api-key-here
```

### Running the Application

1. Start the Flask application

```bash
python app.py
```

This will start the server on http://127.0.0.1:5001

2. Open the application in your web browser by navigating to:

```
http://127.0.0.1:5001
```

## Usage

1. **Enter Resume Text**: Paste your resume content into the left text area

2. **Enter Job Description**: Paste the job description into the right text area

3. **Click "Analyze Resume"**: The system will process your resume against the job description

4. **View Progress**: Watch real-time progress updates during analysis

5. **Review Results**: Explore the three tabs of analysis results:
   - **Resume Comparison**: See how your resume matches against the job requirements
   - **Improvement Suggestions**: Get actionable feedback to enhance your resume
   - **Ideal Resume**: View an AI-generated ideal resume for the position

## Development Notes

- **Processing Time**: Analysis typically takes 20-30 seconds depending on resume and job description complexity
- **Rate Limits**: If you encounter slower processing, it may be due to Groq API rate limits
- **Logging**: Check the terminal output for detailed logs during processing

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://github.com/langchain-ai/langgraph) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

[^1]: https://python.langchain.com/docs/concepts/#tools

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "agent": {
      "type": "object",
      "properties": {
        "model": {
          "type": "string",
          "default": "anthropic/claude-3-5-sonnet-20240620",
          "description": "The name of the language model to use for the agent's main interactions. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic/claude-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.0",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.1",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-5-sonnet-20240620",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-haiku-20240307",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-opus-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-sonnet-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-instant-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      },
      "environment": [
        "TAVILY_API_KEY"
      ]
    }
  }
}
-->