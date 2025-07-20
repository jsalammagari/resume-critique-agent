"""Test the resume critique agent functionality."""

import asyncio
import sys
import os
import pathlib
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import after loading environment variables
from backend.tools import generate_ideal_resume, compare_resumes, generate_resume_improvements

# Sample job description
JOB_DESCRIPTION = """
Software Engineer (Python/AI)
Company: TechInnovate Inc.

About the Role:
We're looking for a talented Software Engineer with strong Python skills and experience in AI/ML to join our team. You'll be working on building and improving our AI-powered products.

Responsibilities:
- Design, develop, and maintain efficient, reusable, and reliable Python code
- Implement machine learning models and integrate them into our products
- Collaborate with cross-functional teams to define, design, and ship new features
- Optimize applications for maximum speed and scalability
- Write clean, maintainable code with proper test coverage

Requirements:
- Bachelor's degree in Computer Science or related field
- 3+ years experience with Python
- Experience with at least one machine learning framework (PyTorch, TensorFlow, scikit-learn)
- Knowledge of software engineering practices (CI/CD, testing, code reviews)
- Familiarity with REST APIs and microservices architecture
- Strong problem-solving skills and attention to detail

Bonus:
- Experience with cloud platforms (AWS, GCP, Azure)
- Knowledge of containerization (Docker, Kubernetes)
- Experience with NLP or computer vision projects
- Open-source contributions
"""

# Sample user resume
USER_RESUME = """
John Doe
Phone: 555-123-4567
Email: john.doe@example.com
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe

EDUCATION
Bachelor of Science in Computer Science
University of Technology, 2018-2022

SKILLS
- Programming Languages: Python, JavaScript, Java
- Web: HTML, CSS, React
- Tools: Git, Docker, VS Code
- Other: Basic understanding of machine learning concepts

WORK EXPERIENCE
Junior Software Developer
WebTech Solutions, 2022-Present
- Developed and maintained web applications using Python and Django
- Implemented RESTful APIs for mobile application integration
- Fixed bugs and improved application performance
- Participated in code reviews and team meetings

Intern Software Developer
DataCorp Inc., Summer 2021
- Assisted in developing features for the company's data analysis tool
- Fixed bugs in the Python codebase
- Created basic data visualizations using Matplotlib

PROJECTS
Personal Website
- Built a personal portfolio website using React and Node.js
- Implemented responsive design and animations

Inventory Management System
- Created a simple inventory system using Python and SQLite
- Implemented basic CRUD operations and search functionality

Weather App
- Built a weather application that fetches data from a public API
- Used HTML, CSS, and JavaScript for the frontend
"""

async def test_generate_ideal_resume():
    """Test the generate_ideal_resume function."""
    print("\nTesting generate_ideal_resume...")
    result = await generate_ideal_resume.ainvoke({"job_description": JOB_DESCRIPTION})
    print(f"Generated ideal resume: {result[:500]}...")  # Print first 500 chars
    return result

async def test_compare_resumes(ideal_resume):
    """Test the compare_resumes function."""
    print("\nTesting compare_resumes...")
    result = await compare_resumes.ainvoke({"user_resume": USER_RESUME, "ideal_resume": ideal_resume})
    print(f"Resume comparison result: {result[:500]}...")  # Print first 500 chars
    return result

async def test_generate_improvements():
    """Test the generate_resume_improvements function."""
    print("\nTesting generate_resume_improvements...")
    result = await generate_resume_improvements.ainvoke({
        "user_resume": USER_RESUME, 
        "job_description": JOB_DESCRIPTION
    })
    print(f"Resume improvement suggestions: {result[:500]}...")  # Print first 500 chars
    return result

async def main():
    """Run all tests."""
    # Make sure GROQ_API_KEY is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it and try again.")
        return

    try:
        ideal_resume = await test_generate_ideal_resume()
        await test_compare_resumes(ideal_resume)
        await test_generate_improvements()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
