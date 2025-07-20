from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama3-70b-8192",  # Using Llama 3 70B model
    temperature=0.3
)

# --- Resume Comparison Tool ---

class ResumeComparisonInput(BaseModel):
    user_resume: str = Field(..., description="The user's actual resume in plain text")
    ideal_resume: str = Field(..., description="The ideal resume generated based on the job description")

@tool("compare_resumes", args_schema=ResumeComparisonInput)
def compare_resumes(*, user_resume: str, ideal_resume: str) -> str:
    """
    Compare a user resume with an ideal resume and return a similarity score with a detailed analysis.
    """
    prompt = f"""
You are an expert technical recruiter. Compare the following two resumes with attention to detail:

1. Ideal Resume:
\"\"\"
{ideal_resume}
\"\"\"

2. User Resume:
\"\"\"
{user_resume}
\"\"\"

Evaluate alignment on the following categories:
- Skills (technical skills, soft skills, tools/frameworks)
- Projects (relevance, complexity, technical alignment)
- Work experience (role responsibilities, industry relevance, achievements)
- Education (degree relevance, specializations, certifications)
- Overall presentation and formatting

For each category:
1. Provide a similarity score (0-100%)
2. Detailed analysis of strengths
3. Specific gaps or areas for improvement

Finally, provide:
- Overall similarity score (weighted average across all categories)
- Summary of key strengths (3-5 points)
- Prioritized list of 3-5 specific improvement recommendations

Format your response in clear sections with markdown headers.
"""
    response = llm.invoke(prompt)
    return response.content


# --- Ideal Resume Generator Tool ---

class JobDescriptionInput(BaseModel):
    job_description: str = Field(..., description="The full job description including company, responsibilities, and qualifications.")

@tool("generate_ideal_resume", args_schema=JobDescriptionInput)
def generate_ideal_resume(*, job_description: str) -> str:
    """
    Generate the most ideal resume based on a provided job description.
    """
    prompt = f"""
You are an expert resume writer with deep knowledge of ATS (Applicant Tracking Systems) and industry hiring standards.

Based on the job description below, generate the most ideal resume tailored to the role. 
Ensure the resume is both ATS-optimized (using relevant keywords) and appealing to human recruiters.

Job Description:
\"\"\"
{job_description}
\"\"\"

Include these sections:
- Full Name
- Mobile Number
- LinkedIn Profile Link
- GitHub Profile Link
- Email ID
- Professional Summary (tailored to the job description with key qualifications)

Education:
- List the most aligned degree(s) with relevant coursework.

Projects:
- Include 3-4 project titles and detailed summaries precisely aligned with the job requirements.
- For each project, highlight technologies used, your role, and measurable outcomes.

Work Experience:
- Add highly relevant experiences based on the job description.
- Format each entry with: Job Title, Company, Dates, and 3-5 bullet points with accomplishments.
- Use action verbs and quantify achievements wherever possible.

Skills (grouped as):
- Programming Languages
- Web Technologies
- Database Systems
- Tools / Frameworks / Platforms
- Domain-Specific Skills
- Soft Skills

Certifications & Professional Development:
- Include relevant certifications for the position.

Other Interests & Achievements:
- Include a few that strengthen the candidate's profile.

Use markdown formatting for structure and ensure the resume is comprehensive but concise.
"""
    response = llm.invoke(prompt)
    return response.content

# --- Resume Improvement Tool ---

class ResumeImprovementInput(BaseModel):
    user_resume: str = Field(..., description="The user's actual resume in plain text")
    job_description: str = Field(..., description="The job description for which the resume is being tailored")

@tool("generate_resume_improvements", args_schema=ResumeImprovementInput)
def generate_resume_improvements(*, user_resume: str, job_description: str) -> str:
    """
    Generate specific, actionable improvements for a user's resume based on a job description.
    """
    prompt = f"""
You are an expert resume consultant specializing in resume optimization for job applications.

Analyze the following user's resume in relation to this specific job description:

User Resume:
\"\"\"
{user_resume}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Provide a comprehensive resume improvement plan with the following elements:

1. **Keyword Analysis**:
   - Identify 8-10 critical keywords/phrases from the job description that should be incorporated
   - Show how to naturally integrate these keywords into the resume

2. **Section-by-Section Improvements**:
   - Professional Summary: Specific rewrites to better align with the role
   - Work Experience: How to reframe accomplishments to match job requirements
   - Skills Section: What to add, remove, or reorganize
   - Projects: How to better highlight relevant projects or reframe existing ones
   - Education/Certifications: Recommendations for emphasis or additional information

3. **ATS Optimization Tips**:
   - Formatting recommendations to ensure ATS compatibility
   - Structure suggestions to improve machine readability

4. **Content Enhancement Suggestions**:
   - Specific examples of weak statements and how to improve them with metrics/achievements
   - Identification of irrelevant content that could be removed
   - Suggestions for addressing potential experience gaps

Provide your recommendations in a clear, actionable format with specific examples wherever possible.
"""
    response = llm.invoke(prompt)
    return response.content

# Expose all tools to the LangGraph agent
TOOLS = [generate_ideal_resume, compare_resumes, generate_resume_improvements]
