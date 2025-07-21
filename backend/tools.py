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

# --- Resume Optimization Tool ---

class ResumeOptimizationInput(BaseModel):
    user_resume: str = Field(..., description="The user's actual resume in plain text")
    ideal_resume: str = Field(..., description="The ideal resume generated based on the job description")
    job_description: str = Field(..., description="The job description for which the resume is being tailored")


@tool("optimize_user_resume", args_schema=ResumeOptimizationInput)
def optimize_user_resume(*, user_resume: str, ideal_resume: str, job_description: str) -> str:
    """
    Optimize the user's resume to better match the ideal resume for the job without adding false information.
    Reorganizes projects, updates vocabulary for ATS optimization, and improves presentation.
    """
    prompt = (
        f"You are an expert resume optimizer with deep knowledge of ATS systems and hiring standards.\n\n"
        f"Your task is to optimize the USER RESUME below to more closely match the IDEAL RESUME without adding any false information.\n"
        f"This is crucial: Only use information that already exists in the USER RESUME. Do not invent or add experiences,\n"
        f"projects, skills, or qualifications that are not already mentioned in some form.\n\n"
        f"USER RESUME:\n"
        f"\"\"\"\
{user_resume}\n\"\"\"\
\n"
        f"IDEAL RESUME (Reference only - do not copy directly):\n"
        f"\"\"\"\
{ideal_resume}\n\"\"\"\
\n"
        f"JOB DESCRIPTION:\n"
        f"\"\"\"\
{job_description}\n\"\"\"\
\n"
        f"Please optimize the USER RESUME by doing the following:\n\n"
        f"1. Reorganize content to match the structure and flow of the ideal resume where appropriate\n"
        f"2. Reorder projects and experiences to prioritize those most relevant to the job description\n"
        f"3. Update vocabulary and phrasing to be more ATS-friendly and match keywords from the job description\n"
        f"4. Improve bullet points with more impactful language and quantifiable achievements (only if such details exist in original)\n"
        f"5. Enhance formatting and section organization for better readability\n\n"
        f"Rules you MUST follow:\n"
        f"- DO NOT add work experiences that don't exist in the original resume\n"
        f"- DO NOT add skills that aren't mentioned or implied in the original resume\n"
        f"- DO NOT invent projects or technical expertise not evidenced in the original resume\n"
        f"- DO NOT fabricate accomplishments or metrics not supported by the original content\n"
        f"- DO maintain the truthfulness of the original resume at all costs\n"
        f"- DO use the ideal resume only as a structural and stylistic guide\n\n"
        f"Return the complete optimized resume in a clean, professional format with appropriate markdown formatting.\n\n"
        f"IMPORTANT: Your response should be ONLY the optimized resume text in markdown format. Do not include any explanations, disclaimers or notes before or after the resume."
    )
    response = llm.invoke(prompt)
    return response.content

# Expose all tools to the LangGraph agent
TOOLS = [generate_ideal_resume, compare_resumes, generate_resume_improvements, optimize_user_resume]
