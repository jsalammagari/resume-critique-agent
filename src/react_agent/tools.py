from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0.3)

# Define the input schema
class JobDescriptionInput(BaseModel):
    job_description: str = Field(..., description="The full job description including company, responsibilities, and qualifications.")

# Define the tool
@tool("generate_ideal_resume", args_schema=JobDescriptionInput)
def generate_ideal_resume(input: JobDescriptionInput) -> str:
    """
    Generate the most ideal resume based on the given job description.
    """
    prompt = f"""
You are an expert resume writer.

Based on the job description below, generate the most ideal resume tailored to this job.

Job Description:
\"\"\"
{input.job_description}
\"\"\"

The resume must include:

- Full Name
- Mobile Number
- LinkedIn Profile Link
- GitHub Profile Link
- Email ID

Education:
- List the most aligned degree.

Projects:
- Include 5–6 project titles and summaries that align closely with the job requirements.

Work Experience:
- Add highly relevant experiences based on the job description.

Skills (grouped as below):
- Programming Languages
- Web
- Database
- Tools / Frameworks / Platforms

Other Interests & Achievements:
- Add a few that would enhance the resume for this job.

Use markdown formatting for structure.
"""

    response = llm.invoke(prompt)
    return response.content

# Expose the tool(s) to the graph
TOOLS = [generate_ideal_resume]
