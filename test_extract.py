from src.react_agent.file_utils import extract_resume_text

resume_text = extract_resume_text("samples/user_resume.docx")  # or .docx / .txt
print(resume_text)