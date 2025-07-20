import os
import sys
import time
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log'))
    ]
)
logger = logging.getLogger('resume-critique')

# Load environment variables at the very beginning
# Get the absolute path to the .env file
env_path = Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)
logger.info("Environment variables loaded successfully")

# Now import other dependencies after environment is loaded
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import asyncio
from backend.tools import generate_ideal_resume, compare_resumes, generate_resume_improvements

# Dictionary to store request progress
request_progress = {}

# Helper function to update progress
def update_progress(request_id, stage, message, percentage=None):
    if request_id not in request_progress:
        request_progress[request_id] = []
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    update = {
        "stage": stage,
        "message": message,
        "percentage": percentage,
        "timestamp": timestamp
    }
    request_progress[request_id].append(update)
    logger.info(f"[{request_id}] {stage} - {message} - {percentage if percentage else ''}%")
    return update

app = Flask(__name__, 
            static_folder="frontend/static",
            template_folder="frontend/templates")

@app.route('/')
def index():
    """Render the main page of the resume critique application."""
    return render_template('index.html')

@app.route('/api/critique', methods=['POST'])
def critique_resume():
    """API endpoint to process resume and job description."""
    data = request.json
    
    user_resume = data.get('resume', '')
    job_description = data.get('jobDescription', '')
    
    # Generate a unique request ID
    request_id = f"req_{int(time.time())}_{hash(user_resume[:50])}"
    logger.info(f"New request {request_id} received")
    
    if not user_resume or not job_description:
        logger.warning(f"[{request_id}] Missing required fields")
        return jsonify({'error': 'Resume and job description are required'}), 400
    
    # Initialize progress tracking for this request
    update_progress(request_id, "start", "Request received and processing started", 0)
    
    try:
        # Create event loop to run async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Log resume and job description lengths
        logger.info(f"[{request_id}] Resume length: {len(user_resume)} chars, Job description length: {len(job_description)} chars")
        update_progress(request_id, "preparation", "Analyzing input data", 10)
        
        start_time = time.time()
        
        # Generate ideal resume based on job description
        update_progress(request_id, "ideal_resume", "Generating ideal resume based on job description", 20)
        ideal_resume_start = time.time()
        ideal_resume = loop.run_until_complete(
            generate_ideal_resume.ainvoke({"job_description": job_description})
        )
        ideal_resume_time = time.time() - ideal_resume_start
        logger.info(f"[{request_id}] Generated ideal resume in {ideal_resume_time:.2f} seconds")
        update_progress(request_id, "ideal_resume", f"Ideal resume generated in {ideal_resume_time:.2f} seconds", 50)
        
        # Compare user resume with ideal resume
        update_progress(request_id, "comparison", "Comparing your resume with ideal candidate profile", 60)
        comparison_start = time.time()
        comparison_result = loop.run_until_complete(
            compare_resumes.ainvoke({
                "user_resume": user_resume,
                "ideal_resume": ideal_resume
            })
        )
        comparison_time = time.time() - comparison_start
        logger.info(f"[{request_id}] Completed resume comparison in {comparison_time:.2f} seconds")
        update_progress(request_id, "comparison", f"Resume comparison completed in {comparison_time:.2f} seconds", 75)
        
        # Generate improvement suggestions
        update_progress(request_id, "improvements", "Generating personalized improvement suggestions", 80)
        improvements_start = time.time()
        improvement_suggestions = loop.run_until_complete(
            generate_resume_improvements.ainvoke({
                "user_resume": user_resume,
                "job_description": job_description
            })
        )
        improvements_time = time.time() - improvements_start
        logger.info(f"[{request_id}] Generated improvement suggestions in {improvements_time:.2f} seconds")
        update_progress(request_id, "improvements", f"Improvement suggestions generated in {improvements_time:.2f} seconds", 95)
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Total processing time: {total_time:.2f} seconds")
        update_progress(request_id, "complete", f"Analysis completed in {total_time:.2f} seconds", 100)
        
        return jsonify({
            'requestId': request_id,
            'idealResume': ideal_resume,
            'comparisonResult': comparison_result,
            'improvementSuggestions': improvement_suggestions,
            'processingTime': f"{total_time:.2f}s",
            'progress': request_progress[request_id]
        })
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"[{request_id}] Error processing request: {error_message}", exc_info=True)
        update_progress(request_id, "error", f"Error: {error_message}", None)
        return jsonify({
            'error': f'Error processing request: {error_message}',
            'progress': request_progress.get(request_id, [])
        }), 500

# API endpoint to get progress for a specific request
@app.route('/api/progress/<request_id>', methods=['GET'])
def get_progress(request_id):
    """Return progress information for a specific request ID."""
    if request_id not in request_progress:
        return jsonify({'error': 'Request ID not found'}), 404
    
    return jsonify({
        'requestId': request_id,
        'progress': request_progress[request_id]
    })

# Cleanup old request progress data periodically
def cleanup_old_requests():
    """Remove old request progress data to prevent memory leaks"""
    while True:
        time.sleep(3600)  # Run once per hour
        current_time = time.time()
        to_remove = []
        
        for req_id in request_progress:
            # Extract timestamp from request ID
            try:
                req_time = int(req_id.split('_')[1])
                if current_time - req_time > 86400:  # Older than 24 hours
                    to_remove.append(req_id)
            except (IndexError, ValueError):
                pass
        
        for req_id in to_remove:
            logger.info(f"Removing old request progress data for {req_id}")
            request_progress.pop(req_id, None)

# Start the cleanup thread when the application starts
import threading
cleanup_thread = threading.Thread(target=cleanup_old_requests, daemon=True)
cleanup_thread.start()
logger.info("Started cleanup thread for old request progress data")

if __name__ == '__main__':
    logger.info("Starting Resume Critique Agent server on port 5001")
    app.run(debug=True, port=5001)
