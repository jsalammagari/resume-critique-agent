// Resume Critique Agent - Enhanced Frontend JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Initialize AOS animations
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        mirror: false
    });

    // DOM Elements
    const resumeForm = document.getElementById('resumeForm');
    const resumeInput = document.getElementById('resumeInput');
    const jobDescriptionInput = document.getElementById('jobDescriptionInput');
    const analyzeButton = document.getElementById('analyzeButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const comparisonContent = document.getElementById('comparisonContent');
    const improvementsContent = document.getElementById('improvementsContent');
    const idealContent = document.getElementById('idealContent');
    const optimizedContent = document.getElementById('optimizedContent');

    // Sample data for testing UI (uncomment if needed for development)
    // const sampleData = {
    //    resume: "JOHN DOE\nSoftware Engineer\n5+ years of experience...",
    //    jobDescription: "Senior Software Engineer position requiring..."
    // };
    // if you want to prefill for testing: resumeInput.value = sampleData.resume;

    // Enhanced form validation with visual feedback
    function validateInput(input, errorMessage) {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            const feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            feedback.textContent = errorMessage;
            input.parentNode.appendChild(feedback);
            input.focus();
            return false;
        } else {
            input.classList.remove('is-invalid');
            const existingFeedback = input.parentNode.querySelector('.invalid-feedback');
            if (existingFeedback) {
                existingFeedback.remove();
            }
            return true;
        }
    }

    // Form submission handler with enhanced UX
    resumeForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        // Remove any existing validations
        document.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
        document.querySelectorAll('.invalid-feedback').forEach(el => el.remove());
        
        // Enhanced validation
        const resumeValid = validateInput(resumeInput, 'Please enter your resume text');
        const jobDescriptionValid = validateInput(jobDescriptionInput, 'Please enter the job description');
        
        if (!resumeValid || !jobDescriptionValid) {
            return;
        }
        
        // Show loading state with enhanced UI
        analyzeButton.disabled = true;
        loadingSpinner.classList.remove('d-none');
        const originalButtonText = analyzeButton.innerHTML;
        analyzeButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing Resume...';
        
        // Create loading overlay with progress tracking
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = `
            <div class="loading-content">
                <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <h4 class="mt-3 loading-title">Analyzing Your Resume</h4>
                <p class="mt-2 loading-text">Initializing analysis...</p>
                <div class="progress mt-3" style="height: 15px; width: 300px;">
                    <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">
                        <span id="progressPercent">0%</span>
                    </div>
                </div>
                <div id="progressStages" class="progress-stages mt-3">
                    <div class="progress-stage" data-stage="preparation">
                        <i class="fas fa-hourglass-start"></i> Preparation
                        <span class="stage-status">Waiting...</span>
                    </div>
                    <div class="progress-stage" data-stage="ideal_resume">
                        <i class="fas fa-file-alt"></i> Ideal Resume Generation
                        <span class="stage-status">Waiting...</span>
                    </div>
                    <div class="progress-stage" data-stage="comparison">
                        <i class="fas fa-balance-scale"></i> Resume Comparison
                        <span class="stage-status">Waiting...</span>
                    </div>
                    <div class="progress-stage" data-stage="improvements">
                        <i class="fas fa-clipboard-check"></i> Improvement Suggestions
                        <span class="stage-status">Waiting...</span>
                    </div>
                    <div class="progress-stage" data-stage="optimization">
                        <i class="fas fa-star"></i> Resume Optimization
                        <span class="stage-status">Waiting...</span>
                    </div>
                </div>
                <div id="progressLog" class="progress-log mt-3">
                    <div class="log-entry">Starting analysis, please wait...</div>
                </div>
            </div>
        `;
        document.body.appendChild(loadingOverlay);
        
        // Initialize progress tracking variables
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const progressStages = document.getElementById('progressStages');
        const progressLog = document.getElementById('progressLog');
        const loadingText = document.querySelector('.loading-text');
        
        // Function to update progress display
        function updateProgressDisplay(stage, message, percentage) {
            // Update progress bar
            if (percentage !== null && !isNaN(percentage)) {
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
                progressPercent.textContent = `${percentage}%`;
            }
            
            // Update status message
            if (message) {
                loadingText.textContent = message;
                
                // Add to log
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.innerHTML = `<span class="log-time">${new Date().toLocaleTimeString()}</span> ${message}`;
                progressLog.appendChild(logEntry);
                progressLog.scrollTop = progressLog.scrollHeight; // Auto-scroll to bottom
            }
            
            // Update stage indicators
            if (stage) {
                const stageElement = progressStages.querySelector(`[data-stage="${stage}"]`);
                if (stageElement) {
                    // Mark all previous stages as complete
                    let markComplete = true;
                    progressStages.querySelectorAll('.progress-stage').forEach(el => {
                        if (el === stageElement) {
                            el.classList.add('stage-active');
                            el.classList.remove('stage-complete');
                            el.querySelector('.stage-status').textContent = 'In Progress...';
                            markComplete = false;
                        } else if (markComplete) {
                            el.classList.remove('stage-active');
                            el.classList.add('stage-complete');
                            el.querySelector('.stage-status').textContent = 'Complete';
                        }
                    });
                }
            }
        }
        
        // Initial progress update
        updateProgressDisplay('preparation', 'Starting analysis...', 0);
        
        try {
            // Add a timeout to periodically check for progress
            let pollingTimer = null;
            let requestId = null;
            const pollingInterval = 1000; // Poll every second
            
            // Function to poll for progress updates
            async function pollProgress() {
                if (!requestId) return;
                
                try {
                    const progressResponse = await fetch(`/api/progress/${requestId}`);
                    if (progressResponse.ok) {
                        const progressData = await progressResponse.json();
                        const updates = progressData.progress || [];
                        
                        if (updates.length > 0) {
                            const latestUpdate = updates[updates.length - 1];
                            updateProgressDisplay(
                                latestUpdate.stage,
                                latestUpdate.message,
                                latestUpdate.percentage
                            );
                            
                            // If we're not at 100% yet, schedule another poll
                            if (latestUpdate.percentage < 100) {
                                pollingTimer = setTimeout(pollProgress, pollingInterval);
                            }
                        }
                    }
                } catch (e) {
                    console.error('Error polling for progress:', e);
                    // Even if there's an error, keep polling
                    pollingTimer = setTimeout(pollProgress, pollingInterval);
                }
            }
            
            // Start the API call
            // First update to show request is being sent
            updateProgressDisplay('preparation', 'Sending request to server...', 5);
            
            // Call the API
            const response = await fetch('/api/critique', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    resume: resumeInput.value.trim(),
                    jobDescription: jobDescriptionInput.value.trim()
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                // If there's progress information in the error response, display it
                if (errorData.progress && Array.isArray(errorData.progress)) {
                    errorData.progress.forEach(update => {
                        updateProgressDisplay(update.stage, update.message, update.percentage);
                    });
                }
                throw new Error(errorData.error || 'An error occurred during analysis');
            }
            
            const data = await response.json();
            
            // Capture the request ID if available
            if (data.requestId) {
                requestId = data.requestId;
                console.log(`Request ID captured: ${requestId}`);
                
                // Start polling for progress updates
                pollProgress();
            }
            
            // Process progress information if available
            if (data.progress && Array.isArray(data.progress)) {
                data.progress.forEach(update => {
                    updateProgressDisplay(update.stage, update.message, update.percentage);
                });
                
                // Add final completion message if not already included
                if (data.processingTime) {
                    updateProgressDisplay('complete', `Analysis completed in ${data.processingTime}`, 100);
                    
                    // Clear any remaining polling timers
                    if (pollingTimer) {
                        clearTimeout(pollingTimer);
                        pollingTimer = null;
                    }
                }
            }
            
            // Display processing time information
            if (data.processingTime) {
                const processingTimeInfo = document.createElement('div');
                processingTimeInfo.className = 'alert alert-info mb-4';
                processingTimeInfo.innerHTML = `
                    <i class="fas fa-info-circle me-2"></i>
                    Analysis completed in <strong>${data.processingTime}</strong>
                `;
                resultsSection.prepend(processingTimeInfo);
                
                // Auto-dismiss after 10 seconds
                setTimeout(() => {
                    processingTimeInfo.classList.add('fade');
                    setTimeout(() => processingTimeInfo.remove(), 500);
                }, 10000);
            }
            
            // Enhanced results display with formatting
            comparisonContent.innerHTML = marked.parse(data.comparisonResult);
            improvementsContent.innerHTML = marked.parse(data.improvementSuggestions);
            idealContent.innerHTML = marked.parse(data.idealResume);
            optimizedContent.innerHTML = marked.parse(data.optimizedResume);
            
            // Make sure the results section is visible
            resultsSection.classList.remove('d-none');
            resultsSection.style.display = 'block';
            resultsSection.setAttribute('data-aos', 'fade-up');
            resultsSection.setAttribute('data-aos-duration', '800');
            
            // Force AOS to refresh and apply animations
            setTimeout(() => {
                AOS.refresh();
                console.log('Results section should now be visible');
                
                // Initialize Bootstrap tabs to ensure they're properly activated
                const tabElements = document.querySelectorAll('[data-bs-toggle="tab"]');
                tabElements.forEach(tabEl => {
                    const tab = new bootstrap.Tab(tabEl);
                    
                    // Add event listeners to log tab changes for debugging
                    tabEl.addEventListener('shown.bs.tab', event => {
                        console.log(`Tab switched to: ${event.target.id}`);
                    });
                });
                
                // Ensure all tab panes are properly initialized
                const tabPanes = document.querySelectorAll('.tab-pane');
                console.log(`Found ${tabPanes.length} tab panes`);
                
                // Log tab content for debugging
                console.log(`Comparison content length: ${document.getElementById('comparisonContent').innerHTML.length}`);
                console.log(`Improvements content length: ${document.getElementById('improvementsContent').innerHTML.length}`);
                console.log(`Ideal resume content length: ${document.getElementById('idealContent').innerHTML.length}`);
            }, 100);
            
            // Apply enhanced formatting
            enhanceFormatting();
            
            // Scroll to results with smooth animation
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 200);
            
        } catch (error) {
            console.error('Error:', error);
            
            // Enhanced error message
            const errorBox = document.createElement('div');
            errorBox.className = 'alert alert-danger alert-dismissible fade show mt-3';
            errorBox.setAttribute('role', 'alert');
            errorBox.innerHTML = `
                <strong><i class="fas fa-exclamation-triangle me-2"></i>Analysis Error</strong>
                <p class="mb-0 mt-2">${error.message}</p>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            resumeForm.appendChild(errorBox);
            
            // Auto-dismiss after 8 seconds
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(errorBox);
                bsAlert.close();
            }, 8000);
            
        } finally {
            // Only remove loading overlay when we're actually finished (either with success or error)
            // This ensures the loading UI stays visible throughout the process
            if (loadingOverlay && loadingOverlay.parentNode) {
                // Add a small delay before removing the overlay to ensure results are ready
                setTimeout(() => {
                    document.body.removeChild(loadingOverlay);
                    
                    // Force a reflow to ensure results are visible
                    if (!resultsSection.classList.contains('d-none')) {
                        resultsSection.style.display = 'block';
                        resultsSection.style.opacity = '1';
                        console.log('Force showing results after overlay removal');
                        
                        // Scroll to results again to make sure they're visible
                        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 200);
            }
            
            // Reset button state immediately when results are received
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = originalButtonText;
            analyzeButton.classList.remove('loading');
            console.log('Button state reset to normal');
            
            // Log completion to console for debugging
            console.log('Analysis process completed (success or error)');
        }
    });
    
    // Enhanced formatting and visualization
    function enhanceFormatting() {
        // Highlight similarity scores
        highlightScores();
        
        // Add copy buttons to code blocks if any
        document.querySelectorAll('.markdown-content pre').forEach(block => {
            const button = document.createElement('button');
            button.className = 'btn btn-sm btn-outline-secondary copy-btn';
            button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            button.addEventListener('click', () => {
                const code = block.querySelector('code').textContent;
                navigator.clipboard.writeText(code).then(() => {
                    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        button.innerHTML = '<i class="fas fa-copy"></i> Copy';
                    }, 2000);
                });
            });
            block.appendChild(button);
        });
        
        // Make section headings more prominent
        document.querySelectorAll('.markdown-content h2, .markdown-content h3').forEach(heading => {
            heading.className += ' mt-4 mb-3 fw-bold border-bottom pb-2';
        });
        
        // Add checkmarks to bullet points for better readability
        document.querySelectorAll('.markdown-content ul li').forEach(item => {
            item.innerHTML = '<i class="fas fa-check text-success me-2"></i>' + item.innerHTML;
        });
    }
    
    // Enhanced score highlighting with better visualization
    function highlightScores() {
        const scoreRegex = /Similarity Score:\s*(\d+)%/g;
        
        if (comparisonContent.innerHTML) {
            comparisonContent.innerHTML = comparisonContent.innerHTML.replace(
                scoreRegex, 
                (match, score) => {
                    const scoreNum = parseInt(score);
                    let scoreClass = 'low-score';
                    let icon = 'exclamation-circle';
                    let scoreText = 'Needs Improvement';
                    
                    if (scoreNum >= 80) {
                        scoreClass = 'high-score';
                        icon = 'check-circle';
                        scoreText = 'Excellent Match';
                    } else if (scoreNum >= 60) {
                        scoreClass = 'medium-score';
                        icon = 'info-circle';
                        scoreText = 'Good Match';
                    }
                    
                    // Create enhanced score display
                    return `
                    <div class="score-container p-3 mb-4 rounded">
                        <div class="d-flex align-items-center">
                            <span class="fs-5 me-2">Similarity Score:</span>
                            <span class="${scoreClass} fs-4">${score}%</span>
                            <i class="fas fa-${icon} ${scoreClass} ms-2"></i>
                            <span class="ms-2 badge bg-light text-dark">${scoreText}</span>
                        </div>
                        <div class="progress mt-2" style="height: 10px;">
                            <div class="progress-bar ${scoreClass === 'high-score' ? 'bg-success' : scoreClass === 'medium-score' ? 'bg-warning' : 'bg-danger'}" 
                                role="progressbar" style="width: ${score}%" 
                                aria-valuenow="${score}" aria-valuemin="0" aria-valuemax="100">
                            </div>
                        </div>
                    </div>`;
                }
            );
        }
    }
    
    // Tooltips and help text for better UX
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(tooltipTriggerEl => {
        new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Tab switching enhancements
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', event => {
            const targetId = event.target.getAttribute('data-bs-target').substring(1);
            document.getElementById(targetId).setAttribute('data-aos', 'fade-in');
            AOS.refresh();
        });
    });
});
