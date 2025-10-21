document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const form = document.getElementById('predictionForm');
    const resultsDiv = document.getElementById('results');
    const predictionScore = document.getElementById('predictionScore');
    const loader = document.getElementById('loader');
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Animate form elements on load
    animateFormElements();
    
    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Disable submit button and show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = 'Predicting...';
        
        // Show loader with fade-in animation
        loader.style.display = 'block';
        loader.style.opacity = '0';
        setTimeout(() => { loader.style.opacity = '1'; }, 10);
        
        // Hide results with fade-out animation
        resultsDiv.style.opacity = '0';
        setTimeout(() => { resultsDiv.style.display = 'none'; }, 300);
        
        try {
            // Get form data
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });
            
            // Simulate API call with timeout for demo
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Send request to server
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            
            // Animate the score counter
            animateValue(predictionScore, 0, result.prediction.toFixed(1), 1500);
            
            // Show results with fade-in animation
            resultsDiv.style.display = 'block';
            resultsDiv.style.opacity = '0';
            setTimeout(() => { resultsDiv.style.opacity = '1'; }, 10);
            
            // Add celebration effect for high scores
            if (result.prediction > 80) {
                celebrate();
            }
            
            // Scroll to results smoothly
            setTimeout(() => {
                resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 500);
            
        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred while making the prediction. Please try again.');
        } finally {
            // Hide loader with fade-out animation
            loader.style.opacity = '0';
            setTimeout(() => { 
                loader.style.display = 'none';
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Predict Math Score <span>→</span>';
            }, 300);
        }
    });
    
    // Animate value counter
    function animateValue(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = Math.floor(progress * (end - start) + start);
            element.textContent = value.toFixed(1);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
    
    // Show error message with animation
    function showError(message) {
        // Remove any existing error messages
        const existingError = document.querySelector('.error-message');
        if (existingError) existingError.remove();
        
        // Create and show error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Style the error message
        errorDiv.style.cssText = `
            background: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            text-align: center;
            animation: slideDown 0.3s ease-out;
            border-left: 4px solid #ef5350;
        `;
        
        // Insert error message after the form
        form.parentNode.insertBefore(errorDiv, form.nextSibling);
        
        // Remove error message after 5 seconds
        setTimeout(() => {
            errorDiv.style.opacity = '0';
            setTimeout(() => errorDiv.remove(), 300);
        }, 5000);
    }
    
    // Add confetti effect for high scores
    function celebrate() {
        const colors = ['#4361ee', '#4cc9f0', '#f72585', '#7209b7', '#3a0ca3'];
        
        function createConfetti() {
            const confetti = document.createElement('div');
            confetti.style.cssText = `
                position: fixed;
                width: 10px;
                height: 10px;
                background: ${colors[Math.floor(Math.random() * colors.length)]};
                border-radius: 50%;
                top: -10px;
                left: ${Math.random() * 100}%;
                opacity: ${Math.random() * 0.5 + 0.5};
                transform: scale(${Math.random() * 0.5 + 0.5});
                animation: fall ${Math.random() * 3 + 2}s linear forwards;
                z-index: 1000;
                pointer-events: none;
            `;
            
            document.body.appendChild(confetti);
            
            // Remove confetti after animation
            setTimeout(() => {
                confetti.remove();
            }, 5000);
        }
        
        // Add confetti style
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fall {
                to {
                    transform: translateY(${window.innerHeight + 10}px) rotate(720deg);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
        
        // Create multiple confetti pieces
        for (let i = 0; i < 100; i++) {
            setTimeout(createConfetti, i * 50);
        }
    }
    
    // Animate form elements on page load
    function animateFormElements() {
        const formGroups = document.querySelectorAll('.form-group');
        formGroups.forEach((group, index) => {
            group.style.animation = `fadeIn 0.5s ease-out ${index * 0.1}s forwards`;
        });
        
        // Add animation for the submit button
        submitBtn.style.animation = 'fadeIn 0.5s ease-out 0.6s forwards';
        submitBtn.style.opacity = '0';
    }
    
    // Add input validation with visual feedback
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        // Add focus effect
        input.addEventListener('focus', function() {
            this.parentNode.classList.add('focused');
        });
        
        // Remove focus effect
        input.addEventListener('blur', function() {
            this.parentNode.classList.remove('focused');
        });
        
        // Add validation for number inputs
        if (input.type === 'number') {
            input.addEventListener('input', function() {
                const value = parseFloat(this.value);
                if (this.value && (value < 0 || value > 100)) {
                    this.setCustomValidity('Score must be between 0 and 100');
                    this.parentNode.classList.add('error');
                } else {
                    this.setCustomValidity('');
                    this.parentNode.classList.remove('error');
                }
                
                // Add shake animation if invalid
                if (this.validity.valid) {
                    this.style.borderColor = '#4CAF50';
                    setTimeout(() => {
                        this.style.borderColor = '#e9ecef';
                    }, 2000);
                } else {
                    this.style.animation = 'shake 0.5s';
                    setTimeout(() => {
                        this.style.animation = '';
                    }, 500);
                }
            });
        }
    });
    
    // Add shake animation for invalid inputs
    const shakeStyle = document.createElement('style');
    shakeStyle.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .form-group.focused label {
            color: var(--primary-color);
            transform: translateY(-2px);
        }
        
        .form-group.error input,
        .form-group.error select {
            border-color: #ef5350 !important;
            background-color: #ffebee;
        }
    `;
    document.head.appendChild(shakeStyle);
});
