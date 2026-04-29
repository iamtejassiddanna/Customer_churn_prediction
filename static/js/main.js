document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('span');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultPanel = document.getElementById('result-panel');
    
    const churnStatus = document.getElementById('churn-status');
    const statusIcon = document.getElementById('status-icon');
    const predictionText = document.getElementById('prediction-text');
    
    const probFill = document.getElementById('prob-fill');
    const probValue = document.getElementById('prob-value');
    const insightText = document.getElementById('insight-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading state
        btnText.style.display = 'none';
        btnSpinner.style.display = 'block';
        predictBtn.disabled = true;
        resultPanel.style.display = 'none';

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                // Update UI based on result
                const isChurn = result.prediction === 'Yes';
                
                churnStatus.className = `churn-status ${isChurn ? 'churn-yes' : 'churn-no'}`;
                statusIcon.textContent = isChurn ? '⚠️' : '✅';
                predictionText.textContent = isChurn ? 'High Risk of Churn' : 'Customer is likely to stay';
                
                const probNumber = parseFloat(result.probability);
                probFill.style.width = '0%'; // Reset for animation
                
                setTimeout(() => {
                    probFill.style.width = result.probability;
                    probFill.style.background = isChurn 
                        ? 'linear-gradient(90deg, #f59e0b, #ef4444)' 
                        : 'linear-gradient(90deg, #10b981, #3b82f6)';
                }, 100);
                
                probValue.textContent = result.probability;
                
                if (isChurn) {
                    insightText.textContent = "This customer profile matches historical patterns of users who cancelled their service. Consider offering retention incentives or reaching out to address potential concerns.";
                } else {
                    insightText.textContent = "This customer shows strong retention signals. Maintain current service quality to ensure continued satisfaction.";
                }

                resultPanel.style.display = 'block';
                
                // Scroll to result softly
                resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert('Error predicting churn: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while connecting to the prediction server.');
        } finally {
            // Revert loading state
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
            predictBtn.disabled = false;
        }
    });

    // Add some interactivity: automatically update TotalCharges if MonthlyCharges or tenure changes
    const tenureInput = document.getElementById('tenure');
    const monthlyInput = document.getElementById('MonthlyCharges');
    const totalInput = document.getElementById('TotalCharges');

    const updateTotal = () => {
        const tenure = parseFloat(tenureInput.value) || 0;
        const monthly = parseFloat(monthlyInput.value) || 0;
        // Simple estimation for UI convenience, users can still overwrite it
        totalInput.value = (tenure * monthly).toFixed(2);
    };

    // Update total only when user changes tenure or monthly
    tenureInput.addEventListener('change', updateTotal);
    monthlyInput.addEventListener('change', updateTotal);
});
