// Tab functionality
function openTab(evt, tabName) {
    // Hide all tab content
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove("active");
    }

    // Remove active class from all tab buttons
    const tabButtons = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    // Show the selected tab content and mark button as active
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Toggle details row in batch results
function toggleDetails(detailsId) {
    const detailsRow = document.getElementById(detailsId);
    if (detailsRow.style.display === "table-row") {
        detailsRow.style.display = "none";
    } else {
        // Hide all other detail rows first
        const allDetails = document.querySelectorAll('.details-row');
        allDetails.forEach(row => {
            row.style.display = "none";
        });
        
        // Show the selected details
        detailsRow.style.display = "table-row";
    }
}

// Export batch results to CSV
function exportResults() {
    // Get all rows from the results table
    const table = document.querySelector('.results-table table');
    const rows = table.querySelectorAll('tbody tr:not(.details-row)');
    
    // Create CSV content
    let csvContent = "Transaction,Amount,Type,Prediction,Fraud Probability\n";
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 5) {
            const transaction = cells[0].textContent.trim();
            const amount = cells[1].textContent.trim();
            const type = cells[2].textContent.trim();
            const prediction = cells[3].textContent.trim();
            const probability = cells[4].textContent.trim().split('%')[0].trim();
            
            csvContent += `${transaction},${amount},${type},${prediction},${probability}%\n`;
        }
    });
    
    // Create download link
    const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "fraud_detection_results.csv");
    document.body.appendChild(link);
    
    // Trigger download
    link.click();
    document.body.removeChild(link);
}

// File upload validation
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const filePath = this.value;
            const allowedExtensions = /(\.csv)$/i;
            
            if (!allowedExtensions.exec(filePath)) {
                alert('Please upload a CSV file only');
                this.value = '';
                return false;
            }
        });
    }
    
    // Form validation for single transaction
    const singleForm = document.querySelector('#single form');
    if (singleForm) {
        singleForm.addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[type="number"]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (input.value === '') {
                    isValid = false;
                    input.style.borderColor = 'red';
                } else {
                    input.style.borderColor = '';
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all fields');
            }
        });
    }
    // Add this to your existing DOMContentLoaded event listener
    document.addEventListener('DOMContentLoaded', function() {
        // Set meter fill widths
        const meterFills = document.querySelectorAll('.meter-fill');
        meterFills.forEach(fill => {
            const parent = fill.parentElement;
            const widthPercentage = parseFloat(fill.getAttribute('data-percentage') || 0);
            const parentWidth = parent.offsetWidth;
            fill.style.width = (parentWidth * widthPercentage / 100) + 'px';
        });
        
        // Set mini-meter fill widths
        const miniMeterFills = document.querySelectorAll('.mini-meter-fill');
        miniMeterFills.forEach(fill => {
            const parent = fill.parentElement;
            const widthPercentage = parseFloat(fill.getAttribute('data-percentage') || 0);
            const parentWidth = parent.offsetWidth;
            fill.style.width = (parentWidth * widthPercentage / 100) + 'px';
        });
    });
});
