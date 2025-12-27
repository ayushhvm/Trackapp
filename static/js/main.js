// Image preview for file inputs
function setupImagePreview(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    
    if (input && preview) {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        });
    }
}

// Multiple image preview
function setupMultipleImagePreview(inputId, previewContainerId) {
    const input = document.getElementById(inputId);
    const container = document.getElementById(previewContainerId);
    
    if (input && container) {
        input.addEventListener('change', function(e) {
            container.innerHTML = '';
            const files = Array.from(e.target.files);
            
            files.forEach((file, index) => {
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'image-preview';
                        img.style.width = '150px';
                        img.style.height = '150px';
                        img.style.objectFit = 'cover';
                        img.style.margin = '8px';
                        img.style.borderRadius = '4px';
                        container.appendChild(img);
                    };
                    reader.readAsDataURL(file);
                }
            });
        });
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Setup attendance image preview
    setupImagePreview('attendance-image-input', 'attendance-image-preview');
    
    // Setup multiple image preview for student registration
    setupMultipleImagePreview('id_images', 'image-preview-container');
});

// Form validation feedback
function showFormError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (field) {
        field.style.borderBottom = '2px solid #f44336';
        const errorDiv = document.createElement('div');
        errorDiv.className = 'help-text';
        errorDiv.style.color = '#f44336';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    }
}

// Loading state for forms
function setFormLoading(formId, loading) {
    const form = document.getElementById(formId);
    if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = loading;
            if (loading) {
                submitBtn.innerHTML = '<span class="spinner"></span> Processing...';
            }
        }
    }
}

