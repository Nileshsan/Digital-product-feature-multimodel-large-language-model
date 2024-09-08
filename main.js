document.getElementById('imageUploader').addEventListener('change', function(event) {
  const files = event.target.files;
  const previewDiv = document.getElementById('preview');
  previewDiv.innerHTML = '';  // Clear previous previews

  // Loop through all selected files
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const reader = new FileReader();

    reader.onload = function(e) {
      // Create image element and append to the preview section
      const imgElement = document.createElement('img');
      imgElement.src = e.target.result;
      imgElement.style.maxWidth = '200px';  // Limit image size
      imgElement.style.margin = '10px';
      previewDiv.appendChild(imgElement);
    };

    reader.readAsDataURL(file);  // Convert image to base64 URL for preview
  }
});

document.getElementById('imageForm').addEventListener('submit', function(event) {
  event.preventDefault();

  const context = document.getElementById('context').value;
  const images = document.getElementById('imageUploader').files;
  const formData = new FormData();

  // Append context and files to the FormData
  formData.append('context', context);
  for (let i = 0; i < images.length; i++) {
    formData.append('images', images[i]);
  }

  // Send FormData via fetch to the backend
  fetch('/api/generate-test-cases', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);  // Handle response
    document.getElementById('output').innerText = data.test_cases;
  })
  .catch(error => console.error('Error:', error));
});
