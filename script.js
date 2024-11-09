// Handle form submission
document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission

    const realImageInput = document.getElementById("real_image");
    const sampleImageInput = document.getElementById("sample_image");

    const formData = new FormData();
    
    // Check if both files are selected
    if (!realImageInput.files.length || !sampleImageInput.files.length) {
        alert("Please select both images.");
        return;
    }

    // Append files to FormData
    formData.append("real_image", realImageInput.files[0]);
    formData.append("sample_image", sampleImageInput.files[0]);

    // Display loading message
    document.getElementById("loading").style.display = "block";
    document.getElementById("fileNames").innerHTML = `
        Real Image: ${realImageInput.files[0].name}<br>
        Sample Image: ${sampleImageInput.files[0].name}
    `;
    document.getElementById("responseMessage").innerHTML = '';

    // Make the POST request to Flask backend
    fetch("http://127.0.0.1:5000/check-currency", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading message
        document.getElementById("loading").style.display = "none";
        
        // Display response message from backend
        document.getElementById("responseMessage").innerHTML = `<strong>${data.message}</strong>`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("loading").style.display = "none";
        document.getElementById("responseMessage").innerHTML = "An error occurred. Please try again.";
    });
});
