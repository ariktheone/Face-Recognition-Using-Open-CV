document.addEventListener("DOMContentLoaded", function() {
    // Add event listener to the image upload input field
    let imageUpload = document.getElementById("imageUpload");
    let uploadButton = document.getElementById("uploadButton");

    uploadButton.addEventListener("click", function() {
        let file = imageUpload.files[0];
        let imageName = document.getElementById("imageName").value;
        
        // Check if a file is selected
        if (file) {
            // Validate file type
            let fileType = file.type.split('/')[1];
            if (['png', 'jpg', 'jpeg'].includes(fileType)) {
                // Create form data and append the file and name
                let formData = new FormData();
                formData.append("image", file);
                formData.append("name", imageName);
                
                // Send the form data to the server using fetch
                fetch("/upload_image", {
                    method: "POST",
                    body: formData
                })
                .then(response => {
                    if (response.ok) {
                        // Clear input fields on successful upload
                        imageUpload.value = null;
                        document.getElementById("imageName").value = "";
                        return response.text();
                    }
                    throw new Error("Network response was not ok.");
                })
                .then(result => {
                    console.log(result);
                    // Show alert message on successful upload
                    alert("Image uploaded successfully!");
                })
                .catch(error => {
                    console.error("There was a problem with the fetch operation:", error);
                });
            } else {
                alert("Unsupported file type. Only PNG, JPG, and JPEG files are allowed.");
                // Clear input fields if file type is not supported
                imageUpload.value = null;
                document.getElementById("imageName").value = "";
            }
        }
    });
});
