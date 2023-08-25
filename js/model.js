async function classifyImage() {
    const imageElement = document.getElementById('imageInput');
    const resultElement = document.getElementById('result');
    
    // Load the MobileNet model.
    const model = await mobilenet.load();
    
    // Get the selected image file.
    const imageFile = imageElement.files[0];
    
    if (imageFile) {
        const image = document.createElement('img');
        const reader = new FileReader();
        
        reader.onload = async (e) => {
            image.src = e.target.result;
            await image.onload;
            
            // Make a prediction.
            const predictions = await model.classify(image);
            
            // Display the prediction results.
            resultElement.innerHTML = `Prediction: ${predictions[0].className} (Probability: ${predictions[0].probability.toFixed(2)})`;
        };
        
        reader.readAsDataURL(imageFile);
    } else {
        resultElement.innerHTML = 'Please select an image.';
    }
}
