from flask import Flask, request, render_template, send_file
from PIL import Image
import io
import base64
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file.stream)
        
        # Perform object detection
        results = model(image)
        
        # Save the result image to a file
        result_image_path = 'static/result.png'
        results[0].save(filename=result_image_path)
        
        # Convert the result image to a base64 string for display
        with open(result_image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return render_template('result.html', img_data=img_base64, result_image_path=result_image_path)
    return render_template('upload.html')

@app.route('/download')
def download_file():
    # Send the result image as a downloadable file
    return send_file('static/result.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)