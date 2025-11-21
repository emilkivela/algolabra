import io
import base64
import os
from PIL import Image

def convert_pmg(image_path):
    img = Image.open(image_path)
    file_name = os.path.basename(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    file = {"filename" : file_name, "base64" : encoded}
    return file
