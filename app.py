import os
import io
import uuid
import sys
import yaml
import traceback

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, flash
import flask
from PIL import Image
import numpy as np

from white_box_cartoonizer.cartoonize import WB_Cartoonize

app = Flask(__name__)

app.config['UPLOAD_FOLDER_IMAGES'] = 'static/uploaded_images'
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'

app.config['OPTS'] = opts

wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):

    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                img = flask.request.files["image"].read()
                
                image = convert_bytes_to_image(img)

                img_name = str(uuid.uuid4())
                
                cartoon_image = wb_cartoonizer.infer(image)
                
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
                
                return render_template("index_cartoonized.html", cartoonized_image=cartoonized_img_name)

        except Exception:
            print(traceback.print_exc())
            flash("Our server hiccuped :/ Please upload another file! :)")
            return render_template("index_cartoonized.html")
    else:
        return render_template("index_cartoonized.html")

if __name__ == "__main__":
    app.run(debug=True)