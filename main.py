from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import operator
from io import BytesIO
import requests
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
Bootstrap(app)

# Default Values
file_name = 'https://images.unsplash.com/photo-1534759926787-89fa60f35848?ixid=MnwxMjA3fDB8MHxzZWFyY2h8OHx8Y29sb3JmdWx8ZW58MHx8MHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
granularity = 1
sorted_hex_dict = {}


@app.route('/')
def start():
    return redirect(url_for('process_image'))


@app.route('/home', methods=["GET", "POST"])
def home():
    global file_name, granularity
    return render_template("index.html", hex_dict=sorted_hex_dict, image_loc=file_name)


@app.route('/process_image')
def process_image():
    global sorted_hex_dict

    if "http" in file_name:
        image_url = requests.get(file_name)
        img = Image.open(BytesIO(image_url.content))
    else:
        img = Image.open(file_name)

    resized_img = img.resize((300, 300))
    img_array = np.array(resized_img)

    colour_vector = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    clt = KMeans(n_clusters=10*granularity)
    clt.fit(colour_vector)
    numlabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numlabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    final_info = clt.cluster_centers_.astype(np.uint8)
    hex_dict = {}
    for item in range(len(final_info)):
        r = final_info[item][0]
        g = final_info[item][1]
        b = final_info[item][2]
        hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        percentage = round(hist[item] * 100, 1)
        hex_dict[hex_code] = percentage

    sorted_hex_dict = dict(list(sorted(hex_dict.items(), key=operator.itemgetter(1), reverse=True))[0:10])
    return redirect(url_for('home'))


@app.route('/upload_url', methods=["GET", "POST"])
def upload_url():
    global file_name

    if request.method == "POST":
        file_name = request.form.get("image_url")
        print(file_name)
        return redirect(url_for('process_image'))

    return redirect(url_for('process_image'))


@app.route('/update_granularity', methods=["GET", "POST"])
def update_granularity():
    global granularity

    if request.method == "POST":
        granularity = int(request.form.get("granularity"))
        print(granularity)
        return redirect(url_for('process_image'))

    return redirect(url_for('process_image'))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
