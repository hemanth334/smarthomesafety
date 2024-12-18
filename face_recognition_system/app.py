from flask import Flask, render_template
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def index():
    unknown_faces_dir = "dataset/unknown_faces"
    images = [
        {
            "path": f"/static/unknown_faces/{file}",
            "timestamp": datetime.fromtimestamp(os.path.getmtime(os.path.join(unknown_faces_dir, file))).strftime('%Y-%m-%d %H:%M:%S')
        }
        for file in os.listdir(unknown_faces_dir)
    ]
    return render_template("index.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)
