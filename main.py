import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from audio import ImprovedAudioSteganography

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

DEFAULT_USERNAME = os.getenv("APP_USERNAME", "operator")
DEFAULT_PASSWORD = os.getenv("APP_PASSWORD", "terminal123")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_platform_status():
    return {
        "local_testing_mode": True,
        "storage_provider": "Local uploads",
        "metadata_provider": "Local test mode",
    }


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authenticated"):
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session["authenticated"] = True
            session["username"] = username
            return redirect(url_for("index"))

        flash("invalid credentials", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        platform_status=get_platform_status(),
    )


@app.route("/encrypt", methods=["GET", "POST"])
@login_required
def encrypt():
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        instrument = request.form.get("instrument", "pad")
        timbre_mode = request.form.get("timbre_mode", "single")
        genre = request.form.get("genre", "pop")
        output_format = request.form.get("output_format", "wav").lower()
        seed_raw = request.form.get("seed", "").strip()

        if output_format not in ALLOWED_EXTENSIONS:
            output_format = "wav"

        if not message:
            flash("Enter a message before generating audio.", "error")
            return redirect(url_for("encrypt"))

        try:
            seed = int(seed_raw) if seed_raw else None
        except ValueError:
            flash("Seed must be a valid integer.", "error")
            return redirect(url_for("encrypt"))

        try:
            stego = ImprovedAudioSteganography()

            output_filename = f"hidden_message_{uuid.uuid4().hex}.{output_format}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            stego.embed_message(
                message,
                output_file=output_path,
                instrument=instrument,
                timbre_mode=timbre_mode,
                genre=genre,
                seed=seed,
            )

            platform_status = get_platform_status()
            download_url = None

            return render_template(
                "download.html",
                filename=output_filename,
                download_url=download_url,
                seed=stego.last_seed,
                storage_provider=platform_status["storage_provider"],
            )
        except Exception as exc:
            flash(f"Audio generation failed: {exc}", "error")
            return redirect(url_for("encrypt"))

    return render_template("encrypt.html", platform_status=get_platform_status())


@app.route("/decrypt", methods=["GET", "POST"])
@login_required
def decrypt():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Upload an audio file to extract a message.", "error")
            return redirect(url_for("decrypt"))

        file = request.files["file"]
        if file.filename == "":
            flash("Select an audio file before continuing.", "error")
            return redirect(url_for("decrypt"))

        if not allowed_file(file.filename):
            flash("Only WAV, MP3, FLAC, and OGG files are supported.", "error")
            return redirect(url_for("decrypt"))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            stego = ImprovedAudioSteganography()
            extracted_message = stego.extract_message(file_path)



            return render_template("result.html", message=extracted_message)
        except Exception as exc:
            flash(f"Message extraction failed: {exc}", "error")
            return redirect(url_for("decrypt"))

    return render_template("decrypt.html", platform_status=get_platform_status())


@app.route("/download/<filename>")
@login_required
def download(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], secure_filename(filename), as_attachment=True)


@app.route("/stream/<filename>")
@login_required
def stream(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], secure_filename(filename), as_attachment=False)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "platform": get_platform_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )





@app.errorhandler(404)
def page_not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(_error):
    return render_template("500.html"), 500


@app.before_request
def cleanup_old_files():
    try:
        current_time = time.time()
        for file_name in os.listdir(app.config["UPLOAD_FOLDER"]):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
            if not os.path.isfile(file_path):
                continue
            if current_time - os.path.getmtime(file_path) > 3600:
                os.remove(file_path)
    except Exception:
        pass


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
