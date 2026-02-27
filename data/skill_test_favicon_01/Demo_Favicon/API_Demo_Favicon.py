from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder="assets", static_url_path="/assets")


@app.get("/")
def home():
    return {"project": "Demo Favicon", "status": "ok"}


@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.get("/favicon.ico")
def favicon():
    favicon_dir = BASE_DIR / "assets" / "img"
    favicon_path = favicon_dir / "favicon.png"
    if favicon_path.exists():
        return send_from_directory(str(favicon_dir), "favicon.png", mimetype="image/png")
    return Response(status=204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
