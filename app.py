import os
import logging
from dotenv import load_dotenv

from flask import Flask
from flask_cors import CORS

from blueprints import blueprints


load_dotenv()
config = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)

    ALLOWED_EXTENSIONS = {'xlsx'}
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    for rule in app.url_map.iter_rules():
        app.logger.info("%s → %s", rule.rule, ",".join(rule.methods))

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    create_app().run(host='0.0.0.0', port=port, debug=debug_mode)