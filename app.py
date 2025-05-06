from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import logging
from gliner import GLiNER
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Загрузка списков из txt файлов
def load_txt_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def is_valid_furniture_term(term, dictionaries):
    term_lower = term.lower()
    return any(
        furniture_word in term_lower
        for furniture_word in dictionaries['furniture_labels']
    )

# Загрузка всех словарей
def load_dictionaries():
    return {
        'furniture_labels': load_txt_file('furniture_labels'),
        'blacklist': load_txt_file('blacklist'),
        'standalone_blacklist': load_txt_file('standalone_blacklist')
    }


dictionaries = load_dictionaries()

# Инициализация модели GLiNER
try:
    gliner_model = GLiNER.from_pretrained("urchade/gliner_base")
    logger.info("GLiNER model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load GLiNER model: {e}")
    gliner_model = None


# Получение текста с URL
def fetch_page_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Удаление ненужных элементов
        for tag in ['script', 'style', 'nav', 'footer', 'header']:
            for element in soup.find_all(tag):
                element.decompose()

        return ' '.join(p.get_text() for p in soup.find_all('p')), None
    except Exception as e:
        return None, str(e)


# Извлечение мебели
def extract_furniture(text):
    if not gliner_model:
        return [], "Model not loaded"

    try:
        entities = gliner_model.predict_entities(
            text,
            dictionaries['furniture_labels'],
            threshold=0.3
        )

        items = set()
        for entity in entities:
            item = entity['text'].strip()
            if (len(item) >= 3 and
                    item.lower() not in dictionaries['blacklist'] and
                    not any(word.lower() == item.lower() for word in dictionaries['standalone_blacklist'])
                    and is_valid_furniture_term(item, dictionaries)):
                items.add(item)

        return sorted(items), None
    except Exception as e:
        return [], str(e)


@app.route('/', methods=['GET', 'POST'])
def index():
    items, error = [], None
    url = request.form.get('url', '')

    if request.method == 'POST' and url:
        text, err = fetch_page_text(url)
        if err:
            error = err
        else:
            items, error = extract_furniture(text)

    return render_template('index.html', items=items, error=error, url=url)


if __name__ == '__main__':
    app.run(debug=True)