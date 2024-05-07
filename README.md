# WikiKnowlege
App for article's sentiment analisys, plagiarism detection and compliance with the encyclopedic style

Приложение для анализа нейтральности, корректности и стиля написания статей, и отсутствие плагиата.

__*Файлы:*__

app.py - логика приложения для Streamlit

func.py - функции, реализующие применение нейросетей для решения поставленных задач

__*Данные:*__
Предоставлен датасет из 29259 статей, размешенных на ресурсе Кейсодержателя.

__*Решение:*__
__Проверка энциклопедичности и соответствия Wiki-разметеке.__
1) Дрбавлено 20370 статей неэнциклопедического стиля из датасета Dezzpil/otus-fw (источник: https://huggingface.co/datasets/Dezzpil/otus-fw). 90% трейн часть, 10% валидация
2) Дообучена модель bert-base-uncased (2 эпохи, accuracy 99,7%) для классификации на 0 ("Текст не соответствует энциклопедическому стилю") и 1 ("Текст соответствует энциклопедическому стилю")

__Исправление ошибок:__
Применение открытой модели YandexSpeller - исправляет орфографические и пунктуационные ошибки в русском языке. Возвращает слова, в которых предположительно допущены ощибки и текст с исправлениями.

__Оценка нейтральности статьи:__
Применение предобученной нейросети cointegrated/rubert-tiny-sentiment-balanced (источник: https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced) - классифицирует текст на негативный, нейтральный и позитивный.

__Проверка на плагиат:__
Применение открытой библиотеки для поиска плагиата pyfiglet - возвращает ссылку на первоисточник и процент совпадения текста
