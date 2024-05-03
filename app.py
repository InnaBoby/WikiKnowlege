import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from pyaspeller import YandexSpeller
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from func import *

#os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

#модели для проверки энциклопедичности
tokenizer_enciclopedic = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#декоратор для загрузки модели в кэш
@st.cache_resource
def load_model_enciclopedic():
    model_enciclopedic = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    if torch.cuda.is_available():
        model_enciclopedic.cuda()
    return model_enciclopedic

model_enciclopedic = load_model_enciclopedic()

prompt = "Учитывая следующий текст, определите, написан ли он в энциклопедическом стиле:\n\nТекст:"
prompt_wiki = "Учитывая следующий текст, определите, соответствует ли он MediaWiki разметке:\n\nТекст:"


#модели для проверки нейтральности
tokenizer_neutrality = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
@st.cache_resource
def load_model_neutrality():
    model_neutrality = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
    if torch.cuda.is_available():
        model_neutrality.cuda()
    return model_neutrality

model_neutrality =  load_model_neutrality()


#Загрузка и очистка данных

data = st.text_area('Введите текcт', 'text input')
clean_data = clean_up(str(data))

option = st.selectbox('Выберите проверку',
    ('', 'Энциклопедичность', 'Соответствие Wiki разметке', 'Исправление ошибок', 'Нейтральность', 'Плагиат'))

if option == 'Энциклопедичность':
    enciclopedic=enciclopedic(data, tokenizer_enciclopedic, model_enciclopedic, prompt)
    st.write(enciclopedic)

elif option == 'Соответствие Wiki разметке':
    wiki = MediaWiki_text_style(data, tokenizer_enciclopedic, model_enciclopedic, prompt_wiki)
    st.write(wiki)

elif option == 'Исправление ошибок':
    speller = YandexSpeller()
    data_correct = speller.spelled(clean_data)
    if clean_data == data_correct:
        st.write('Нет ошибок')
    else:
        st.write('Статья содержит ошибки')


elif option == 'Нейтральность':
    label = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'label')
    score = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'score')
    proba = get_sentiment(clean_data, tokenizer_neutrality, model_neutrality, 'proba')
    st.write(label, ' на ', score)


elif option == 'Плагиат':
    plagiarism = plagiarism(clean_data)
    st.write(plagiarism)

else:
    st.write('Не выбрана ни одна проверка!!')



