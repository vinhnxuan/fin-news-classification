import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article

st.set_page_config(page_title='FNSA Tool', page_icon='./images/newspaper.ico')
from libs.classifiers.llama2 import LLAMA2Classifier

if "model" not in st.session_state:
    finetuned_model_path = "./trained_weigths/"
    classifer = LLAMA2Classifier()
    classifer.merge_finetuned_model(finetuned_model_path)
    st.session_state.model = classifer


def fetch_top_news(topic):
    site = 'https://news.google.com/rss/search?q={}%20news&hl=en-US&gl=US&ceid=US%3Aen'.format(topic.lower())
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list

import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        color = "#808080"
        news_text_splits = news.title.text.split("-")
        if len(news_text_splits) > 2:
            news_text = " ".join(news_text_splits[0:-1])
        else:
            news_text = news_text_splits[0]
        print(news_text)
        result = st.session_state.model.predict(news_text)
        news_class = result[0]['generated_text'].split("=")[-1]
        img_path = "./images/neutral.png"
        print(news_class)
        if "positive" in news_class:
            color = "#008000"
            img_path = "./images/positive.png"
        elif "negative" in news_class:
            color = "#FF0000"
            img_path = "./images/negative.png"
        st.markdown(f'<h1 style="background-color:{color};font-size:24px;"><img src="data:image/png;base64,{img_to_bytes(img_path)}" class="img-fluid">({c})[ {news.title.text}]</h1>', unsafe_allow_html=True)
            
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        #fetch_news_poster(news_data.top_image)
        with st.expander(news.title.text):
            st.markdown(
                '''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                unsafe_allow_html=True)
            st.markdown("[Reat more at {}...]({})".format(news.source.text, news.link.text))
        st.success("Date of publication: " + news.pubDate.text)
        if c >= news_quantity:
            break


def run():
    st.title("Financial News Sentiment Analysis")
    image = Image.open('./images/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")
    category = ['--Select options--', 'Top financial news 🔥', 'Top market news 💙', 'Top economic news 🔍',  'Other news']
    cat_op = st.selectbox('Select', category)
    if cat_op == category[0]:
        st.warning('Select an option!!')
    elif cat_op == category[1]:
        st.subheader("✅ Here are the top financial news 🔥 for you")
        no_of_news = st.slider('Number of news:', min_value=5, max_value=25, step=1)
        news_list = fetch_top_news(topic = "financial")
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        st.subheader("✅ Here are the top market news 🔥 for you")
        no_of_news = st.slider('Number of news:', min_value=5, max_value=25, step=1)
        news_list = fetch_top_news(topic = "market")
        display_news(news_list, no_of_news)
    elif cat_op == category[3]:
        st.subheader("✅ Here are the top economic news 🔥 for you")
        no_of_news = st.slider('Number of news:', min_value=5, max_value=25, step=1)
        news_list = fetch_top_news(topic = "economic")
        display_news(news_list, no_of_news)

    elif cat_op == category[4]:
        user_topic = st.text_input("Name of topic 🔍")
        no_of_news = st.slider('Number of news:', min_value=5, max_value=15, step=1)

        if st.button("Search") and user_topic != '':
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_top_news(topic=user_topic_pr)
            if news_list:
                st.subheader("✅ Here are some {} news for you".format(user_topic))
                display_news(news_list, no_of_news)
            else:
                st.error("Error with fetching news {}".format(user_topic))
        else:
            st.warning("Enter topic name to search 🔍")


run()
