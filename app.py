from typing import Any
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


URL_DATA = 'https://github.com/alexvaroz/data_science_alem_do_basico/raw/master/netflix_titles.csv.zip'


st.set_page_config(
        "RecSys App, by AVR",
        initial_sidebar_state="expanded",
        layout="wide",
    )


@st.cache
def load_data():
    data = pd.read_csv(URL_DATA)
    return data


def get_cosine_sim(data: Any):
    tfidf = TfidfVectorizer(stop_words='english')
    data["description"] = data["description"].fillna('')
    tfidf_matriz = tfidf.fit_transform(data['description'])
    return linear_kernel(tfidf_matriz)


def load_reverserd_index(data: Any):
    return pd.Series(data.index, index=data.title).drop_duplicates()


def get_recommendations_by_title(title, cosine_sim, data, reversed_index):
    idx = reversed_index[title]
    sim_scores = enumerate(list(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    result_indexes = [i[0] for i in sim_scores]
    return data[['title', 'director', 'country', 'listed_in',  'description']].loc[result_indexes]


df = load_data()
cosine_sim = get_cosine_sim(df)
reversed_index = load_reverserd_index(df)


# SIDE BAR
movies_list = df['title'].unique()
st.sidebar.header("Filme")
movie_selected = st.sidebar.selectbox('Selecione um filme', movies_list)


# MAIN
st.title(movie_selected)
st.markdown(df[df['title'] == movie_selected]['description'].values[0])
st.markdown(df[df['title'] == movie_selected]['director'].values[0])
# st.write(get_recommendations_by_title(movie_selected, cosine_sim, df, reversed_index))
data = get_recommendations_by_title(movie_selected, cosine_sim, df, reversed_index)

gb = GridOptionsBuilder.from_dataframe(data)

gb.configure_pagination()
gb.configure_side_bar()
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
gb.configure_selection(selection_mode="single", use_checkbox=True)
gridOptions = gb.build()


data = AgGrid(data, gridOptions=gridOptions, enable_enterprise_modules=True,
       allow_unsafe_jscode=True, update_mode=GridUpdateMode.SELECTION_CHANGED)


movie_selected = st.write(data["selected_rows"][0]['title'])
st.write(movie_selected)
