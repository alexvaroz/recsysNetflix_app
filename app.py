from typing import Any
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


URL_DATA = 'https://github.com/alexvaroz/data_science_alem_do_basico/raw/master/netflix_titles.csv.zip'
fields_content = ['director', 'country', 'listed_in', 'cast']

st.set_page_config(
        "RecSys App, by AVR",
        initial_sidebar_state="expanded",
        layout="wide",
    )


@st.cache
def load_data():
    data = pd.read_csv(URL_DATA)
    return data


def transform_name_field(name):
    if isinstance(name, list):
        return [str.lower(i.replace(" ", "")) for i in name]
    else:
        if isinstance(name, str):
            return str.lower(name.replace(" ", ""))
        else:
            return ''


def get_cosine_sim(data: Any):
    tfidf = TfidfVectorizer(stop_words='english')
    data["metadados_grouping"] = data["metadados_grouping"].fillna('')
    tfidf_matriz = tfidf.fit_transform(data['metadados_grouping'])
    return linear_kernel(tfidf_matriz)


def make_content_lst(data, field):
    data[field] = data[field].fillna('')
    data[field] = data.apply(lambda row: row[field].split(',') if row[field] != '' else [], axis=1)
    data[field] = data.apply(lambda row: transform_name_field(row[field]), axis=1)
    return data[field][0:2]


def criar_agrupamento_metadados(x, params):
    grouping_metadados = ' '
    params_fields = params.copy()
    if 'description' in params_fields:
        params_fields.remove('description')
    for param in params_fields:
        grouping_metadados = grouping_metadados + ' '.join(x[param]) + '  '
    return grouping_metadados


def create_grouping(data, params):
    if len(params) == 0:
        data['metadados_grouping'] = data['description']
    else:
        data['metadados_grouping'] = data.apply(lambda row: criar_agrupamento_metadados(row, params), axis=1)
        print(params)
        if 'description' in params:
            data['metadados_grouping'] = data['metadados_grouping'] + data['description']
    return data


def load_reverserd_index(data: Any):
    return pd.Series(data.index, index=data.title).drop_duplicates()


def get_recommendations_by_title(title, cosine_sim, data, reversed_index):
    idx = reversed_index[title]
    sim_scores = enumerate(list(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    result_indexes = [i[0] for i in sim_scores]
    return data[['title', 'director', 'cast', 'listed_in']].loc[result_indexes]


df = load_data()
df_lst = df.copy()
for field in fields_content:
    make_content_lst(df_lst, field)


# SIDE BAR
movies_list = df['title'].unique()
st.sidebar.header("Sistemas de Recomendação Baseado no Conteúdo")
st.sidebar.write("Trata-se de um sistema de recomendação baseado no conteúdo (*content based*) "
                 "desenvolvido para verificar a influência dos diversos campos no cálculo da proximidade dos itens."
                 "É utilizado um catálogo da NetFlix e é possível verificar as várias combinações entre os atributos "
                 "disponíveis na base de dados. ")


attributes_selection = st.sidebar.multiselect("Escolha os atributos a serem considerados",
                                              ['director', 'country', 'listed_in', 'cast', 'description'])
movie_selected = st.sidebar.selectbox('Selecione um filme', movies_list)


# MAIN
st.title(movie_selected)


def get_info(selected):
    st.markdown(df[df['title'] == selected]['description'].values[0])
    st.markdown("**Director:** {}".format(df[df['title'] == selected]['director'].values[0]))
    st.markdown("**Cast:** {}".format(df[df['title'] == selected]['cast'].values[0]))
    st.markdown("**Genre:** {}".format(df[df['title'] == selected]['listed_in'].values[0]))


params = attributes_selection
df_lst = create_grouping(df_lst, params)
cosine_sim = get_cosine_sim(df_lst)
reversed_index = load_reverserd_index(df)

get_info(movie_selected)
data = get_recommendations_by_title(movie_selected, cosine_sim, df, reversed_index)

gb = GridOptionsBuilder.from_dataframe(data)

gb.configure_pagination()
gb.configure_side_bar()
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
gb.configure_selection(selection_mode="single", use_checkbox=True)
gridOptions = gb.build()


data = AgGrid(data, gridOptions=gridOptions, enable_enterprise_modules=True,
              allow_unsafe_jscode=True, update_mode=GridUpdateMode.SELECTION_CHANGED)

if data["selected_rows"]:
    movie_selected = data["selected_rows"][0]['title']
    st.write("**{}**".format(data["selected_rows"][0]['title']))
    get_info(movie_selected)
