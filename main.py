#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[24]:


import math

import sklearn 
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
countries.dtypes


# In[7]:


#Colando o separador decimal
countries = pd.read_csv('./countries.csv', decimal = ',')
countries.head(5)


# In[8]:


#Removendo espaçõs nas colunas 'Country' e 'Region'
countries = countries.apply(lambda x: x.str.rstrip() if x.dtype == 'object' else x)
countries['Country'][0]


# In[14]:


info = pd.DataFrame({'dtype': countries.dtypes,
                    'unique_vals': countries.nunique(),
                    'missing%': (countries.isna().sum() / countries.shape[0]) * 100})

info.T


# In[16]:


#Preenchendo dados ausentes com a média da região
countries_fill = countries.copy()

# Obtendo nome da coluna
numeric_cols = countries_fill._get_numeric_data().columns.tolist()

#preenchendo nan com valores médios de groupby por região
for col in numeric_cols:
    countries_fill[col] = countries_fill.groupby('Region')[col].apply(lambda x: x.fillna(x.mean()))

countries_fill.head(5)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[17]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return sorted(countries_fill['Region'].unique())

q1() 


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[29]:


def q2():
    # Retorne aqui o resultado da questão 2.
    # transformando Pop_density em um array do Numpy
    pop_density = countries_fill['Pop. Density (per sq. mi.)'].to_numpy()
    pop_density = pop_density.reshape((-1, 1))
    
    #KBins & fitting do Sklean
    kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy = 'quantile')
    kbins_popd = kbins.fit(pop_density.tolist())
    
    #percentil 90
    percentil_90 = kbins_popd.bin_edges_[0][9]
    
    #slice dataset > p90
    countries_above_p90 = countries_fill[countries_fill['Pop. Density (per sq. mi.)'] > percentil_90]
    return int(countries_above_p90['Country'].count())

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[33]:


climate = countries_fill[['Climate']].to_numpy().reshape((-1,1))
region = countries_fill[['Region']].to_numpy().reshape(-1,1)

#Onehotencoder
label_encoder = OneHotEncoder(categories='auto')
climate_OHE = label_encoder.fit_transform(climate).toarray()
region_OHE = label_encoder.fit_transform(region).toarray()

new_columns = climate_OHE.shape[1] + region_OHE.shape[1]
new_columns


# In[35]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return countries['Region'].nunique() + len(countries['Climate'].unique())
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[37]:


#Colunas numéricas
numeric_cols

#Pipeline para os dados numéricos
#preenchendos dados faltantes com a média
#Standardization

preprocess = Pipeline(steps=[
                        ('imput', SimpleImputer(missing_values=np.nan, strategy = 'median')),
                        ('standard', StandardScaler())
                        ])

preprocessing_country = preprocess.fit(countries[numeric_cols])


# In[38]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[39]:


#Criando um dataframe a partir de Countries, testando e reshaping
test_country = pd.DataFrame(test_country).T
test_country.columns = countries.columns
test_country


# In[40]:


def q4():
    # Retorne aqui o resultado da questão 4.
    test_processed = preprocessing_country.transform(test_country[numeric_cols])
    return float (round(test_processed[0,9],3))

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[41]:


country_q1, country_q3 = countries.quantile(q=0.25), countries.quantile(q=0.75)


# In[43]:


def q5():
    # Retorne aqui o resultado da questão 4.
    country_q1, country_q3 = countries.quantile(q=0.25), countries.quantile(q=0.75)
    iqr = country_q3 - country_q1
    lower, higher = country_q1 - 1.5*iqr, country_q3+1.5*iqr
    
    outlier_lower = countries[countries < lower]
    outlier_higher = countries[countries > higher]
    
    result = (
            int(outlier_lower['Net migration'].dropna().count()),
            int(outlier_higher['Net migration'].dropna().count()),
            False)
    
    return result
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[47]:


# dataset
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[48]:


def q6():
    # Retorne aqui o resultado da questão 4.
    #Word count do Sklearn
    vectorizer = CountVectorizer()
    newsgroup_vector = vectorizer.fit_transform(newsgroup['data'])
    newsgroup_matrix = newsgroup_vector.toarray()
    
    #Feature names
    words_list = vectorizer.get_feature_names()
    
    #Chegando se a matriz e a lista de palavras tem a mesma forma
    newsgroup_matrix.shape[1] == len(words_list)
    
    word_count = dict(zip(words_list, newsgroup_matrix.sum(axis=0)))
    return int(word_count['phone'])

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[51]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tf_vector = TfidfVectorizer(use_idf=True)
    newsgroup_tfvector = tf_vector.fit_transform(newsgroup['data'])
    
    tf_names = tf_vector.get_feature_names()
    tf_array = newsgroup_tfvector.toarray()
    
    tf_dict = dict(zip(tf_names, tf_array.sum(axis=0)))
    
    return float(round(tf_dict['phone'], 3))

q7()


# In[ ]:




