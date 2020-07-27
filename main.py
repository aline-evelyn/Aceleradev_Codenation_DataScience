#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[48]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA


# In[ ]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[2]:


fifa = pd.read_csv("fifa.csv")


# In[3]:


fifa.head()


# In[4]:


fifa.shape


# In[5]:


# Excluindo colunas
columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
fifa.head()


# In[7]:


fifa.shape


# In[9]:


fifa.info()


# In[10]:


fifa_nulos = fifa[fifa.ShotPower.isna()]
print(fifa_nulos.shape)
fifa_nulos.info()


# In[11]:


#criando uma cópia sem os dados nulos
dados_fifa = fifa.copy()


# In[12]:


dados_fifa = fifa[fifa.ShotPower.notna()]


# In[13]:


#confirmando a hipotese
dados_fifa.isna().sum().sum()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[14]:


#aplicando PCA usando a base de dados não nulos
pca = PCA()
pca.fit(dados_fifa)


# In[15]:


#retorna a porcentagem da variancia relativa ao primeiro PC
pca.explained_variance_ratio_[0]


# In[16]:


def q1():
    return round(float(pca.explained_variance_ratio_[0]), 3)


# In[17]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[19]:


#plotando um gráfico cumulativo para ver a quantidade de features devemos retornar com n_componentes
plt.figure(figsize=(12,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Gráfico cumulativo de componentes FIFA 2019')
plt.xlabel('Número de componentes')
plt.ylabel('Variancia explicada cumulativa');
#cria uma linha horizontal para marcar 0.95
plt.hlines(y=0.950, xmin = 0, xmax=37, linestyles='dashed', colors='red')
#Cria uma linha vertical no 15
plt.vlines(x = 15, ymin = 0.6, ymax=1, linestyles='dashed', colors='red')


# In[21]:


pca = PCA(0.95).fit(dados_fifa)
pca.n_components_


# In[24]:


def q2():
    return int(pca.n_components_)


# In[25]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[28]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[29]:


pca_fifa = PCA().fit(dados_fifa)


# In[30]:


#usando a lista para chamar a função dot do numpy e buscando as duas primeiras posições do PCA
pca_fifa.components_.dot(x)[0:2]


# In[31]:


#desempacotando a tupla e arredondando os dados
comp_1, comp_2 = pca_fifa.components_.dot(x)[0:2].round(3)
print(comp_1)
print(comp_2)


# In[32]:


def q3():
    return tuple((comp_1, comp_2))


# In[33]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[34]:


#importando do modelo de regressão linear
from sklearn.linear_model import LinearRegression


# In[35]:


#importando biblioteca e utilizando RFE
from sklearn.feature_selection import RFE


# In[36]:


#dividindo o dataset entre variáveis dependentes e variáveis explicativas
#dependente = y
y = dados_fifa['Overall']
y.head()


# In[37]:


#explicativas  = x
x = dados_fifa.drop(columns = "Overall")
x.head()


# In[38]:


#definindo variáveis
estimator = LinearRegression()
n_features_to_select = 5
step = 1


# In[39]:


#Criando o Objeto
rfe = RFE(estimator, n_features_to_select, step)


# In[40]:


rfe = rfe.fit(x,y)


# In[41]:


rfe.support_


# In[42]:


#criando um DF que junta o nome das colunas e a seleção dos features
colunas = pd.DataFrame({'coluna': x.columns,
                       'bool': rfe.support_})


# In[43]:


#fazendo query para visualização de colunas
colunas.query('bool == True')


# In[44]:


#selecionando os dados do problema
list(colunas.query('bool==True')['coluna'])


# In[45]:


def q4():
    return list(colunas.query('bool == True')['coluna'])


# In[46]:


q4()


# In[ ]:




