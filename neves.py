import streamlit as st
import pandas as pd

def main():
    st.image('neves.jpeg')
    st.title('Repasses')
    file = st.file_uploader('Faça upload da planilha', type = 'csv')
    if file is not None:
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('Markdown')
        st.table(df.head(slider))
        st.write(df.columns)
        st.table(df.groupby('species')['petal_width'].mean())






if __name__ == '__main__':
    main()
