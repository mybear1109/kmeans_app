
from turtle import pd
import streamlit as st

def main():
    st.title('K-Means Clustering APP')

    # 1. csv file upload
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        # 2. 데이터 불러 오기 
        df = pd.read_csv(file)
        st.dataframe( df.head() )


if __name__ == '__main__':
    main()


