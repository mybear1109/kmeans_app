
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
from pandas.api.types import is_integer_dtype,is_float_dtype,is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt





import os
import matplotlib.font_manager as fm

@st.cache_data
def load_fonts():
    """Mac 시스템에서 사용 가능한 폰트 불러오기"""
    font_dirs = ["/System/Library/Fonts/Supplemental/", "/Library/Fonts/"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

def set_font():
    """Mac에서 사용할 기본 한글 폰트를 설정"""
    load_fonts()
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Mac 기본 한글 폰트 설정
    default_font = "AppleGothic" if "AppleGothic" in available_fonts else available_fonts[0]
    
    # 사용자 선택 폰트
    fontname = st.selectbox("사용할 한글 폰트를 선택하세요", np.unique(available_fonts), index=available_fonts.index(default_font))
    
    # Matplotlib에 적용
    plt.rcParams["font.family"] = fontname
    st.write(f"📌 현재 적용된 폰트: {fontname}")


def main():

      # 한글 폰트 설정
    set_font()
    
    st.title('K-Means Clustering APP')

    # 1. csv file upload
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        # 2. 데이터 불러 오기 
        df = pd.read_csv(file)
        st.dataframe( df.head() )

        st.info('NaN 이 있으면 해당 행을 삭제합니다.')
        st.dataframe( df.isna().sum() )
        df.dropna(inplace = True)
        df.reset_index(drop = True, inplace = True)


        # 3. 유저가 컬럼을 선택할 수 있게 한다.
        st.info('K-Means Clustering에 사용할 컬럼을 선택해주세요.')
        selected_columns = st.multiselect('컬럼 선택', df.columns)

        if len(selected_columns) == 0:
            st.error('컬럼을 선택해주세요.')
            return

        df_new = pd.DataFrame()
        # 4. 각 컬럼이 어떤 데이터 타입인지 보여준다.(문자열,숫자)
        for column in selected_columns:
            if is_integer_dtype(df[column]):
                df_new[column] = df[column]

            elif is_float_dtype(df[column]):
                df_new[column] = df[column]

            elif is_object_dtype(df[column]):
                if df[column].nunique() <= 2 : 
                    # 레이블 인코딩
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else : 
                    # 원핫 인코딩
                    ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
                    column_names = sorted( df[column].unique() )
                    df_new[column_names] = ct.fit_transform(df[column].to_frame() )    
            else:
                st.text(f'{column} 컬럼은 사용 불가로 제외되 었습니다.')

        st.info('K-Means 를 수행하기 위한 데이터 프레임입니다.')
        st.dataframe( df_new )


        st.subheader('최적의  k 값을 찾기 위해  WCSS 값을 계산합니다.')

        # 데이터의 갯수가 클러스터링의 갯수보다는 크거나 같아야 하므로
        # 해당 데이터의 갯수로 최대 k 값을 설정한다.
        st.text(f'데이터의 갯수 : {df_new.shape[0]} 개 입니다.')

        if df_new.shape[0] < 10 :
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value= 2, max_value= df_new.shape[0])
             
        else :
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value= 2, max_value= 10)
           
 
        
        wcss = []
        for k in range(1, max_k+1) :
            kmeans = KMeans(n_clusters= k, random_state= 4)
            kmeans.fit(df_new)
            wcss.append( kmeans.inertia_ )

        fig1 = plt.figure()
        plt.plot( range(1, max_k+1) ,  wcss )
        plt.title('The Elbow Method')
        plt.xlabel('클러스터 갯수')
        plt.ylabel('WCSS값')
        st.pyplot( fig1 )

        
    
        st.text('원하는 클러스터링(그룹) 갯수를 입력하세요')
        k = st.number_input('숫자 입력', min_value=2, max_value= max_k)

        kmeans = KMeans(n_clusters= k, random_state= 4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장 되었습니다.')
        st.dataframe( df )

if __name__ == '__main__':
    main()


