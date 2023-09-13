import streamlit as st
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

logo = Image.open('./logo.jpg')
width = logo.width
height = logo.height
logo = logo.resize((width, height))

st.sidebar.image(logo)

############################################### DADOS ############################################

### Upload Widget
with st.sidebar:
    choose = option_menu("Menu", ["Orientações", "Sistema"],
                         icons=['house', 'filter'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#4682B4"},
    }
    )
        
if choose == "Orientações":
    width = logo.width // 12
    height = logo.height // 12
    logo = logo.resize((width, height))
    st.header('Orientações de uso:')

    st.write('Os arquivos devem sempre estar nas extensões corretas para serem utilizados, sendo elas:'+
            '\n- wis.txt (arquivo texto).\n - volume.csv (arquivo separado por virgulas).')
    st.write('Depois de gerar os gráficos, as tabelas são salvas automaticamente na pasta "arquivos" do repositório, então ao finalizar o uso do sistema limpe a pasta para que não acumule arquivos.')


    st.subheader('Filtragens e Retornos:')
    st.write('- Período: atividades por matrícula 20 menores, atividades por matrícula 20 maiores e atividades total por período.'+
            '\n- Matrícula + Período: desempenho diário (ou hora) e atividades mais realizadas.'+
            '\n- Produto + Período: atividades realizadas e andamento de produção.'+
            '\n- Matrícula + Período + Flow: atividades realizadas, atividades mais realizadas e tempo médio de execução.'+
            '\n- Volume + Período: volume total por data, volume e atividades entregues por período e volume de peças fabricadas.')

    st.subheader('Padrão nome dos arquivos:')
    st.write('- **Período:** peri_dia inicial_dia final_abreviação do nome do gráfico.'+
            '\n- **Matrícula + Período:** n° da matricula_dia inicial_dia final_abreviação do nome do gráfico.'+
            '\n- **Produto + Período:** n° do produto_dia inicial_dia final_abreviação do nome do gráfico.'+
            '\n- **Matrícula + Período + Flow:** n° da matricula_dia inicial_dia final_flow_abreviação do nome do gráfico.'+
            '\n- **Volume + Período:** vol_dia inicial_dia final_abreviação do nome do gráfico.')

    st.subheader('Duvidas ou problemas:')
    st.markdown("- **Arthur Paschoalotto:** "+" [![Foo](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arthur-paschoalotto-488839174/)"+
            "[![Foo](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arthurpaschoalotto)")

elif choose == "Sistema":
    image_derm = st.file_uploader(label="Selecione a imagem",type=["jpg","png","jpeg"]) 