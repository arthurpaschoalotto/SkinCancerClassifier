import streamlit as st
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import plotly.express as px

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
        "nav-link-selected": {"background-color": "#50bb54"},
    }
    )
        
if choose == "Orientações":
    width = logo.width // 12
    height = logo.height // 12
    logo = logo.resize((width, height))
    st.header('Orientações de uso:')

    st.subheader('As imagens devem estar no formato:'+
            '\n- JPEG.\n - PNG.\n - JPG.')
    st.write('Apenas esses formatos são aceitos!')


    st.subheader('Resultados:')
    st.write('- Ele apresentará o tipo de lesão com maior probabilidade.'+
            '\n- Uma breve explicação sobre a lesão.'+
            '\n- Um gráfico de pizza com as porcentagens dos resultados.'+
            '\n- A imagem apresentada + Mapa de calor deixando em evidência o local da lesão.')

    st.subheader('Observações:')
    st.write('- O sistema só consgue diagnosticar lesão dentro das 7 classes: Doença de Bowen, Carcinoma basocelular, Ceratose benignas, Dermatofibroma, Melanoma, Nevos melanocíticos, Lesões vasculares'+
            '\n- Problemas de peles além das classes treinadas o sistema irá falhar e dignosticar para uma das classes cadastradas que mais se aproxima'+
            '\n- O Modelo teve 93% ACC no treinamento e 82% no teste, logo ele não é totalmente preciso, em uma análise real busque ajuda de um especialista.'+
            '\n- Esse sistema foi desenvolvido como projeto de trabalho final de uma especialização.')

    st.subheader('Duvidas ou problemas:')
    st.markdown("- **Arthur Paschoalotto:** "+" [![Foo](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arthur-paschoalotto-488839174/)"+
            "[![Foo](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arthurpaschoalotto)")

elif choose == "Sistema":
        # Defina o caminho para o arquivo .pth do modelo treinado
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features  # Extraindo features da camada fc
        model.fc = nn.Linear(num_features, 7)  # Número de classes

        model.load_state_dict(torch.load('model.pth'))
        model.to(device)
        model.eval()

        st.title("Análise de Imagens de Pele")
        st.write("Faça o upload de uma imagem e analisaremos os resultados para você!")

        # Faça o upload da imagem
        image_derm = st.file_uploader(label="Selecione a imagem", type=["jpg", "png", "jpeg"])

        if image_derm is not None:
                # Pré-processamento da imagem
                def preprocess(image_derm):
                        # Carregue a imagem e faça o pré-processamento
                        image = Image.open(image_derm)
                        # Aplicar transformações à imagem para prepará-la para o modelo
                        preprocess = transforms.Compose([
                        transforms.Resize(256),      # Redimensione para 256x256 pixels
                        transforms.CenterCrop(224),  # Realize um corte central de 224x224 pixels
                        transforms.ToTensor(),       # Converta a imagem em um tensor
                        transforms.Normalize(        # Normalize os valores dos canais RGB
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                        )
                        ])

                        # Aplicar as transformações à imagem
                        input_tensor = preprocess(image)
                        input_batch = input_tensor.unsqueeze(0)  # Adicione uma dimensão de lote (batch)

                        return input_batch, input_tensor

                # Analisar a imagem
                def analise(input_batch, model):
                        # Faça a previsão
                        with torch.no_grad():
                                input_batch = input_batch.to(device)  # Mova o tensor de entrada para o dispositivo (GPU)
                                output = model(input_batch)

                                # Aplicar a função softmax para obter probabilidades
                                probabilities = torch.softmax(output, dim=1)

                                # Nome das classes
                                class_names = ['Ceratose benignas', 'Carcinoma basocelular', 'Doença de Bowen', 'Dermatofibroma', 
                                               'Melanoma', 'Nevos melanocíticos', 'Lesões vasculares']

                                # Obtenha as probabilidades para cada classe
                                class_probabilities = {class_names[i]: prob.item() * 100 for i, prob in enumerate(probabilities[0])}

                                # Ordene as probabilidades em ordem decrescente
                                sorted_class_probabilities = dict(sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True))

                        return sorted_class_probabilities

                # Gerar mapa de calor
                def map_calor(input_tensor, input_batch, model):
                        # Calcule a ativação da camada convolucional desejada
                        # Neste exemplo, usaremos a camada 'layer4[1].conv1' como no seu código anterior
                        activation = model.layer4[1].conv1.weight
                        activation = activation[0].unsqueeze(0)  # Adicione uma dimensão de lote (batch)

                        # Redimensione a ativação para o tamanho da imagem original
                        activation = activation.detach().cpu().numpy()  # Converta para um array NumPy
                        activation -= activation.min()
                        activation /= activation.max()
                        activation *= 255
                        activation = np.uint8(activation)

                        # Redimensione a ativação para o tamanho da imagem original
                        activation = cv2.resize(activation[0], (input_tensor.shape[2], input_tensor.shape[1]))

                        # Aplique um mapa de cores para torná-lo visualmente mais informativo
                        heatmap = cv2.applyColorMap(activation, cv2.COLORMAP_JET)

                        # Misture a imagem original com o mapa de calor
                        input_image = input_batch.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                        input_image = np.uint8(input_image)
                        superimposed_img = cv2.addWeighted(input_image, 0.7, heatmap, 0.3, 0)

                        # Converta a imagem de volta para PIL e mostre-a
                        heatmap_image = Image.fromarray(superimposed_img)
                        st.image(heatmap_image, caption='Mapa de Calor', use_column_width=True)

##################
                def analise_e_retorna_resumo(input_batch, model):
                        # Faça a previsão
                        with torch.no_grad():
                                input_batch = input_batch.to(device)
                                output = model(input_batch)

                        # Aplicar a função softmax para obter probabilidades
                        probabilities = torch.softmax(output, dim=1)

                        # Nome das classes
                        class_names = [
                                'Doença de Bowen',
                                'Carcinoma basocelular',
                                'Ceratose benignas',
                                'Dermatofibroma',
                                'Melanoma',
                                'Nevos melanocíticos',
                                'Lesões vasculares'
                        ]

                        # Obtenha as probabilidades para cada classe
                        class_probabilities = {class_names[i]: prob.item() * 100 for i, prob in enumerate(probabilities[0])}

                        # Ordene as probabilidades em ordem decrescente
                        sorted_class_probabilities = dict(sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True))

                        # Mapeie as classes para descrições resumidas
                        class_descriptions = {
                                'Nevos melanocíticos': 'Lesões de pele benignas comuns que geralmente são pintas escuras.',
                                'Melanoma': 'Um tipo de câncer de pele maligno que requer atenção médica imediata.',
                                'Doença de Bowen': 'Uma condição pré-cancerígena que geralmente aparece como manchas vermelhas escamosas.',
                                'Dermatofibroma': 'Um nódulo benigno na pele que é muitas vezes marrom ou vermelho.',
                                'Lesões vasculares': 'Anormalidades nos vasos sanguíneos da pele, como hemangiomas.',
                                'Carcinoma basocelular': 'Um câncer de pele comum que geralmente parece uma mancha de pele elevada.',
                                'Ceratose benignas': 'Lesões ásperas e escamosas da pele geralmente causadas pelo sol.'
                        }

                        # Criar o texto resumido com base na classe mais provável
                        classe_mais_provavel = list(sorted_class_probabilities.keys())[0]
                        probabilidade = sorted_class_probabilities[classe_mais_provavel]
                        descricao = class_descriptions.get(classe_mais_provavel, 'Descrição não disponível.')

                        # Exibir os resultados formatados no Streamlit
                        st.header(f"**Diagnostico:** {classe_mais_provavel}")
                        st.write(f"**Descrição Resumida:** {descricao}")
                        st.write(f"**Probabilidade:** {probabilidade:.2f}%")

                        return sorted_class_probabilities
#################

                input_batch, input_tensor = preprocess(image_derm)

                # Chame a função e obtenha os resultados
                sorted_class_probabilities = analise_e_retorna_resumo(input_batch, model)

                # Analisar a imagem e exibir os resultados em um gráfico de pizza
                sorted_class_probabilities = analise(input_batch, model)

                # Extrair os nomes das classes e as probabilidades
                class_names = list(sorted_class_probabilities.keys())
                probabilities = list(sorted_class_probabilities.values())

                # Arredondar as probabilidades para duas casas decimais
                rounded_probabilities = [round(prob, 2) for prob in probabilities]

                # Criar um dataframe para usar com o Plotly
                data = {'Classe': [f"{class_name} ({prob} %)" for class_name, prob in zip(class_names, rounded_probabilities)],
                        'Probabilidade (%)': rounded_probabilities}
                df = pd.DataFrame(data)

                # Criar um gráfico de pizza
                fig = px.pie(df, values='Probabilidade (%)', names='Classe', title='Resultados da Análise')

                # Ajustar o tamanho da fonte das legendas e do texto no gráfico
                fig.update_layout(legend=dict(font=dict(size=10)), title=dict(font=dict(size=16)))

                # Exibir o gráfico de pizza
                st.plotly_chart(fig)

                col1, col2 = st.columns(2)
                with col1:
                        st.image(image_derm, caption='Imagem de Entrada', use_column_width=True)
                with col2:
                        # Gerar e exibir o mapa de calor
                        map_calor(input_tensor, input_batch, model)