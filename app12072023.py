import streamlit as st
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
# import pickle
# import xgboost
# import joblib
import sys
import logging
from typing import Union
from google.cloud import storage
import altair as alt
import validar_preprocesar_predecir_organizarrtados
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors_plotly

import base64
# import os
import json
import pickle
import uuid
import re

# Para Botones sin recargar

# -----------------------------------------------------IAP GCP

app = Flask(__name__)

# -----------------------------------------------------

# Configure this environment variable via app.yaml
# CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']


@app.route('/')
def index() -> str:
    return """
<form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="datos" name="datos">
    <input type="submit">
</form>
"""


@app.route('/upload', methods=['POST'])
def upload(csvdata, bucketname, blobname):
    client = storage.Client()
    bucket = client.get_bucket(bucketname)
    blob = bucket.blob(blobname)
    blob.upload_from_string(csvdata)
    gcslocation = 'gs://{}/{}'.format(bucketname, blobname)
    logging.info('Uploaded {} ...'.format(gcslocation))
    return gcslocation


@app.errorhandler(500)
def server_error(e: Union[Exception, int]) -> str:
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

# -----------------------------------------------------


CERTS = None
AUDIENCE = None


def certs():
    """Returns a dictionary of current Google public key certificates for
    validating Google-signed JWTs. Since these change rarely, the result
    is cached on first request for faster subsequent responses.
    """
    import requests

    global CERTS
    if CERTS is None:
        response = requests.get(
            'https://www.gstatic.com/iap/verify/public_key'
        )
        CERTS = response.json()
    return CERTS


def get_metadata(item_name):
    """Returns a string with the project metadata value for the item_name.
    See https://cloud.google.com/compute/docs/storing-retrieving-metadata for
    possible item_name values.
    """
    import requests

    endpoint = 'http://metadata.google.internal'
    path = '/computeMetadata/v1/project/'
    path += item_name
    response = requests.get(
        '{}{}'.format(endpoint, path),
        headers={'Metadata-Flavor': 'Google'}
    )
    metadata = response.text
    return metadata


def audience():
    """Returns the audience value (the JWT 'aud' property) for the current
    running instance. Since this involves a metadata lookup, the result is
    cached when first requested for faster future responses.
    """
    global AUDIENCE
    if AUDIENCE is None:
        project_number = get_metadata('numeric-project-id')
        project_id = get_metadata('project-id')
        AUDIENCE = '/projects/{}/apps/{}'.format(
            project_number, project_id
        )
    return AUDIENCE


def validate_assertion(assertion):
    """Checks that the JWT assertion is valid (properly signed, for the
    correct audience) and if so, returns strings for the requesting user's
    email and a persistent user ID. If not valid, returns None for each field.
    """
    from jose import jwt
    try:
        info = jwt.decode(
            assertion,
            certs(),
            algorithms=['ES256'],
            audience=audience()
        )
        return info['email'], info['sub']
    except Exception as e:
        print('Failed to validate assertion: {}'.format(e), file=sys.stderr)
        return None, None


def download_excel(df_v, nombre='LogErrores', col=st):
    df_v.to_excel(nombre+'.xlsx', index=False)
    filename = nombre+'.xlsx'
    with open(nombre+'.xlsx', 'rb') as file:
        contents = file.read()
    if col == st:
        # st.download_button(label='Descargar '+nombre, data=contents, file_name=nombre+'.xlsx')
        download_button_str = download_button(contents, filename, nombre)
        st.markdown(download_button_str, unsafe_allow_html=True)

    else:
        # col.download_button(label='Descargar '+nombre, data=contents, file_name=nombre+'.xlsx')
        download_button_str = download_button(contents, filename, nombre)
        col.markdown(download_button_str, unsafe_allow_html=True)


def download_button(object_to_download, download_filename, button_text, pickle_it=False):

    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                -webkit-box-align: center;
                align-items: center;
                -webkit-box-pack: center;
                justify-content: center;
                font-weight: 400;
                padding: 0.25rem 0.75rem;
                border-radius: 50px;
                margin: 0px;
                line-height: 1.6;
                color: #461e7d;
                border: 2px solid #461e7d;
                font-weight: bold !important;
                width: auto;
                height: 35px;
                user-select: none;
                background-color: white;
                width: 200px !important;
                font-family: "Roboto", sans-serif;
                font-size: 13px;
                text-decoration: none;
            }}
            #{button_id}:active {{
                color: #461e7d;
                border-color: #461e7d;
                background-color: #ffffff;
                text-decoration: none;
            }}
            #{button_id}:focus:not(:active) {{
                color: #461e7d;
                border-color: #461e7d;
                background-color: #ffffff;
                text-decoration: none;
            }}
            #{button_id}:focus {{
                box-shadow: none;
                outline: none;
                text-decoration: none;
            }}
            #{button_id}:hover {{
                background-color: #461e7d;
                color: #ffff3c;
                border: 2px solid #461e7d;
                text-decoration: none;
            }}
        </style> """

    dl_link = custom_css + \
        f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def download_excel_torta(df_v, nombre='LogErrores', col=st):
    df_v.to_excel(nombre+'.xlsx', index=False)

    filename = nombre+'.xlsx'
    with open(filename, 'rb') as file:
        contents = file.read()

    if col == st:
        download_button_str = download_button(contents, filename, nombre)
        st.markdown(download_button_str, unsafe_allow_html=True)

    else:
        download_button_str = download_button(contents, filename, nombre)
        col.markdown(download_button_str, unsafe_allow_html=True)


# ,cols = [col11, col12, col13, col14,col15]}
def botones_descarga(Xf=None, variable='RangoConsumo', col=None):

    for categoria in Xf[variable].unique():

        download_excel_torta(
            Xf[Xf[variable] == categoria], nombre=categoria, col=col)


def download_txt(nombre, logs):

    # Especifica la ruta y el nombre de archivo del archivo de texto
    archivo_txt = "archivo.txt"

    # Abre el archivo en modo escritura
    with open(archivo_txt, "w") as archivo:
        # Escribe cada valor de la lista en una línea separada
        for valor in logs:
            archivo.write(valor + "\n \n")
    # Leer el contenido del archivo
    with open(archivo_txt, 'rb') as file:
        contents = file.read()

    # Descargar el archivo

    st.download_button(nombre + '.txt', data=contents, file_name="archivo.txt")


@app.route('/', methods=['GET'])
def say_hello():
    from flask import request
    assertion = request.headers.get('X-Goog-IAP-JWT-Assertion')
    email, id = validate_assertion(assertion)
    page = "<h1>Hello {}</h1>".format(email)
    return page

# ------------------------------------------------------

def agregar_k(valor):
    return str(valor) + 'K'

def generar_graficos(df_t, configuraciones, mayus=True, color=1, auto_orden=False, total=False):
    for config in configuraciones:
        df_group = df_t.groupby(by=config['groupby'], as_index=True)[
            'NIT9'].count()
        df_group = pd.DataFrame(df_group)
        # st.write(df_group)

        if mayus == True:
            if not auto_orden:
                # Reordenar el DataFrame según el orden deseado
                df_group = df_group.reindex(config['order'])
            else:
                df_group.sort_values(by='NIT9', ascending=False, inplace=True)
        else:
            # Ordenar de mayor a menor
            df_group.sort_values(by='NIT9', ascending=False, inplace=True)

        # Extrae indice a columna
        df_group.reset_index(inplace=True, drop=False)
        df_group.dropna(inplace=True)
        df_group.reset_index(inplace=True, drop=True)
        # st.write(df_group)

        df_group.rename(
            {'NIT9': 'Cantidad_n', config['groupby']: config['y_axis']}, axis=1, inplace=True)
        df_group['Cantidad_n'] = pd.to_numeric(df_group['Cantidad_n'])
        df_group['Cantidad_n'] = df_group['Cantidad_n']*100
        
        if mayus == True:

            keys = config['order']
            values = config['order_f']

            diccionario = dict(zip(keys, values))

            df_group[config['y_axis']] = df_group[config['y_axis']
                                                  ].replace(diccionario)

        df_group[config['y_axis']] = pd.Categorical(
            df_group[config['y_axis']], ordered=True)

        df_group['Porcentaje'] = df_group['Cantidad_n'] / \
            df_group['Cantidad_n'].sum() * 100
        df_group['Porcentaje'] = df_group['Porcentaje'].round(2)
        df_group['Porcentaje'] = df_group['Porcentaje'].apply(
            lambda x: ' {:.2f}%'.format(x))
        
        # df_group['Cantidad'] = df_group['Cantidad_n'].apply(agregar_k)  
        
        # st.write(df_group)
        if color == 0:
            color_b = "#52a5d9"  # 00d26a"# "#2afd95"#'#717171' #'#023059'
        if color == 1:       # Morado oscuro
            color_b = '#311557'
        elif color == 2:      # Naranja
            color_b = '#fd6600'
        elif color == 3:     # Azul cielo
            color_b = '#2cfef9'
        elif color == 4:    # Verde Claro
            color_b = '#2afd95'
        elif color == 5:  # Amarillo Claro
            color_b = '#fef367'
        elif color == 6:     # Fuccia
            color_b = '#ff006e'
        elif color == 7:     # Morado Claro
            color_b = '#5738ff'

        if mayus == True:
            if total == False:

                #nuevos_valores_xticks = [5,10,15]#'5 K', '10 K', '15 K', '20 K'
                bar = alt.Chart(df_group).mark_bar().encode(
                    x=alt.X('Cantidad_n', axis=alt.Axis(
                        ticks=True, title=config['x_axis_title'],
                        #values=nuevos_valores_xticks
                        ),
                    #scale=alt.Scale(type='ordinal')
                    ),
                    y=alt.Y(config['y_axis'] + ":N", sort=list(
                        df_group[config['y_axis']]), axis=alt.Axis(ticks=False, title='')),
                    tooltip=[config['y_axis']+":N",
                    'Cantidad_n:Q', 'Porcentaje:O'
                        # alt.Tooltip(config['y_axis']+":N"),
                        # alt.Tooltip('Cantidad:Q', format=''),  # Add k letter before number
                        # alt.Tooltip('Porcentaje:O', format='%')
                             ],
                    text=alt.Text('Porcentaje:N')
                ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False)  # .configure_axisY(
               
               
               
               
                # labelFont='Roboto',  # Cambia la fuente de las etiquetas
                # labelFontSize=1,   # Cambia el tamaño de las etiquetas
                # labelColor='#8568a8'    # Cambia el color de las etiquetas
                # )#.properties(width=100,  # Especifica el ancho del gráfico en píxeles
                #           height=200)  # Especifica la altura del gráfico en píxeles
                # .configure_axisX(ticks=True, labels=True)

                config['col'].altair_chart(
                    bar, use_container_width=True, theme="streamlit")
                st.write("")
            else:

                bar = alt.Chart(df_group).mark_bar().encode(
                    x=alt.X('Cantidad', axis=alt.Axis(
                        ticks=False, title=config['x_axis_title'])),
                    y=alt.Y(config['y_axis'] + ":N", sort=list(
                        df_group[config['y_axis']]), axis=alt.Axis(ticks=False, title='')),
                    tooltip=[config['y_axis']+":N",
                             'Cantidad:Q', 'Porcentaje:O'],
                    text=alt.Text('Porcentaje:N')
                ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False).configure_axisX(ticks=False, labels=False)
                # .configure_axisY(
                #     labelFont='Roboto',  # Cambia la fuente de las etiquetas
                #     labelFontSize=22,   # Cambia el tamaño de las etiquetas
                #     labelColor='#8568a8'    # Cambia el color de las etiquetas
                # )
                config['col'].altair_chart(
                    bar, use_container_width=True, theme="streamlit")
                st.write("")

        else:
            bar = alt.Chart(df_group).mark_bar().encode(
                x=alt.X('Cantidad', axis=alt.Axis(
                    ticks=False, title=config['x_axis_title'])),
                # y=config['y_axis'] + ":N",
                y=alt.Y(config['y_axis'] + ":N", sort=list(df_group[config['y_axis']]),
                        axis=alt.Axis(ticks=False, title=None)),  # , title=None
                tooltip=[config['y_axis']+":N", 'Cantidad:Q', 'Porcentaje:O'],
                text=alt.Text('Porcentaje:N')
            ).configure_mark(color=color_b).configure_view(fill="none").configure_axis(grid=False)
            # .configure_axisY(
            # labelFont='Roboto',  # Cambia la fuente de las etiquetas
            # labelFontSize=16,   # Cambia el tamaño de las etiquetas
            # labelColor='#8568a8'    # Cambia el color de las etiquetas
            # )

            config['col'].altair_chart(
                bar, use_container_width=True, theme="streamlit")
            st.write("")

# ------------------------------------------------------


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_excel()  # .encode('utf-8')


def dona_plotly(df_prob_prod, producto='INSTALACIONES', col=None, titulo=None, tamano_pantalla=(400, 400)):

    valores = df_prob_prod.loc[:, producto].astype(int)*100
    etiquetas = ['Alta', 'Media', 'Baja']

    # Colores personalizados
    colores = ['#3257A3', '#52A5D9', '#ACD7F2']
    # colores = ['#3F6CA6','#2C4A73','#A0C4F2']
    # colores = ['#9AB5D9','#CEDEF2','#035AA6']

    total = sum(valores)
    conteos = [str(valor) for valor in valores]
    porcentajes = [f'{(valor/total)*100:.1f}%' for valor in valores]

    fig = go.Figure(data=[
        go.Pie(
            labels=etiquetas,
            values=valores,
            hole=0.55,
            textinfo='none',  # 'label+text+percent',
            # text=conteos,
            hovertemplate='%{label}',  # '%{label}<br>%{text} (%{percent})',
            marker=dict(colors=colores)
        )
    ])
    if titulo:
        fig.update_layout(title={
            'text': titulo,
            'y': 0.95,
            'yanchor': 'top',
            'font': {'size': 24}
        }
        )

    fig.update_layout(width=tamano_pantalla[0], height=tamano_pantalla[1])
    # fig.update_layout(  legend=dict(
    #                     orientation='v',  # Orientación horizontal
    #                     yanchor='bottom', # Ancla en la parte inferior
    #                     y=1.3,  # Desplazamiento vertical
    #                     xanchor='right',  # Ancla en el extremo derecho
    #                     x=0.9 # Desplazamiento horizontal
    #                     ,font=dict(size=12)  # Tamaño de la fuente
    #                     ))

    # Ocultar el legend
    fig.update_layout(
        showlegend=False
    )
    fig.update_traces(
        text=conteos,
        textinfo='label+text+percent',  # Activa el texto personalizado
        textposition='outside'  # Mueve el texto fuera de la dona
    )

# -----------------------------------------------------

    # Reducir el tamaño de las etiquetas
    fig.update_traces(
        textfont=dict(
            size=13  # Tamaño de la fuente de las etiquetas
        )
    )

# -----------------------------------------------------

    col.plotly_chart(fig, use_container_width=True)
    # Encabezado inicial
    # header = st.empty()


def espacio(col, n):
    if n > 0:
        for i in range(n):
            col.write('')


def scatter_plot(df, col=None):
    # Definir los colores base
    color_azul = 'rgb(70, 30, 125)'
    color_amarillo = 'rgb(254, 243, 103)'

    # Crear la paleta de color
    colores = [color_azul, color_amarillo]

    # Crear la escala de color continua
    colorscale = colors_plotly.make_colorscale(colores)
    # Crear el gráfico scatter utilizando plotly express
    fig = px.scatter(df, x='DEPARTAMENTO', y='ACTIVIDADES',
                     # 'OPORTUNIDADESCOTIZADAS(#)',
                     color='OPORTUNIDADESVENDIDAS', size='OPORTUNIDADESCOTIZADAS($)',
                     # 'Plasma'#px.colors.sequential.Cividis#'Plotly3'#'matter_r'#'purples_r'
                     color_continuous_scale=colorscale
                     )

    # Personalizar el diseño del gráfico

    fig.update_layout(coloraxis_colorbar=dict(len=1, ypad=0))

    fig.update_layout(xaxis_title='Departamento', yaxis_title='Actividad económica',
                      coloraxis_colorbar=dict(title='Ventas'), width=875, height=500)

    fig.update_layout(coloraxis_colorbar=dict(
        tickmode='array',  # Usar modo de ticks de arreglo
        tickvals=list(range(0, 27, 2)),  # Valores de los ticks personalizados
        ticktext=list(range(0, 27, 2))  # Etiquetas de los ticks personalizados
    ))

    fig.update_traces(
        hovertemplate='<b>Departamento</b>: %{x}<br>'
        '<b>Actividad económica</b>: %{y}<br>'
        '<b>Oportunidades vendidas</b>: %{marker.color}<br>'
        '<b>Oportunidades cotizadas</b>: %{marker.size:,}<extra></extra>'
    )

    col.plotly_chart(fig, use_container_width=True,)

    col.write('')


def main():

    # Configura titulo e icon de pagina
    st.set_page_config(page_title="CHUBB",
                       page_icon="img/Icono.png", layout="wide")

    # Leer el contenido del archivo CSS
    css = open('styles.css', 'r').read()

    # Agregar estilo personalizado
    st.markdown(
        f'<style>{css}</style>',
        unsafe_allow_html=True)

    # Variable que controla la visibilidad de la imagen
    b = False
    vista2,vista1  = st.tabs(
        [ "Reporte descriptivo","Resultado modelo"])  # 'Inicio', vista0,

    # Menú y logo
    # st.sidebar.image("img/enelX_logo_negative-resize.png", width=200)
    # st.sidebar.write("")
    container01 = st.sidebar.container()
    container01.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');
        .custom-container {
            background-color: #9e9ac8;
            padding: 2.5px;
            font-family: 'Orbitron', sans-serif;
        }
        h6 {
            font-family: 'Orbitron', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    container01.markdown(f'<div style="width: 300px;height: 60px;aspect-ratio: 16 / 8.1;border-radius: 12px;background-color: #ffffff;background-repeat: no-repeat;background-size: cover;background-position: center center;box-shadow: rgba(86, 90, 97, 0.12) 0px 4px 12px;margin-bottom: 21px; text-align: center;display: flex;justify-content: center;"><h6>CHUBB</h6></div>', unsafe_allow_html=True)

    # Estilo botón
    st.markdown("""
            <style>
            div.stButton > button:hover {
                background-color:#f0f2f6;
                color:#461e7d
            }
            </style>""", unsafe_allow_html=True)

    # a, b, c = False, False, False

    st.markdown(
        """
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    </head>

    <div style="display: flex; justify-content: center;">
        <a href="https://www.facebook.com/addicol" style="color: #1877F2; margin: 0 15px;">
            <i class="fab fa-facebook" style="font-size: 30px;"></i>
        </a>
        <a href="https://www.instagram.com/addi/" style="color: #E4405F; margin: 0 15px;">
            <i class="fab fa-instagram" style="font-size: 30px;"></i>
        </a>
        <a href="https://www.tiktok.com/@addicolombia" style="color: #000000; margin: 0 15px;">
            <i class="fab fa-tiktok" style="font-size: 30px;"></i>
        </a>
    </div>
    """,
        unsafe_allow_html=True
    )

    with st.sidebar.expander("MODELO MÚLTIPLES CLIENTES", expanded=False):

        try:
            datos = st.file_uploader("Subir archivos: ", type=["xlsx"])
            # b=False
            if datos is not None:

                dataframe = pd.read_excel(datos)
                dataframe.index = range(1, len(dataframe)+1)

                try:
                    dataframe['FECHACONSTITUCION'] = dataframe['FECHACONSTITUCION'].astype(
                        'datetime64[ns]')
                except:
                    pass

                # try:

                # Validación archivo
                original_len = len(dataframe.copy())
                ob = validar_preprocesar_predecir_organizarrtados.Modelos_2(
                    dataframe)
                df_v, text, final_flag = ob.Validar_todo()


# -----------------------------------------------------

# -----------------------------------------------------

# -----------------------------------------------------

                if final_flag == False:

                    logs, logs_riesgo, indices_posibles = ob.Logs()

                    # st.write(len(logs_riesgo))
                    # st.write(logs_riesgo)

                    if '1' not in logs_riesgo:

                        tx_registros_aptos = str('Registros aptos para recomendar: ') + str(len(
                            indices_posibles)/10)+'K ('+str(round(100*(len(indices_posibles))/original_len, 2))+'%)'
                        st.success(tx_registros_aptos, icon="✅")
                        b = st.button("Ejecutar Modelo", type="primary")

                    download_txt(logs=logs, nombre='Log_errores')

                    for i, j in zip(range(len(logs)), logs_riesgo):

                        if i == 0:       # Si es el primer log agrega '¡Ups! Parece que hay un problema.'

                            if j == '1':
                                st.write(
                                    '<div align="center"><h2>¡Ups! Parece que hay un problema.</h2></div>', unsafe_allow_html=True)

                            if (len(logs[i]) > 150) & (j == '1'):
                                st.warning(logs[i][:172]+'...', icon="⚠️")

                            elif (len(logs[i]) > 150) & (j == 0):
                                st.info(logs[i][:172]+'...', icon="ℹ️")

                            elif (len(logs[i]) <= 150) & (j == '1'):
                                st.warning(logs[i][:], icon="⚠️")

                            elif (len(logs[i]) <= 150) & (j == 0):
                                st.info(logs[i][:], icon="ℹ️")
                        else:
                            if (len(logs[i]) > 150) & (j == '1'):
                                st.warning(logs[i][:172]+'...', icon="⚠️")

                            elif ((len(logs[i]) > 150) & (j == 0)):
                                st.info(logs[i][:172]+'...', icon="ℹ️")

                            elif ((len(logs[i]) <= 150) & (j == '1')):
                                st.warning(logs[i], icon="⚠️")

                            elif ((len(logs[i]) <= 150) & (j == 0)):
                                st.info(logs[i], icon="ℹ️")

                    st.write('')

                else:
                    st.success(text+' (100%)', icon="✅")
                    b = st.button("Ejecutar Modelo", type="primary")

        except UnboundLocalError:
            st.warning('Error. Problemas con caracteristicas del archivo.')

    # with st.sidebar.expander("MODELO UNITARIO ", expanded=False):
    #     try:
    #         # Lectura de datos
    #         nit = st.number_input("Digite el número del Nit",
    #                               min_value=1000000, max_value=99999999999)
    #         actEcon = st.text_input(
    #             "Actividad económica", value='Administración Empresarial')
    #         tamEmp = st.selectbox("Tamaño de la empresa", [
    #                               'Gran Empresa', 'Mediana Empresa', 'Pequeña Empresa'])
    #         flegal = st.selectbox("Forma Legal", [
    #                               'SAS', 'LTDA', 'SA', 'ESAL', 'SUCURSALEXTRANJERA', 'SCA', 'UNDEFINED', 'SCS', 'PERSONANATURAL'])
    #         numEmpl = st.number_input(
    #             "Número de empleados", min_value=1, step=1)
    #         activos = st.number_input("Activos Totales")
    #         ingresosOp = st.number_input("Total Ingresos Operativos")
    #         TotPatr = st.number_input("Total Patrimonio")
    #         ganDespImpto = st.number_input("Ganancias después de Impuestos")
    #         fecha_constitucion = st.date_input(
    #             "Fecha de constitución", min_value=date(1000, 1, 1), max_value=date.today())
    #         consprom = st.number_input("Consumo promedio kWh", min_value=0)

    #         # button
    #         boton_c = st.button("Ejecutar Modelo",
    #                             key="boton_ejecutar", type="primary")
    #         dataframe_u = pd.DataFrame({'NIT9': nit,
    #                                     'ACTIVIDADPRINCIPAL(EMIS)': actEcon,
    #                                     'TAMANOEMPRESA': tamEmp,
    #                                     'FORMALEGAL': flegal,
    #                                     'NUMERODEEMPLEADOS': numEmpl,
    #                                     'ACTIVOSTOTALES': activos,
    #                                     'TOTALINGRESOOPERATIVO': ingresosOp,
    #                                     'TOTALDEPATRIMONIO': TotPatr,
    #                                     'GANANCIASDESPUESDEIMPUESTOS': ganDespImpto,
    #                                     'FECHACONSTITUCION': fecha_constitucion,
    #                                     'CONSPROM': consprom}, index=[1])

    #         try:
    #             dataframe_u['FECHACONSTITUCION'] = dataframe_u['FECHACONSTITUCION'].astype(
    #                 'datetime64[ns]')
    #             # dataframe_u['NUMERODEEMPLEADOS']=dataframe_u['NUMERODEEMPLEADOS'].astype('int64')
    #         except:
    #             pass
    #         if boton_c == True:
    #             ob_u = validar_preprocesar_predecir_organizarrtados.Modelos_2(
    #                 dataframe_u)

    #         else:
    #             pass

    #     except UnboundLocalError:
    #         st.warning('Error. Problemas con los datos ingresados.')

    if b == True:
        # -----------------------------------------------------13/06/23

        with vista1:    # Modelo Multiples Clientes
            try:
                Xi, Xf = ob.predict_proba()

                # Modifico nombres de categorias
                keys = ['SINCATALOGAR', 'MENORA5000',
                        'ENTRE5000Y10000', 'ENTRE10000Y55000',  'MAYORA55000']
                values = ['Sin catalogar', 'Menor a 5000 kW⋅h',
                          'Entre 5000 y 10000 kW⋅h', 'Entre 10000 y 55000 kW⋅h']

                dic_rango_consumo = dict(zip(keys, values))

                Xf['RANGOCONSUMO'] = Xf['RANGOCONSUMO'].replace(
                    dic_rango_consumo)
                # st.write(Xi)

                hm_df = pd.DataFrame({'index': ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA',
                                                'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']})

                productos = ['Producto_1', 'Producto_2',
                             'Producto_3']  # Solo 3 primeras
                # print(hm_df,Xf)
                for i in productos:
                    hm_df = pd.merge(hm_df, pd.DataFrame(Xf[i].value_counts(
                        dropna=False)).reset_index(drop=False), how='outer', on='index')

                # Suma # primeras predicciones
                df_tmp = pd.DataFrame(hm_df['index'].copy())
                df_tmp.rename({'index': 'Productos'}, axis=1, inplace=True)

                # print(hm_df.columns)                                                                        ##------------

                df_tmp['Top 3'] = hm_df[['Producto_1',
                                         'Producto_2', 'Producto_3']].sum(axis=1)

                df_tmp['Porcentaje'] = df_tmp['Top 3'] / \
                    df_tmp['Top 3'].sum() * 100
                df_tmp['Porcentaje'] = df_tmp['Porcentaje'].round(2)
                # df_tmp['Porcentaje'] = df_tmp['Porcentaje'].apply(lambda x: ' {:.2f}%'.format(x))

                #
                keys = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA',
                        'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']
                values = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS DE CARGA',
                          'FIBRA OPTICA', 'REDES ELECTRICAS', 'ILUMINACION', 'CUENTAS NUEVAS']
                diccionario = dict(zip(keys, values))

                df_tmp['Productos'] = df_tmp['Productos'].replace(
                    diccionario)  # Corrijo nombre de los productos

                # Obtener la paleta de colores 'Purples'
                colors = plt.cm.Purples(range(256))
                # Seleccionar los tres tonos deseados
                C = [colors[80], colors[170], colors[255]]

                merged_df = pd.DataFrame(index=['Alta', 'Media', 'Baja'])
                df_prob_prod = pd.DataFrame()

                productos = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA',
                             'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']

                for prod in productos:

                    df_tmp1 = pd.DataFrame(
                        Xf[Xf['Producto_1'] == prod]['Probabilidad_1'].value_counts())
                    df_tmp2 = pd.DataFrame(
                        Xf[Xf['Producto_2'] == prod]['Probabilidad_2'].value_counts())
                    df_tmp3 = pd.DataFrame(
                        Xf[Xf['Producto_3'] == prod]['Probabilidad_3'].value_counts())

                    merged_df = pd.DataFrame(index=['Alta', 'Media', 'Baja'])

                    merged_df = merged_df.merge(
                        df_tmp1, left_index=True, right_index=True, how='outer')
                    merged_df = merged_df.merge(
                        df_tmp2, left_index=True, right_index=True, how='outer')
                    merged_df = merged_df.merge(
                        df_tmp3, left_index=True, right_index=True, how='outer')

                    merged_df = merged_df.fillna(0)
                    merged_df['Total'] = merged_df.sum(axis=1)
                    df_prob_prod[prod] = merged_df['Total']

                df_prob_prod = df_prob_prod.reindex(['Alta', 'Media', 'Baja'])

                for prod in productos:
                    df_prob_prod['P_'+prod] = np.round(
                        df_prob_prod[prod]/df_prob_prod[prod].sum() * 100, 2)

                # st.write(df_prob_prod)

                container0 = st.container()
                container0.markdown(
                    """
                    <style>
                    .custom-container {
                        background-color: #9e9ac8;
                        padding: 2.5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                container0.markdown(
                    f'<div class="custom-container"></div>', unsafe_allow_html=True)

                # Crear el primer contenedor
                container1 = st.container()
                # Aplicar CSS personalizado al contenedor
                container1.markdown(
                    """
                    <style>
                    .custom-container {
                        background-color: #f2f0f7;
                        padding: 2.5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Crear el segundo contenedor
                container2 = st.container()
                # Dividir el primer contenedor en dos columnas
                col1_container1, col2_container1, col3_container1 = container1.columns(spec=[
                                                                                       2.5, 2.3, 1])
                # Dividir el segundo contenedor en tres columnas
                col1_container2, col2_container2 = container2.columns(
                    2)  # , col3_container2, col4_container2


# -----------------------------------------------------

                # # Título con tamaño y color configurables
                tamaño1 = 30  # Tamaño1 del título
                tamaño2 = 60  # Tamaño2 del título
                color1 = '#54278f'  # '#54278f#'#461e7d'  # Color del título en formato hexadecimal
                color2 = '#9e9ac8'

                texto1 = 'Total clientes analizados'  # +'\n'+str(len(Xf))
                texto2 = str('  '+str(len(Xf)/10)+' K')  # +' Clientes')

                # col1_container1.text_align("center")
                container1.markdown(
                    f'<div class="custom-container"></div>', unsafe_allow_html=True)

                col1_container1.markdown(
                    f'<h1 style="text-align: center; font-size: {tamaño1}px; color: {color1};">{texto1}</h1>', unsafe_allow_html=True)

                col1_container1.markdown(
                    f'<h1 style="text-align: center; font-size: {tamaño2}px; color: {color2}">{texto2}</h1>', unsafe_allow_html=True)

                configuraciones = [
                    {
                        'groupby': 'Producto_1',
                        'count_col': 'NIT9',
                        'x_axis_title': None,
                        'y_axis': 'Producto 1',
                        # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                        'col': col2_container1,
                        # Orden deseado de las categorías
                        'order': ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS'],
                        # Orden deseado de las categorías']ACIONES','MANTENIMIENTO','ESTUDIOS','AUMENTOS_CARGA','FIBRA_OPTICA','REDESELECTRICAS','ILUMINACION','CUENTASNUEVAS0
                        'order_f':['Compra protegida', 'AP', 'Equipos electrónicos móviles', 'Autocontenidos', 'Renta hogar', 'Exequias', 'Bolso protegido', 'AP Travel']
                    }]

                # espacio(0,0,1,9)
                espacio(col2_container1, 1)
                generar_graficos(
                    Xf, auto_orden=True, configuraciones=configuraciones, color=0, total=False)
                espacio(col3_container1, 9)
                # col3_container1)
                
                Xf = Xf.loc[:,['NIT9','Producto_1',	'Probabilidad_1',	'Valor_probabilidad1',	'Producto_2',
                               'Probabilidad_2',	'Valor_probabilidad2',	'Producto_3',	'Probabilidad_3','Valor_probabilidad3',
                               'Producto_4',	'Probabilidad_4',	'Valor_probabilidad4',	'Producto_5',	'Probabilidad_5',
                               'Valor_probabilidad5',	'Producto_6',	'Probabilidad_6',	'Valor_probabilidad6',	'Producto_7',
                               'Probabilidad_7',	'Valor_probabilidad7',	'Producto_8',	'Probabilidad_8',	'Valor_probabilidad8']]
                
                dic1 = ['INSTALACIONES', 'MANTENIMIENTO', 'ESTUDIOS', 'AUMENTOS_CARGA', 'FIBRA_OPTICA', 'REDESELECTRICAS', 'ILUMINACION', 'CUENTASNUEVAS']
                dic2 = ['Compra protegida', 'AP', 'Equipos electrónicos móviles', 'Autocontenidos', 'Renta hogar', 'Exequias', 'Bolso protegido', 'AP Travel']
                Xf = Xf.replace(dict(zip(dic1, dic2)))
                download_excel(Xf, 'Resultado', col=col2_container1)

                # INSTALACIONES
                dona_plotly(df_prob_prod=df_prob_prod, producto='INSTALACIONES',
                            titulo='Compra Protegida', col=col1_container2)

                # ######## Mantenimiento
                # dona('MANTENIMIENTO',0 , 1, 'Mantenimiento')
                dona_plotly(df_prob_prod=df_prob_prod, producto='MANTENIMIENTO',
                            titulo='Equipos Electrónicos Móviles', col=col2_container2)

                # ######## Estudios
                # dona('ESTUDIOS',0 , 2, 'c')
                dona_plotly(df_prob_prod=df_prob_prod, producto='ESTUDIOS',
                            titulo='AP', col=col1_container2)

                # #AUMENTOS_CARGA
                # dona('AUMENTOS_CARGA',0 , 3, 'Aumentos de carga')
                dona_plotly(df_prob_prod=df_prob_prod, producto='AUMENTOS_CARGA',
                            titulo='Renta Hogar', col=col2_container2)

                # #FIBRA OPTICA
                # dona('FIBRA_OPTICA',1 , 0, 'Fibras ópticas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='FIBRA_OPTICA',
                            titulo='Autocontenidos', col=col1_container2)

                # #REDESELECTRICAS
                # dona('REDESELECTRICAS',1 , 1, 'Redes eléctricas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='REDESELECTRICAS',
                            titulo='AP Travel', col=col2_container2)

                # #ILUMINACION
                # dona('ILUMINACION',1 , 2, 'Iluminación')
                dona_plotly(df_prob_prod=df_prob_prod, producto='ILUMINACION',
                            titulo='Bolso Protegido', col=col1_container2)

                # #CUENTASNUEVAS
                # dona('CUENTASNUEVAS',1 , 3, 'Cuentas nuevas')
                dona_plotly(df_prob_prod=df_prob_prod, producto='CUENTASNUEVAS',
                            titulo='Exequias', col=col2_container2)

            except UnboundLocalError:
                st.warning(
                    'Error. En el menú de la izquierda cargar archivo en la sección Modelo múltiples clientes')

        with vista2:    # Descriptiva
            try:
                tab4, tab2 = st.tabs(
                    [ "Demográfico","Ventas" ])

                df_t, _ = ob.transform_load()  # _graf
                df_t = df_t.copy()
                # with tab1:  # RANGOCONSUMO piee
                #     st.write("")
                #     st.write("")

                #     col10, col11, col12, col13 = st.columns(
                #         spec=[0.35, 5, 1, 0.25])

                #     configuraciones = [
                #         {
                #             'groupby': 'RANGOCONSUMO',
                #             'count_col': 'NIT9',
                #             'x_axis_title': 'Cantidad de clientes',
                #             'y_axis': 'Rango de Consumo',
                #             'col': col11,
                #             'order': ['SINCATALOGAR', 'MENORA5000', 'ENTRE5000Y10000', 'ENTRE10000Y55000',  'MAYORA55000'],
                #             'order_f':['Sin catalogar', 'Menor a 5000 kW⋅h',  'Entre 5000 y 10000 kW⋅h', 'Entre 10000 y 55000 kW⋅h',   'Mayor a 55000 kW⋅h']
                #         }
                #     ]

                #     col11.subheader("Rango de consumo")
                #     # ob.generar_graficos_pie(configuraciones)
                #     col11.plotly_chart(ob.generar_graficos_pie(
                #         configuraciones, paleta=1), use_container_width=True)

                #     n = 0

                #     col12.write("")

                #     col12.write("Descargar:")
                #     col12.write("")

                #     keys = ['Sin Catalogar', 'Menor a 5000', 'Entre 5000 y 10000',
                #             'Entre 10000 y 55000', 'Mayor a 55000']
                #     values = ['Sin catalogar', 'Menor a 5000 kW⋅h',
                #               '5000 a 10000 kW⋅h', '10000 a 55000 kW⋅h', 'Mayor a 55000 kW⋅h']

                #     dic_rango_consumo = dict(zip(keys, values))

                #     Xf['RANGOCONSUMO'] = Xf['RANGOCONSUMO'].replace(
                #         dic_rango_consumo)

                #     print(Xf['RANGOCONSUMO'].unique())
                #     botones_descarga(Xf=Xf, variable='RANGOCONSUMO', col=col12)

                #     # st.write(Xf)
                #     # boton_descarga(label ='Sin Catalogar',data=Xf[Xf['RANGOCONSUMO']=='SINCATALOGAR']  )
                #     # download_excel(Xf[Xf['RANGOCONSUMO']=='SINCATALOGAR'], 'Sin Catalogar')
                #     # download_excel(Xf[Xf['RANGOCONSUMO']=='MENORA5000'], 'Menor a  5000')


# -----------------------------------------------------

                with tab2:  
                    st.write("")
                    st.write("")
                    # st.subheader("VENTAS")
                    # st.write("")
                    col1,col2,col3 = st.columns(spec=[1,5,1]) #

                    col2.subheader("Clientes con producto")
 
                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'RANGODECOMPRA($)',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Rango de compra',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col2,
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'PEQUENOCOMPRADOR', 'MEDIANOCOMPRADOR', 'GRANCOMPRADOR', 'COMPRADORMEGAPROYECTOS'],  # Orden deseado de las categorías
                            'order_f':['AP','AP','Compra protegida','Equipos móviles','Exequias', 'Bolso protegido','No comprador']
                        }]
                    generar_graficos(df_t, configuraciones,color =1)   

                    col2.subheader("Oferta último semestre")    
                    configuraciones = [
                        {
                            'groupby': 'RANGORECURRENCIACOMPRA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Recurrencia de compra',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col2,
                            'order': ['SINCATALOGAR', 'NOCOMPRADOR', 'UNICACOMPRA', 'BAJARECURRENCIA', 'RECURRENCIAMEDIA', 'GRANRECURRENCIA'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar', 'AP','Compra protegida','Exequias' ,'Renta hogar','Bolso protegido']
                            #'order_f':['Sin catalogar', 'No comprador','Unica compra','Baja recurrencia','Recurrencia media','Gran recurrencia']                   
                        }]
                    generar_graficos(df_t, configuraciones,color =2)

                    col2.subheader("Frecuencia de contacto")
                    configuraciones = [
                        {
                            'groupby': 'TIPOCLIENTE#OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tipo de cliente por numero de oportunidades',
                            # 'chart_title': 'Gráfico 3 TIPOCLIENTE#OPORTUNIDADES',
                            'col': col2,
                            'order':['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN', 
                                    'COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar','Entre 31 y 60 días','Entre 61 y 90 días','Entre 91 y 120 días',
                                       'Entre 121 y 150 días','Entre a 151 y 180 días', 'Mayores a 180 días'],

                           
                            # 'order_f':['Sin catalogar','Ni compra - ni cotiza','Solo cotizan','Cotizan mas de lo que compran',
                            #            'Compran y cotizan','Compran  mas de lo que cotizan', 'Siempre compran'],
                        }]
                    generar_graficos(df_t, configuraciones,color =3)

                    col2.subheader("Valor de prima")
                    configuraciones = [
                        {
                            'groupby': 'TIPOCLIENTE$OPORTUNIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tipo de cliente por valor de oportunidades',
                            # 'chart_title': 'Gráfico 4 TIPOCLIENTE$OPORTUNIDADES',
                            'col': col2,
                            'order':['SINCATALOGAR', 'NICOMPRA-NICOTIZA', 'SOLOCOTIZAN', 'COTIZANMASDELOQUECOMPRAN', 
                                    'COMPRANYCOTIZAN', 'COMPRANMASDELOQUECOTIZAN', 'SIEMPRECOMPRAN'],
                            'order_f':['Sin catalogar','Menos a 40 mil','Entre 40 mil y 60 mil','Entre 60 mil y 80 mil',
                                       'Entre 80 mil y 100 mil','Entre 100 mil y 120 mil', 'Mayor a 120 mil']   # Orden deseado de las categorías
                        }]
                    generar_graficos(df_t, configuraciones,color =4)

                with tab4:            
                    st.write("")
                    st.write("")
                    
                    col31,col32,col33 = st.columns(spec=[1,5,1])  
                    
                    df_c = ob.Agrupar_actividades('ACTIVIDADPRINCIPAL(EMIS)')
                                        
                    # --------------------------------------- 16/06/23                    
                    
                    configuraciones = [
                        {
                            'groupby': 'ACTIVIDADES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Sector económico',
                            'col': col32,
                            'order':  ['SERVICIOS','AGROPECUARIO', 'INDUSTRIAL', 'TRANSPORTE',  'COMERCIO','FINANCIERO','CONSTRUCCION' ,'ENERGETICO','COMUNICACIONES'],
                            'order_f':['Generación Baby Boomers','Millennials',  'Generación Z','Transporte' , 'Generación X','Generación Baby boomers', 'Construcción','Energético','Comunicaciones']
                        }
                        ]
               
                    col32.subheader("Generación digital")
                    # ob.generar_graficos_pie(configuraciones)
                    col32.plotly_chart(ob.generar_graficos_pie(configuraciones,paleta=1,width =500,height=300),use_container_width=True)
                    
                    # -------------------------------------------
                                     
                    # Configuraciones de los gráficos barras
                    
                    configuraciones = [
                        {
                            'groupby': 'TAMANOEMPRESA',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Tamaño de la empresa',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col32,
                            'order': ['SINCATALOGAR', 'PEQUENAEMPRESA', 'MEDIANAEMPRESA', 'GRANEMPRESA'],  # Orden deseado de las categorías
                            'order_f':['Sin catalogar', 'Profesional', 'Tecnólogo','Bachiller']   # Orden deseado de las categorías
                        },
                        {
                            'groupby': 'CATEGORIZACIONSECTORES',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Categoria del sector',
                            # 'chart_title': 'Gráfico 2 RangoRecurrenciaCompra',
                            'col': col32,
                            'order': ['SINCATALOGAR', 'OTROSSECTORES', 'SECTORALTOVALOR'] ,
                            'order_f':['Sin catalogar', 'Empleado', 'Independiente']   # Orden deseado de las categorías    
                        },
                        {
                            'groupby': 'ESTATUSOPERACIONAL',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Estatus operacional',
                            # 'chart_title': 'Gráfico 3 TIPOCLIENTE#OPORTUNIDADES',
                            'col': col32,
                            'order': ['NOSECONOCEELESTATUS', 'BAJOINVESTIGACIONLEGAL', 'OPERACIONAL'] , # Orden deseado de las categorías
                            'order_f': ['No se conoce el estatus', 'Bajo investigacion legal',  'Operacional']   # Orden deseado de las categorías
                        }]
                    
                    col32.subheader("Nivel educativo")
                    generar_graficos(df_c, configuraciones[0:1],color =3)

                    col32.subheader("Situación laboral")
                    generar_graficos(df_c, configuraciones[1:2],color =4)
# Demograficas
                    st.write("")
                    st.write("")

                    col000, col0, col002, col003 = st.columns(
                        spec=[0.35, 5, 1, 0.25])

                    # Configuraciones de los gráficos
                    configuraciones = [
                        {
                            'groupby': 'CATEGORIADEPARTAMENTO',
                            'count_col': 'NIT9',
                            'x_axis_title': 'Cantidad de clientes',
                            'y_axis': 'Categoria de departamento',
                            # 'chart_title': 'Gráfico 1 Ventas/Rango de Compra($)',
                            'col': col0,
                            # Orden deseado de las categorías
                            'order': ['NOSECONOCEELDEPARTAMENTO', 'OTROSDEPARTAMENTOS', 'COSTA', 'CUNDINAMARCA', 'BOGOTADC'],
                            # Orden deseado de las categorías
                            'order_f': ['No se conoce el departamento',  'Otros departamentos',  'Costa',  'Cundinamarca',  'Bogotá DC']

                        }]
                    # col0.subheader("Departamento")
                    # col0.plotly_chart(ob.generar_graficos_pie(
                    #     configuraciones, paleta=1), use_container_width=True)
                    # col002.write("")
                    # # col002.write("")
                    # col002.write("Descargar:")
                    # col002.write("")

                    # keys = ['No se conoce el departamento',   'Otros Departamentos',  'Costa',
                    #         'Cundinamarca',  'BOGOTADC']  # Orden deseado de las categorías
                    # values = ['No se conoce el departamento',  'Otros deptos',  'Costa',
                    #           'Cundinamarca',  'Bogotá DC']   # Orden deseado de las categorías

                    # dic_dep = dict(zip(keys, values))
                    # # print( 'uniques',Xf['CATEGORIADEPARTAMENTO'].unique() )
                    # Xf['CATEGORIADEPARTAMENTO'] = Xf['CATEGORIADEPARTAMENTO'].replace(
                    #     dic_dep)

                    # botones_descarga(
                    #     Xf=Xf, variable='CATEGORIADEPARTAMENTO', col=col002)

                # with tab5:

                #     df_c = ob.Agrupar_actividades('ACTIVIDADPRINCIPAL(EMIS)')
                #     df_g = df_c.groupby(by=['DEPARTAMENTO', 'ACTIVIDADES'], as_index=False)[
                #         ['OPORTUNIDADESVENDIDAS', 'OPORTUNIDADESCOTIZADAS($)']].sum()

                #     col51, col52, col53 = st.columns(spec=[0.35, 5, 0.6])

                #     col52.write('')
                #     col52.write('')
                #     col52.subheader(
                #         "Actividad económica - departamento - ventas - cotizaciones")
                #     col52.write('')
                #     scatter_plot(df_g, col=col52)

            except UnboundLocalError:
                st.warning(
                    'No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')

#     if boton_c == True:    # MODELO UNITARIO

#         with :  # Modelo Unitario
#             # st.balloons()
#             try:
#                 st.write("")
#                 # st.write(dataframe.head())

#                 Xi, Xf = ob_u.predict_proba()
#                 # st.write(Xf)

#                 Xf.replace({'INSTALACIONES': 'Instalaciones',
#                             'MANTENIMIENTO': 'Mantenimiento',
#                             'ILUMINACION': 'Iluminación',
#                             'ESTUDIOS': 'Estudios',
#                             'FIBRA_OPTICA': 'Fibra óptica',
#                             'AUMENTOS_CARGA': 'Aumentos de carga',
#                             'REDESELECTRICAS': 'Redes eléctricas',
#                             'CUENTASNUEVAS': 'Cuentas nuevas'
#                             }, inplace=True)

# # -----------------------------------------------------14/06/23

#                 colU1, colU2, colU3 = st.columns(spec=[1, 1, 1])

#                 tamaño1 = 35
#                 color_rec = '#9e9ac8'
#                 # color_prob =

#                 # Centrar el contenido de colU1

#                 colU1.write(
#                     f'<p style="text-align: center;color: {color_rec};">Primer recomendación</p>', unsafe_allow_html=True)

#                 # colU1.subheader(str(Xf.loc[0,'Producto_1']))

#                 # ; color: {color1}
#                 colU1.markdown(
#                     f'<h1 style="text-align: center; font-size: {tamaño1}px;">{str(Xf.loc[0,"Producto_1"])}</h1>', unsafe_allow_html=True)
#                 tx_proba1 = str(Xf.loc[0, 'Probabilidad_1']) + " probabilidad " + '(' + str(
#                     np.round(Xf.loc[0, 'Valor_probabilidad1'] * 100, 2)) + ' %)'
#                 colU1.write(
#                     f'<p style="text-align: center;">{tx_proba1}</p>', unsafe_allow_html=True)

#                 # Centrar el contenido de colU2
#                 colU2.write(
#                     f'<p style="text-align: center;color: {color_rec};">Segunda recomendación</p>', unsafe_allow_html=True)
#                 # colU2.subheader(str(Xf.loc[0,'Producto_2']))
#                 # tamaño1=40
#                 # ; color: {color1}
#                 colU2.markdown(
#                     f'<h1 style="text-align: center; font-size: {tamaño1}px;">{str(Xf.loc[0,"Producto_2"])}</h1>', unsafe_allow_html=True)

#                 tx_proba2 = str(Xf.loc[0, 'Probabilidad_2']) + " probabilidad " + '(' + str(
#                     np.round(Xf.loc[0, 'Valor_probabilidad2'] * 100, 2)) + ' %)'
#                 colU2.write(
#                     f'<p style="text-align: center;">{tx_proba2}</p>', unsafe_allow_html=True)

#                 # Centrar el contenido de colU3
#                 colU3.write(
#                     f'<p style="text-align: center;color: {color_rec};">Tercer recomendación</p>', unsafe_allow_html=True)
#                 # colU3.subheader(str(Xf.loc[0,'Producto_3']))
#                 # ; color: {color1}
#                 colU3.markdown(
#                     f'<h1 style="text-align: center; font-size: {tamaño1}px;">{str(Xf.loc[0,"Producto_3"])}</h1>', unsafe_allow_html=True)

#                 tx_proba3 = str(Xf.loc[0, 'Probabilidad_3']) + " probabilidad " + '(' + str(
#                     np.round(Xf.loc[0, 'Valor_probabilidad3'] * 100, 2)) + ' %)'
#                 colU3.write(
#                     f'<p style="text-align: center;">{tx_proba3}</p>', unsafe_allow_html=True)

#                 st.write('')
#                 st.write('')
#                 st.write('')
#                 download_excel(Xf, 'Resultado')

#     #             )
#             except UnboundLocalError:
#                 st.warning(
#                     'No ha cargado un archivo para procesar!. En el menú de la izquierda cargar archivo en la sección Modelo Múltiples Variables')


if __name__ == '__main__':
    main()
