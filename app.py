import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --------------------- Carrega e Limpa o DataFrame ---------------------

# Carrega a base original e mantém cópia
df_raw = pd.read_csv(r"C:\Users\gvbb2\Desktop\Data Science\Projetos\Car Price Prediction using StreamLit\Database\car_price_prediction.csv")
df = df_raw.copy()

# Limpeza - Mileage
df['Mileage'] = df['Mileage'].str[:-3].astype(int)

# Limpeza - Doors
mask = df['Doors'].isin(['04-May', '02-Mar'])
df.loc[mask, 'Doors'] = df.loc[mask, 'Doors'].str[:2]
df = df[df['Doors'] != '>5']
df['Doors'] = df['Doors'].astype(int)

# Remover outliers pontuais
i = df[(df.ID == 45812886) | (df.Price < 1000)].index
df.drop(i, inplace=True)

# Criar a coluna log_price
df["Price_log"] = np.log1p(df["Price"])

# Criar a coluna log_mileage
df["Mileage_log"] = np.log1p(df["Mileage"])

# Limpeza - Engine Volume
df["Turbo"] = df['Engine volume'].str.contains('Turbo')
df["Engine volume"] = df["Engine volume"].str.replace(' Turbo', '', regex=False)
df['Engine volume'] = df['Engine volume'].astype(float)
df = df[(df['Engine volume'] >= 1.0) & (df['Engine volume'] < 20)]
df['Engine volume'] = df['Engine volume'].astype(str)

# Limpeza - Manufacturer
# Remoção de dois SAAB
ids = [45791286, 45807151]
df = df[~df['ID'].isin(ids)]

# Limpeza - Airbags
df['Airbags'] = df['Airbags'].astype(str)

# Limpeza - Cylinders
df['Cylinders'] = df['Cylinders'].astype(str)

# --------------------- LabelEncoder + mapeamentos ---------------------

# Instancia encoder
le_manufacturer = LabelEncoder()
le_model = LabelEncoder()
le_engine = LabelEncoder()
le_airbags = LabelEncoder()
le_cylinders = LabelEncoder()

# Aplica codificação
df["Manufacturer_enc"] = le_manufacturer.fit_transform(df["Manufacturer"])
df["Model_enc"] = le_model.fit_transform(df["Model"])
df["Engine volume_enc"] = le_engine.fit_transform(df["Engine volume"])
df["Airbags_enc"] = le_airbags.fit_transform(df["Airbags"])
df["Cylinders_enc"] = le_cylinders.fit_transform(df["Cylinders"])

# Cria os dicionários reversos (nome → código)
manufacturer_map = dict(zip(le_manufacturer.classes_, le_manufacturer.transform(le_manufacturer.classes_)))
model_map = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
# --------------------- Configurações do Streamlit ---------------------

# Sidebar para seleção de páginas
st.sidebar.title("Configurações")
pages = {
    "🧽 Data Cleaning": "cleaning",
    "📊 Análise Exploratória": "explore",
    "🤖 Modelagem ML": "model",
    "🔮 Fazer Previsões": "predict"
}
selected_page = st.sidebar.radio("Navegação", list(pages.keys()))

# Título principal
st.title("Car Price Prediction Using StreamLit")

# --------------------- Página: Data Cleaning ---------------------
if pages[selected_page] == "cleaning":
    st.subheader("Data Cleaning")
    
    st.code(f"O data frame bruto contém {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas!")
  
    st.dataframe(df_raw)

    st.dataframe(df_raw.dtypes)
    
    st.write("A seguir, mostro o passo a passo das limpezas aplicadas para gerar a base final:")

    # Mileage
    st.success(f'✅ Para a coluna Mileage, removi os 3 últimos caracteres (_km) e alterei o tipo para {df["Mileage"].dtype}')

    # Doors
    st.success(f'✅ Já para a coluna Doors, removi os 4 últimos (-May ou -Mar) e excluí o valor >5. Também, converti o tipo para {df["Doors"].dtype}')

    # Engine Volume
    st.success(f'✅ Na coluna Engine Volume, primeiro criei a coluna booleana de Turbo, removi a palavra Turbo da coluna original e transformei em {df["Engine volume"].dtype}')

    # Price_log
    st.success('✅ Criei a coluna Price_log para comprimir a escala de preços e reduzir assimetria')
    
    # Mileage_log
    st.success('✅ Também, a coluna Mileage_log foi criada para comprimir a escala de quilometragem')
    
    # Label Encoding
    st.success('✅ Por fim, codifiquei as categorias Manufacturer, Cylinders, Airbags, Engine Volume e Model em numerais usando LabelEncoder')

    # Verificando valores nulos
    st.write("Agora, checarei por valores nulos:")
    n_columns = df.shape[1]
    missing_data = df.isnull().sum() / n_columns * 100
    pctg_miss = pd.DataFrame({"Porcentagem de Dados Faltantes (%)": missing_data})
    st.dataframe(pctg_miss)

    st.success(f"Como é possível ver no data frame acima, não há dados faltantes nessa base de dados")

    # Estatísticas gerais
    st.dataframe(df.describe())

    st.warning(f"⚠️Mais de 19 mil linhas e os valores de preço variando entre 1 e 26 milhões de doláres?")

    # Preços anômalos
    st.dataframe(df_raw.sort_values(by='Price', ascending=False))

    st.warning(f"💡Ok... esse OPEL provavelmente está com o preço errado 😅 Removerei agora, mas utilizarei seus dados para prever seu real valor!")

    st.dataframe(df.sort_values(by='Price', ascending=False))

    # Finalização
    st.success(f"Fechamos a limpeza de dados com {df.shape[0]} linhas e {df.shape[1]} colunas!")
# --------------------- Página: Análise Exploratória ---------------------
elif pages[selected_page] == "explore":
    

    # ------------------------ Análise das Variáveis Categóricas ------------------------
    st.subheader("Análise das Variáveis Categóricas")

    # Variáveis categóricas
    df_cat_feature = df[[ 
        "Manufacturer", "Model", "Category", "Leather interior", "Fuel type",
        "Gear box type", "Color", "Wheel"
    ]]

    selected_cat_feature = st.selectbox("Escolha uma Feature Categórica:", df_cat_feature.columns)

    contagem = df_cat_feature[selected_cat_feature].value_counts()
    num_categorias = len(contagem)

    if num_categorias >= 10:
        contagem = contagem.nlargest(10)
        titulo = f"Top 10 {selected_cat_feature}"
    else:
        contagem = contagem.nlargest(num_categorias)
        titulo = f"Top {num_categorias} {selected_cat_feature}"

    contagem = contagem.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(contagem.index, contagem.values)
    ax.set_title(titulo)
    ax.set_xlabel('Contagem')
    ax.set_ylabel(selected_cat_feature)
    st.pyplot(fig)

    # ------------------------ Análise das Variáveis Categóricas e Numéricas ------------------------

    st.subheader("Análise das Variáveis Numéricas")

    # Separando variáveis contínuas e discretas
    variaveis_continuas = ["Mileage_log", "Price_log"]
    variaveis_discretas = ["Prod. year", "Airbags", "Doors", "Cylinders", "Engine volume"]

    tipo_grafico = st.radio(
        "Escolha o tipo de variável para análise:",
        ("Contínuas (Boxplot)", "Discretas (Barplot)")
    )

    if tipo_grafico == "Contínuas (Boxplot)":
        selected_continua = st.selectbox("Escolha uma variável contínua:", variaveis_continuas)

        # Boxplot para Mileage e log_price
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[selected_continua], ax=ax)
        ax.set_title(f"Boxplot de {selected_continua}")
        ax.set_xlabel(selected_continua)
        ax.set_ylabel("Valor")
        st.pyplot(fig)

    else:
        selected_discreta = st.selectbox("Escolha uma variável discreta:", variaveis_discretas)

        if selected_discreta == "Prod. year":
            # Agrupando anos por década
            df["Decade"] = (df["Prod. year"] // 10) * 10
            contagem_decada = df["Decade"].value_counts().sort_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(contagem_decada.index.astype(str), contagem_decada.values)
            ax.bar_label(bars, padding=3)
            ax.set_title("Distribuição de Carros por Década")
            ax.set_xlabel("Década")
            ax.set_ylabel("Quantidade")
            st.pyplot(fig)

        elif selected_discreta == "Engine volume":
            # Top 10 Engine volumes mais frequentes
            topn = df["Engine volume"].value_counts().nlargest(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(topn.index.astype(str), topn.values)
            ax.bar_label(bars, padding=3)
            ax.set_title("Top 10 Engine volumes")
            ax.set_xlabel("Engine volume (L)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        else:
            # Barplot para Airbags, Doors e Cylinders
            contagem = df[selected_discreta].value_counts().sort_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(contagem.index.astype(str), contagem.values)
            ax.bar_label(bars, padding=3)
            ax.set_title(f"Distribuição de {selected_discreta}")
            ax.set_xlabel(selected_discreta)
            ax.set_ylabel("Quantidade")
            st.pyplot(fig)

# ------------------------ Modelagem ML ------------------------
elif pages[selected_page] == "model":
    st.subheader("Machine Learning - Random Forest")
  

    # ——— Definição do alvo e das features disponíveis ———
    target = "Price_log"
    features = st.multiselect(
        "Selecione as Features:",
        [
            "Mileage", "Engine volume_enc", "Prod. year",
            "Airbags_enc", "Doors", "Cylinders_enc", "Turbo",
            "Manufacturer_enc", "Model_enc"
        ],
        default=["Mileage", "Engine volume_enc", "Prod. year"]
    )

    # ——— Hiperparâmetros do Random Forest ———
    n_estimators = st.number_input("n_estimators", 50, 500, 100, step=50)
    max_depth    = st.number_input("max_depth",    1,  20,   5)

    # ——— Escolha do tamanho do conjunto de teste ———
    test_size_pct = st.slider("Tamanho do teste (%)", 10, 50, 20) / 100

    # ——— Botão de treinamento ———
    if st.button("Treinar modelo"):
        # 1) Prepara os dados
        X = df[features]
        y = df[target]

        # 2) Split treino/teste usando a proporção escolhida
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size_pct,
            random_state=42
        )

        # 3) Instancia e treina o Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 4) Predição e métricas
        y_pred = model.predict(X_test)
        mae   = mean_absolute_error(y_test, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        r2    = r2_score(y_test, y_pred)

        st.markdown("### Métricas de Desempenho")
        st.write(f"• MAE:  {mae:.2f}")
        st.write(f"• RMSE: {rmse:.2f}")
        st.write(f"• R²:   {r2:.2f}")

        # 5) Gráfico Real vs. Predito
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="red")
        ax.set_xlabel("Valores reais (log scale)")
        ax.set_ylabel("Previsões (log scale)")
        ax.set_title("Real vs. Predito")
        st.pyplot(fig)

        # 6) Importância das features
        importances = model.feature_importances_
        fi = pd.Series(importances, index=features).sort_values()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(fi.index, fi.values)
        ax2.set_title("Importância das Features")
        ax2.set_xlabel("Score")
        st.pyplot(fig2)

# ------------------------ Página: Fazer Previsões ------------------------
elif pages[selected_page] == "predict":
    st.subheader("🔮 Fazer Previsão de Preço")

    st.info("Escolha os valores para cada feature usada no modelo.")

    # Lista de features usadas
    features = [
        "Mileage", "Prod. year",
        "Manufacturer_enc", "Model_enc"
    ]

    # Inicializa o dicionário de entrada
    input_data = {}

    # Cria duas colunas para os inputs
    col1, col2 = st.columns(2)

    with col1:
        input_data["Mileage"] = st.number_input("Quilometragem (Mileage)", min_value=0, value=100_000, step=1000)
        input_data["Prod. year"] = st.number_input("Ano de fabricação", min_value=1940, max_value=2025, value=2015)

    with col2:
        manufacturer_name = st.selectbox("Montadora (Manufacturer)", options=sorted(manufacturer_map.keys()))
        # Filtra apenas os modelos daquela montadora
        modelos_disponiveis = df[df['Manufacturer'] == manufacturer_name]['Model'].unique()
        model_name = st.selectbox("Modelo (Model)", options=sorted(modelos_disponiveis))
        
    # Converte nomes reais para códigos
    input_data["Manufacturer_enc"] = manufacturer_map[manufacturer_name]
    input_data["Model_enc"] = model_map[model_name]

    # Mostra os valores digitados
    input_df = pd.DataFrame([input_data])
    st.write("📥 Valores de entrada:")
    st.dataframe(input_df)

    # Botão para fazer a previsão
    if st.button("Prever Preço"):
        try:
            # Treina o modelo (ou futuramente reutilize o já treinado)
            X = df[features]
            y = df["Price_log"]

            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
            model.fit(X, y)

            # Faz a previsão
            log_pred = model.predict(input_df)[0]
            pred_price = np.expm1(log_pred)

            st.success(f"💰 Preço estimado: **${pred_price:,.2f}**")
        except Exception as e:
            st.error(f"Erro ao fazer previsão: {e}")