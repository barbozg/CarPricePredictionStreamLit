# **Previsão de Preços de Carros com Streamlit: Um Estudo Prático com Aprendizado Real**

## **Resumo**

Este projeto visa prever os preços de veículos com base em seus atributos técnicos e categóricos, utilizando aprendizado de máquina (ML) integrado com a interface interativa do **Streamlit**. A jornada incluiu desde a limpeza de dados até a criação de uma aplicação web funcional. Ao longo do caminho, foi fundamental lidar com limitações da base de dados e refletir sobre conceitos clássicos como *“Garbage in, Garbage out”*.

## **1. Introdução**

 O objetivo foi criar uma aplicação simples, porém funcional, que permitisse analisar, modelar e prever o valor de um carro com base em seus atributos.

A solução foi construída com:
- **Python + Streamlit** para a interface web,
- **Scikit-learn** para modelagem de machine learning,
- **Pandas, Seaborn e Matplotlib** para análise de dados e visualizações.

## **2. Limpeza de Dados: Superando uma Base Problemática**

A base `car_price_prediction.csv` apresentava diversos desafios:
- **Inconsistência de formatos**: valores como “Mileage” e “Doors” vinham com strings misturadas (ex: "04-May", "180_000 km"),
- **Outliers gritantes**: carros com preços acima de $20 milhões, incluindo modelos de marcas como OPEL, claramente errôneos,
- **Distribuição altamente desbalanceada**: poucas observações de marcas como Ferrari e muitas de Hyundai distorciam os resultados em particionamentos treino/teste.

### Ações de limpeza:
- Conversão de *Mileage* para inteiro (remoção de "_km"),
- Padronização da coluna *Doors* (remoção de valores textuais),
- Criação de variáveis transformadas em escala logarítmica: `Price_log` e `Mileage_log` (para reduzir assimetrias),
- Extração de uma variável booleana `Turbo` e limpeza da coluna `Engine volume`,
- Remoção de IDs duplicados ou inconsistentes,
- Aplicação de `LabelEncoder` para variáveis categóricas como fabricante, modelo e número de airbags.

Essas ações foram essenciais para tornar os dados minimamente adequados à modelagem.

## **3. Análise Exploratória: Entendendo o Terreno**

O módulo de **Análise Exploratória** permitiu examinar a distribuição de variáveis categóricas e numéricas:
- Frequência das montadoras, tipos de combustível, câmbios, etc.
- Distribuições de variáveis contínuas e discretas por meio de **boxplots** e **barplots**.
- Análise temporal simplificada agrupando o ano de produção por décadas.

Isso possibilitou insights importantes como:
- A alta concentração de carros produzidos entre décadas,
- A prevalência de alguns volumes de motor e número de airbags.

## **4. Modelagem Preditiva: Random Forest em Ação**

A modelagem foi feita usando o algoritmo **Random Forest Regressor**, selecionado por sua robustez e capacidade de lidar bem com dados mistos (categóricos e contínuos).

### Parametrizações permitidas ao usuário:
- Seleção das variáveis de entrada,
- Definição do tamanho do conjunto de teste,
- Ajuste dos hiperparâmetros `n_estimators` e `max_depth`.

### Métricas avaliadas:
- **MAE** (Erro Absoluto Médio),
- **RMSE** (Raiz do Erro Quadrático Médio),
- **R² Score**.

### Visualizações complementares:
- Gráfico de dispersão entre valores reais vs. previstos,
- Gráfico de importância das variáveis.

**Problema enfrentado**: A divisão entre treino e teste, ao ser aleatória, frequentemente deixava marcas como *Ferrari* apenas em um dos conjuntos, dificultando a generalização e enviesando o modelo. Isso exemplifica bem a teoria **"Garbage in, Garbage out"** — se a entrada de dados é desequilibrada, não há modelo que salve.

## **5. Interface de Previsão: Streamlit e Interatividade**

Por fim, a aba de **previsão** permite que o usuário inserisse características de um carro fictício e obtivesse o preço estimado.

### Funcionalidades:
- Inputs para `Mileage`, `Ano`, `Manufacturer` e `Model`,
- Conversão automática dos nomes para codificações numéricas,
- Previsão baseada em `RandomForest` com `Price_log` e retorno ao valor original com `expm1`.

Essa etapa transforma o modelo técnico em uma ferramenta prática e acessível, mesmo para quem não entende de machine learning!

## **6. Dificuldades e Reflexões**

Esse projeto revelou desafios reais enfrentados em ciência de dados:
- **Qualidade da base de dados**: Mesmo com milhares de registros, dados inconsistentes tornam o modelo frágil,
- **Desequilíbrio nas classes**: Muitas observações de um tipo e poucas de outro induzem viés,
- **Limitações de generalização**: Um modelo que nunca viu uma Ferrari não sabe prever seu preço — ele "chuta".

Este cenário destaca a importância de:
- Métodos de validação mais robustos,
- Técnicas para balanceamento de classes ou *augmentation*,
- Curadoria humana contínua dos dados.

## **7. Conclusão e Perguntas para Reflexão**

Criar um modelo preditivo funcional exige mais do que algoritmos — exige **visão crítica sobre os dados**.

Neste projeto, aprendi que:
- A limpeza e transformação dos dados é o coração de qualquer pipeline de ML.
- A escolha do modelo deve considerar não só performance, mas interpretabilidade e robustez.

**E você?**  
- O que faria ao encontrar uma base com tantos ruídos?
- Utilizaria outro modelo? Tentaria balancear os dados de outra forma?
- Como garantiria uma melhor generalização?

A ciência de dados é mais do que técnica — é **investigação contínua, adaptação e aprendizado**.

## **8. Tutorial para rodar o projeto**
- Caso não tenha, baixe as bibliotecas utilizadas
  pip install streamlit pandas numpy matplotlib seaborn scikit-learn
- Execute o comando: streamlit run app.py

