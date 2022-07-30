
<div id='top'/>  




# 9ª Competição de Machine Learning FLAI
Este repositório contém arquivos relacionados à 9ª Competição de Machine Learning realizada pela [FLAI](https://www.flai.com.br/) em junho de 2022. 

O trabalho aqui apresentado conquistou a 3ª posição na classificação geral, sendo o prêmio o livro [Análise de Séries Temporais: Modelos Lineares Univariados (Volume 1)](https://www.amazon.com.br/An%C3%A1lise-S%C3%A9ries-Temporais-Lineares-Univariados/dp/8521213514/ref=asc_df_8521213514/?tag=googleshopp00-20&linkCode=df0&hvadid=379712528301&hvpos=&hvnetw=g&hvrand=7634287281072666290&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1001773&hvtargid=pla-810564891759&psc=1).

A competição era sobre um problema de previsão de demanda de aluguéis de bicicletas, o qual é um problema de regressão dentro do contexto de machine learning. Para treinar um modelo foi fornecido um conjunto de dados com o histórico de características e as respectivas demandas de aluguéis. O desafio era predizer as demandas de aluguéis para um outro conjunto de dados. Para a avaliação de desempenho na competição e comparação de resultados entre os demais participantes foi usada a métrica de RMSE (raiz quadrada da média dos erros quadráticos).


*******
**Conteúdo**
 1. [Detalhes técnicos da competição](#comp_details)
 2. [Como o trabalho foi desenvolvido](#work_structure)
 3. [Principais insights](#insights)
 4. [Modelo de Machine Learning](#ml_model)
 5. [Principais aprendizados](#learnings)
 6. [Ideias para melhorias](#improvements) 
 7. [Referências](#references)
 
*******
<div id='comp_details'/>  

## Detalhes técnicos da competição
A 9ª Competição de Machine Learning foi lançada pela FLAI no final de maio de 2022. Na ocasião o [prof. Ricardo Rocha](https://github.com/ricardorocha86) promoveu uma live para explicar a dinâmica da competição e também divulgou um [PDF](#1) com as regras e os _links_ pertinentes à competição.

**Problema** : Dado um conjunto de índices sobre clima pedia-se para fazer a previsão de demanda de quantidade de aluguéis de bicicleta

**Detalhes técnicos**
* Dados de treinamento: 4500 amostras formadas por 11 variáveis   

|Variáveis| Descrição | Intervalos |     
| :---: | :----:| :----:|
|hora| Faixa horária | [0, 23] |      
|dia| Dia da semana | [domingo, segunda, terça, quarta, quinta, sexta, sábado] |  
|feriado| Indica se o dia é um feriado | [sim, não] |     
|estação|  Estação do ano | [primavera, verão , outono, inverno] |     
|temperatura| Temperatura observada (graus Celsius)| [6.5, 44.6] |  
|chuva| Quantidade chuva precipitada (mm) | [0, 27.65] |   
|umidade| Umidade relativa no ar | [18,1, 92] |     
|sol| Incidência de radiação solar (?) |  [0, 3.52] | 
|visibilidade| Quanto pode ser visto num certo teste de distância (?) | [0, 97%] |      
|vento| Velocidade do vento (m/s) |  [0.25, 9.13] |   
|**aluguéis**| Variável resposta (a ser predita com os dados de teste) | [0, +$\infty$[|  


* Dados de teste: 3000 amostras formadas pelas mesmas variáveis independentes do conjunto de treinamento. 

* Métrica alvo: o modelo com o menor RMSE

**Dinâmica da competição**: Regras para cada participante

* Envio de até 10 submissões, respeitando a data limite da competição. 
* O ranking da competição era atualizado à medida em que novas submissões fossem feitas. 
* A submissão a ser considerada na classificação geral era aquela que tivesse o melhor desempenho.
<div id='work_structure'/>  


## Como o trabalho foi desenvolvido [:top:](#top) 
O código desenvolvido não é usual em exemplos de machine learning. Boa parte das tarefas foram encapsulada em funções, assim o entendimento do código pode não ser simples para usuários com menos experiência em Python, ou em linguagens de programação. Para facilitar o entendimento do código, nesta introdução são explicados como o código está estruturado. Dessa forma, com um pouco de paciência, os interessados nesse código conseguirão os motivos do notebook ter tantas funções.

Um ponto importante para entender o código é considerar que ele foi construído de forma *ad hoc* à medida que surgiam necessidades ao longo do projeto. Uma boa parte do código foi reutilizado em uma outra competição da FLAI, ou seja, nada saiu do zero nessa competição e muitas outras funções foram incorporadas. Por fim, houve uma reorganização do código para agrupar as funções por características para deixar o código mais coeso. 

Apesar do código parecer complexo, a ideia de fazê-lo dessa forma é exatamente a oposta. Da forma como o código foi montado, para análise de projetos futuros, bastará dar foco no EDA, nas funções personalizadas derivadas do EDA e nos modelos de configuração. Como todo o restante está pronto, bastará rodar as próximas etapas e avaliar os resultados. Foi exatamente isso o que aconteceu nessa competição!


A figura "Estrutura geral" ilustra como foi estruturado o trabalho durante a competição. Na verdade, a estrutura apresentada tem como objetivo ilustrar cada etapa do processo, porém não necessariamente elas foram criadas e executadas de forma sequencial. Durante os trabalhos, dependendo da necessidade, foi realizado alguma atividade ou incorporado uma nova função, assim o notebook foi crescendo à medida que os trabalhos evoluiam. Por exemplo, na tentativa de melhoria do desempenho, o trabalho de EDA era feito seguido da implementação de funções personalizadas, porém não necessariamente a função criada resultou em melhorias do desempenho. Contudo, todo o código criado foi deixado no notebook.

O código em linguagem Python desenvolvido foi dividido em dois notebooks: um para a [análise dos dados (EDA)](#2) e outro para a construção de [modelos de machine learning](#3). Para executar os notebooks foi utilizado a plataforma Google Colab.

Um resumo da EDA desenvolvida pode ser lida [aqui](EDA.md).

- Estrutura geral
	```mermaid 
    flowchart TB
        subgraph Main[Processamento dos dados e dos algoritmos de ML]
            direction TB
			subgraph SUB_EDA [EDA]
			end            

			subgraph SUB_A [Funções personalizadas]
			end            
			subgraph SUB_B [Procura por um candidato]
			end
			subgraph SUB_C [Tunagem do modelo]
			end
            subgraph SUB_D [Finalização do modelo]
            end

            SUB_EDA --> SUB_A
			SUB_A --> SUB_B
            SUB_B --> |candidato escolhido| SUB_C
            SUB_C --> |modelo escolhido| SUB_D;
        end

        subgraph Support [Análise das métricas]
            direction TB;          
            E1[Tabelas/Gráficos]
        end

		SUB_B --> |métricas| Support
		SUB_C --> |métricas| Support
		SUB_D --> |métricas| Support
	```


A primeira etapa foi a análise exploratória dos dados de treinamento (EDA -  *Exploratory Data Analysis*) com o objetivo de se familiarizar com os dados, obter os primeiros _insights_ e identificar as variáveis que continham mais informações para predição da variável resposta. Nas primeiras análises foram utilizados pacotes de automatização que geram gráficos e tabelas sobre as variáveis estudadas e que também exportam os resultados em arquivos[[4]](#4)[[5]](#5). Foi adotada essa estratégia de análise em arquivos, pois era mais fácil estudar o arquivo aberto em uma janela enquanto os códigos de notebooks python eram trabalhados em outra janela. As bibliotecas *Pandas Profiling* e *SweetViz* foram utilizadas para gerar as primeiras análises.


Durante os estudos foi identificada uma outra competição com dados semalhantes na plataforma [Kaggle](#6), assim foi possível estudar alguns notebooks, dentre os quais um deles apresentou uma [EDA](#7) bastante interessante. Dessa forma, foi incorporado ao projeto um conjunto de funções para plotar os dados climáticos agrupados por hora, por dia e por estação. 

Da análise desses gráficos foi possível identificar algumas características que levou à criação de novas variáveis e à transformação de outras. 

As "Funções personalizadas", dentro da estrutura do projeto, são as funções criadas para tratar as variáveis de treinamento cujos _insights_ foram obtidos na fase de EDA. Dessa forma, por exemplo, foram criadas funções para criar novas variáveis para serem usadas no treinamento de modelos de ML.

Como o projeto utiliza o PyCaret muitas das tarefas de preparação dos dados foi deixada para o ele resolver, porém, para buscar melhorias de desempenho, alguns tratamentos foram feitos fora do Pycaret pelas funções personalizadas. Contudo ao tratar variáveis de treinamento antes de rodar o Pycaret é necessários alguns cuidados, pois os mesmos tratamentos precisam ser feitos nas variáveis de testes, então para tornar o processo mais transparente ao longo do código criado, foram criadas algumas funções de suporte para tratar as particularidades do tratamento dos dados. Essas funções têm características semelhantes a das _pipelines_, que não foram usadas no projeto devido a algumas dificuldades de adequação do código para trabalhar em conjunto com o Pycaret. 

> PyCaret é uma biblioteca de auto-ML que permite fazer tratamento nas variáveis, fazer treinamentos de modelos de ML, fazer otimizações de modelos, analisar os modelos e ainda preparar o modelo para *deploy* da solução. Pela simplicidade de uso e apresentar diversos recursos, essa biblioteca foi usada como motor de geração dos modelos de ML.


A "Procura por um candidato", próxima etapa da estrutura do projeto, consiste em varrer uma lista de "modelos de configuração", rodar comandos de iniciação do Pycaret e listar os primeiros resultados de desempenho.

> Modelos de configuração é uma lista que contém dados sobre como o PyCaret deve ser configurado para rodar uma sessão. Cada item da lista, ou seja um modelo, indica como os dados de treinamento devem ser tratados, quais parâmetros do *setup* do PyCaret devem ser carregados, dentre outras configurações possíveis. Para tratar os modelos foi criada uma função que varre a lista e roda o Pycaret (*setup* e *compare_models*) com os dados indicados para cada modelo, gerando como resultado os modelos de machine learning treinados e as respectivas métricas obtidas. 

Da análise dos resultados dessa etapa já é possível identificar os modelos de configuração que apresentaram melhores desempenhos. Dessa forma é possível saber quais estratégias apresentam melhores desempenhos e quais devem ser descartadas, dentre as diversas tentativas listadas nos modelos de configuração. Por exemplo, dessa análise foi percebido que retirar variáveis dos modelos de configuração pioravam o desempenho da métrica, ou seja, no mínimo todas as variáveis deveriam ser usadas no treinamento dos modelos de machine learning.

Identificado um bom modelo de configuração, passa-se para a etapa de "Tunagem do modelo", que é a etapa de escolha dos hiperparâmetros dos algoritmos de machine learning. Contudo, antes da tunagem propriamente dita é necessário executar o setup do PyCaret com o modelo de configuração novamente, porém dessa vez apenas para o modelo escolhido, pois dessa forma os dados dessa configuração ficarão carregados na memória do Pycaret para execução de outras etapas.  

> Para a Tunagem do modelo foi criada uma função que encapsula a função *tune* do PyCaret e a executa para os modelos selecionadas na fase inicial de carga da configuração do Pycaret. Nos modelos testados optou-se por avaliar os cinco melhores modelos.

Da etapa de tunagem também foram capturadas as métricas para avaliação dos resultados de cada algoritmo de machine learning. Dessa análise foram escolhidos algoritmos com melhores desempenho para serem usados em misturas de algoritmos, com técnicas de *ensemble* e de *stacking*. Em geral, os melhores algoritmos foram *LightGMB* e o *Gradient Boosting*, então esses algoritmos foram usados para avaliar o desempenho das misturas. 

Apesar de o código fazer análise da mistura de *stacking*, esse tipo de mistura de algoritmo não foi avaliado durante a competição. Somente após o término da competição é que essa parte do código foi incorporada para novas avaliações.

Em paralelo aos passos até aqui descritos na construção do modelo de machine learning, sempre que uma etapa gerasse dados de desempenho do modelo, esses dados foram armazenados para avaliação posterior. A ideia dessa avaliação foi identificar o modelo que tivesse o melhor desempenho de métrica na validação cruzada e que não apresentasse evidências de overfitting. Essa análise foi feita pelo módulo "Análise de Métricas", onde além das tabelas com dados, também foram gerados alguns índices e gráficos para ajudar na identificação do melhor modelo.

Para armazenar os dados de análises, foi necessário recorrer a alguns recursos do Pycaret para recuperar dados gerados nas análises do próprio Pycaret. Dentre os dados guardados, foram recuperadas as métricas de RSME gerada pela validação cruzada, o RSME obtido ao aplicar o algoritmo com os dados de testes, reservados pelo Pycaret e não usados para treinamento do modelo. Também foram recuperados os dados do R², que foi a forma para avaliar como os modelos estavam acertando, ou errando, as predições dos dados de teste.


Em particular à análise do R² foi percebido que os modelos gerados apresentavam deficiência em acertar valores nos extremos do gráfico gerando um tipo de "achatamento" na predições. No gráfico abaixo é possível identificar à esquerda que o modelo de ML errava bastante com predições altas para valores que deveriam ser baixos e, à direita, percebe-se que há uma "barreira" que limita melhores predições.

<p align="center">
  <img src="https://github.com/mkunyosi/FLAI/blob/learning/DDS9/images/R2_withOutliers.jpg">
<p>


Somente no final da competição, eliminando um tratamento para outliers os erros à direita, apontados na figura acima, foi corrigido. Desse correção foi possível melhorar o desempenho do modelo e garantir algumas posições a mais na classificação geral da competição. A figura abaixo ilustra como a "barreira" à direita deixou de existir.


<p align="center">
  <img src="https://github.com/mkunyosi/FLAI/blob/learning/DDS9/images/R2_withOutliers02.jpg">
</p>

Comentando isso depois de finalizada a competição parece "óbvia" a conclusão que os dados de treinamento não poderiam ter tratamento de outliers, pois os valores "fora do padrão" eram exemplos de exceções, então deveriam ser usados no treinamento dos algoritmos. Contudo, essa constatação não foi óbvia durante a competição, pois sempre os que os outliers eram tratados as métricas na validação cruzada apresentavam melhores desempenhos.


Finalizando a estrutura do projeto, também foram construídas funções para salvar as configurações dos modelos construídos para uso futuro, assim os modelos poderiam ser executados sem a necessidade de novos treinamentos e também utilizando todos os passos de transformações nos dados (*pipeline* criada pelo PyCaret). Contudo, dependendo do caso, ainda havia a necessidade de tratar dados de teste antes de rodar o modelo gerado pelo Pycaret[^1], mas isso ficou encapsulado por funções que avaliavam a necessidade de rodar alguma função personalizada.

[^1]: Aqui vale outra nota importante, pois esse tratamento da funções personalizadas durante a competição não estava sendo feito de forma correta. Somente com a revisão do código foi possível identificar o problema e fazer as devidas correções.

Com o modelo salvo em arquivo do tipo *pickle*, os dados de treinamento e teste novamente eram usados para uma última análise nas métricas de desempenho. Também nessa etapa foi feita uma avaliação de predições erradas com número de aluguéis negativos. Como os algoritmos não sabiam que os números deveriam ser positivos, eventualmente algumas predições apareciam com valores negativos. Para corrigir esses valores, que eram poucos casos (menos de 20), adotou-se uma tática simples de trocar esses valores pelo menor valor observado nos dados de treinamento. Certamente essa não foi a melhor estratégia, pois poderia-se avaliar, por exemplo, os menores valores agrupados por estações, dias da semana e faixas horárias.


<div id='insights'/>  

## Principais insights  [:top:](#top) 

A dinâmica da 9ª competição foi bastante curiosa, pois havia pouco dados para analisar e o problema não era complexo. Contudo, melhorar o desempenho na classificação geral não foi uma tarefa simples.

Ao desenvolver as análises para tratamento de variáveis, o tempo todo levou-se em consideração as "boas práticas" estudadas nos cursos da FLAI e também nas experiência adquiridas em outros projetos. Contudo, um dos tratamentos aplicados, que apresentou melhorias nas métricas obtidas na validação cruzada (CV), e que na prática estava piorando a métrica com os dados de teste, era o tratamento de outliers. 

Em nenhum momento na competição foi percebido que o tratamento feito para os outliers estava "prejudicando" o desempenho com os dados de teste. Somente na última submissão, já sem muitas expectativas de melhorias, o tratamento de outliers foi retirado do "pipeline" de processamento dos dados. Para grande surpresa essa "pequena modificação" levou o modelo final, que gerava RMSE na casa de 235, a gerar o RMSE na casa de 222, em ambos os casos considerando apenas 1/3 da base de testes.

Finalizada a competição e reavaliando o que aconteceu para a melhoria de desempenho, ficou claro que os outliers não eram valores anormais, era apenas valores atípicos, porém que não deveriam ser retirados dos dados de treinamento.

Outros fatores que também melhoraram o modelo:
* Transformação de dados: na fase de EDA percebeu-se que os feriados tinham características semelhantes aos dias de domingo. Então foram alterados os registros de feriados para que os dias fossem domingo.

* Criação de novas variáveis: baseado em [notebook encontrado no Kaggle](#8), foram criadas variáveis para conter informações de dias úteis e horas de pico que resultassem em maior número de aluguéis. Essas novas variáveis contribuiram com melhorias na métrica.

* Utilização do parâmetro "feature_interaction" na configuração do Pycaret. Com esse parâmetro ativo outras variáveis foram criadas gerando melhorias no desempenho do modelo. Por exemplo, foram criadas as variáveis _"visibilidade_multiply_hora"_ e _"temperatura_multiply_hora"_, dentre algumas outras. 


Além do tratamento dos dados para treinamento, também foram feitas misturas de modelos, cujo objetivo era dimuinir a influência de _overfitting_ durante o treinamento dos modelos.

Durante os trabalhos da competição muitas das análises de CV foram baseadas nos dados gerados pelo PyCaret. Depois do _tunning_ de modelos, em geral, os melhores modelos eram o _LightGBM_ e o _GradientBoosting_, com RMSE bastante semelhantes. Para ter mais dados para a escolha do modelo, foi incluída uma medida para avaliar o RMSE considerando todos os dados de treinamento no modelo treinado apenas com parte desses dados. Nas análises, esse novo RMSE é identificado como _RMSE-train_.

> O Pycaret, antes de treinar modelos, separa os dados de treinamento em duas partes: uma para o treinamento dos modelos com a técnica de holdout e outra para avaliar o modelo treinado.  
> Logo após o treinamento de um modelo o PyCaret geralmente informa um RMSE que é valor o calculado na validação cruzada (CV).
> Ao rodar a função _predict_, sem parâmetros adicionais, o valor obtido é calculado a partir dos dados separados para teste (dados não usados no treinamento).
> É importante entender essas diferenças dos valores obtidos no PyCaret, pois é a partir deles que se pode saber se um modelo treinado está, ou não, com _overfitting_.

Com a nova medida do RMSE-train foi percebido que um modelo gerava um valor parecido ao RMSE após o CV e outro gerava um RMSE-train mais baixo (da ordem de 30%), sendo que geralmente o _GradientBoosting_ apresentava o RMSE-train menor. Dessa observação, gerou-se a suspeita de que o modelo havia memorizado os dados e havia perdido o poder de generalização (_overfitting_).

Com a expectativa de minimizar o problema de _overfitting_, o modelo final considerava a combinação dos dois melhores modelos obtidos apos a etapa de tunagem (_LightGBM_ e _GradientBoosting_).

Também foram trabalhados outros _insights_ gerados na etapa de EDA. Percebeu-se que havia falta de dados em determinados grupos de dados, tentou-se, então, criar uma heurística para preeencher os dados faltantes com a expectativa de melhoria do treinamento do modelo de machine learning. Por exemplo, foi detectado que não havia registros de dados, ou havia poucos registros, para uma determinada estação num dia da semana e isso possivelmente gerava uma distorção na curva de aluguéis do período. Como havia dados na "vizinhaça", tentou-se inserir registros a partir dos dados vizinhos para completar a falta de dados. 

Outra tentativa frustrada foi tentar tratar outlier do número de alguéis segmentando os dados por variáveis, ou seja, além de avaliar os outliers na variável dependente, também foi feita uma avaliação considerando as variáveis. As figuras abaixo ilustram os dados antes e depois do tratamento de outliers.


<p align="center">
  <img width==600 height=498 src="https://github.com/mkunyosi/FLAI/blob/learning/DDS9/images/AnaliseOutliers_orig.png">
  <br/>
  Variáveis sem tratamento de outliers.
</p>

<p align="center">
  <img width=600 height=498 src="https://github.com/mkunyosi/FLAI/blob/learning/DDS9/images/AnaliseOutliers_after.png">
  <br/>
  Variáveis com tratamento de outliers.
</p>



Outras tentativas, inclusive mais complexas, foram feitas para melhorar o desempenho na competição, porém o resultado, a princípio bom na validação cruzada, não foi constatado na base de teste da competição. 


<div id='ml_model'/>  

## Modelo de Machine Learning  [:top:](#top) 

Dentre as diversas configurações avaliadas durante a competição, o conjunto que gerou melhores resultados foi o seguinte:

```
    {'model_id': 14,
     'data':  [transf_create_dia_util, transf_create_hora_pico, transf_set_feriado],  # funções personalizadas	
     'test_data': None,  	 
     'setup':[('silent', True), ('session_id', 666),   	 # parâmetros de configuração para o PyCaret
                ('normalize',  True), ('normalize_method', "'minmax'"),  				
                ('feature_interaction', True),  
			] 
    }
```

Foram criadas variáveis para indicar se o dia era útil, se a hora era para demandas mais altas e foi feita a troca do dia de feriado por domingo.

Não foram feitas transformações na variável resposta antes de executar o Pycaret.

Ao chamar o PyCaret, foram escolhida a opção de normalização das variáveis numéricas com o método "minmax" e foi ativada a opção para criação de variáveis a partir de multiplicação de variáveis numéricas ("feature_interaction").

Dessa configuração os algoritmos com melhores resultados foram o *LightGBM* e o *GradientBoosting* misturados pela técnica de *voting*.


<div id='learnings'/>  
 
## Principais aprendizados  [:top:](#top)

* Na fase de EDA foram estudados com mais detalhes como criar gráficos usando as bibiliotecas _matplotlib.pyplot_ e _seaborn_. Mais do que escolher características dos gráficos, foi explorado como obter os dados a serem usados pelos gráficos. O aprofundamento na construção de gráficos permitiu a detecção de algumas características que ajudaram a construir novas variáveis que contribuiram na melhoria da métrica.

* A estruturação do código e a criação de diversas funções permitiu o aprofundamento em programação na linguagem Python. Também surgiram outros *insights* para melhorias futuras como a criação de módulos, ou bibliotecas, em Python que possam simplificar ainda mais a construções de projetos futuros.

* Melhor entendimento de como avaliar se um algoritmo está enviesado e apresentando overfitting com a memorização dos dados de treinamento e como gerar possíveis soluções para minimizar esse tipo de problema. 

* Para melhorias no desempenho das métricas geradas pelo modelos de machine learning, deve-se evitar criar novos dados a partir dos dados existentes, pois o esforço para implementar a ideia é grande, há riscos da ideia ser implementada de forma incorreta, há risco da implementação criar viéses que levarão à predições erradas. Em resumo, como há muitos riscos a serem controlados e ainda pode acontecer que alguns não sejam identificados, muito provavelmente, esse caminho não gerará bons resultados em pouco tempo.

* O tratamento de outliers aplicado nesse projeto não foi feito de forma adequada. Por ora, ainda não foi possível identificar com clareza o que foi feito de forma incorreta. Então fica como "lição de casa" para este autor estudar mais sobre como outliers podem interferir nos modelos de machine learning e quando devem ser tratados. 


<div id='improvements'/>  

## Ideias para melhorias [:top:](#top) 

Para melhorar mais o desempenho é necessário tratar o "achatamento à esquerda", porém ainda não sei como fazer, mas tenho algumas ideias.

Esse achatamento à esquerda ocorre quando a predição é alta, porém o valor deveria ser baixo. Note que há casos com diferença superior a 1000, isso prejudica bastante a métrica do RMSE.


<div id='references'/>  

## Referências  [:top:](#top) 
<a id="1">[1]</a> 
[Instruções da 9ª Competição de Machine Learning](https://github.com/mkunyosi/FLAI/blob/learning/DDS9/9th_competition/INSTRU%C3%87%C3%95ES%209%C2%AA%20COMPETI%C3%87%C3%83O%20DE%20MACHINE.pdf)

<a id="2">[2]</a> 
Notebook com [EDA](https://github.com/mkunyosi/FLAI/blob/learning/DDS9/Notebooks/DDS9_EDA_fim.ipynb).

<a id="3">[3]</a> 
Notebook com análises de modelos de [ML](https://github.com/mkunyosi/FLAI/blob/learning/DDS9/Notebooks/DDS9_ML_fim.ipynb).


<a id="4">[4]</a> 
Arquivo com a análise de dados gerada pelo Pandas Profiling: versão para  [download](https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/EDA/PandasProfiling.html) e versão para [visualização](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/EDA/PandasProfiling.html).

<a id="5">[5]</a> 
Arquivo com a análise de dados gerada pelo SweetViz:versão para [download](https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/EDA/SweetVizCompare.html) e versão para [visualização](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/EDA/SweetVizCompare.html).


<a id="6">[6]</a> 
[Competição](https://www.kaggle.com/competitions/bike-sharing-demand) na plaforma Kaggle para demanda de aluguéis

<a id="7">[7]</a> 
[Notebook com EDA](https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile) para a competição da plataforma Kaggle.

<a id="8">[8]</a> 
[Notebook com engenharia de variáveis](https://www.kaggle.com/code/mohitsital/top-10-bike-sharing-rf-gbm) utilizado na competição do Kaggle.



<!--

pandas-profiling
https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/PandasProfiling.html#overview

SweetViz
https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/SweetVizCompare.html

(https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/PandasProfiling.html).
(https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/SweetVizCompare.html).

--!>

