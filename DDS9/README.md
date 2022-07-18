
<div id='top'/>  

# 9ª Competição de Machine Learning FLAI
Este repositório contém arquivos relacionados à 9ª Competição de Machine Learning realizada pela [FLAI](https://www.flai.com.br/) em junho de 2022. 

O trabalho aqui apresentado conquistou a 3ª posição na classificação geral, sendo o prêmio o livro [Análise de Séries Temporais: Modelos Lineares Univariados (Volume 1)](https://www.amazon.com.br/An%C3%A1lise-S%C3%A9ries-Temporais-Lineares-Univariados/dp/8521213514/ref=asc_df_8521213514/?tag=googleshopp00-20&linkCode=df0&hvadid=379712528301&hvpos=&hvnetw=g&hvrand=7634287281072666290&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1001773&hvtargid=pla-810564891759&psc=1).

A competição era sobre um problema de previsão de demanda de aluguéis de bicicletas, o qual é um problema de regressão dentro do contexto de machine learning. Para treinar um modelo foi fornecido um conjunto de dados com o histórico de características e as respectivas demandas de aluguéis. O desafio era predizer as demandas de aluguéis para um outro conjunto de dados. Para a avaliação de desempenho na competição e comparação de resultados entre os demais participantes foi usada a métrica de RMSE (raiz quadrada da média dos erros quadráticos).


*******
Conteúdo
 1. [Detalhes técnicos da competição](#comp_details)
 2. [Como o trabalho foi desenvolvido](#work_structure)
 3. [Principais insights](#insights)
 4. [Modelo de Machine Learning](#ml_model)
 5. [Principais aprendizados](#learnings)
 
*******
<div id='comp_details'/>  

## Detalhes técnicos da competição
A 9ª Competição de Machine Learning foi lançada pela FLAI no final de maio de 2022. Na ocasião o [prof. Ricardo Rocha](https://github.com/ricardorocha86) promoveu uma live para explicar a dinâmica da competição e também divulgou um [PDF](https://github.com/mkunyosi/FLAI/blob/learning/DDS9/INSTRU%C3%87%C3%95ES%209%C2%AA%20COMPETI%C3%87%C3%83O%20DE%20MACHINE.pdf) com as regras e os _links_ pertinentes à competição.

**Problema** : Dado um conjunto de índices sobre clima pedia-se para fazer a previsão de demanda de quantidade de aluguéis de bicicleta

**Detalhes técnicos**
* Dados de treinamento: 4500 amostras formadas por 11 variáveis   
<br><br/>
**Variáveis independentes:**     
**hora**: faixa horária [0, 23]      
**dia**: dia da semana [domingo, segunda, terça, quarta, quinta, sexta, sábado]   
**feriado**: indica se o dia é um feriado [sim, não]      
**estação**:  estação do ano [primavera, verão , outono, inverno]     
**temperatura**: temperatura observada (graus Celsius)[6.5, 44.6]   
**chuva**: quantidade chuva precipitada (mm) [0, 27.65]   
**umidade**: umidade relativa no ar [18,1, 92]      
**sol**: incidência de radiação solar (?) [0, 3.52]  
**visibilidade**: quanto pode ser visto num certo teste de distância (?) [0, 97%]      
**vento**: velocidade do vento (m/s) [0.25, 9.13]    
<br><br/>
**Variável dependente:**   
**aluguéis**: variável resposta (a ser predita com os dados de teste)   

* Dados de teste: 3000 amostras formadas por 10 variáveis, sendo o desafio prever a demanda de aluguéis (variável resposta)

Mesmas variáveis independentes dos dados de treinamento.

* Métrica alvo: o modelo com o menor RMSE

**Dinâmica da competição**: Regras para cada participante

* Envio de até 10 submissões, respeitando a data limite da competição. 
* O ranking da competição era atualizado à medida em que novas submissões fossem feitas. 
* A submissão a ser considerada na classificação geral era aquela que tivesse o melhor desempenho.
<div id='work_structure'/>  


## Como o trabalho foi desenvolvido <p align="right"> [:top:](#top) </p>
O código desenvolvido não é usual em exemplos de machine learning. Boa parte das tarefas foram encapsulada em funções, assim o entendimento do código pode não ser simples para usuários com menos experiência em Python, ou em linguagens de programação. Para facilitar o entendimento do código, nesta introdução são explicados como o código está estruturado. Dessa forma, com um pouco de paciência, os interessados nesse código consiguirão os motivos do notebook ter tantas funções.

A figura "Estrutura geral" ilustra como foi estruturado o trabalho durante a competição. Na verdade, a estrutra apresentada tem como objetivo ilustrar cada etapa do processo, porém não necessariamente elas foram criadas e executadas de forma sequencial. Durante os trabalhos, dependendo da necessidade, foi realizado alguma atividade ou incorporado uma nova função, assim o notebook foi crescendo à medida que os trabalhos evoluiam. Por exemplo, na tentativa de melhoria do desempenho, o trabalho de EDA era feito seguido da implementação de funções personalizadas, porém não necessariamente a função criada resultou em melhorias do desempenho. Contudo, todo o código criado foi deixado no notebook.

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


A primeira etapa foi a análise exploratória dos dados de treinamento (EDA -  Exploratory Data Analysis) com o objetivo de se familiarizar com os dados, obter os primeiros insights e identificar as variáveis que continham mais informações para predição da variável resposta. Nas primeiras análises foram utilizados pacotes de automatização que geram gráficos e tabelas sobre as variáveis estudadas e que também exportam os resultados em arquivos[^1][^2]. Foi adotada essa estratégia de análise em arquivos, pois era mais fácil estudar o arquivo aberto em uma janela enquanto códigos de notebooks python eram trabalhados em outra janela. As bibliotecas *Pandas profiling* e *SweetViz* foram utilizadas para gerar as primeiras análises.

   
[^1]: Arquivo gerado pelo [pandas-profiling](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/PandasProfiling.html)
[^2]: Arquivo gerado [SweetViz](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/SweetVizCompare.html)


Durante os estudos foi identicada uma outra competição com dados semalhantes na plataforma Kaggle, assim foi possível estudar alguns notebooks, dentre os quais um deles apresentou uma [EDA](https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile) bastante interessante. Dessa forma, foi incorporado ao projeto um conjunto de funções para plotar os dados climáticos agrupados por hora, por dia e por estação. 

Kaggle
https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile

Da análise desses gráficos foi possível identificar algumas características que levou à criação de novas variáveis e à transformação de outras. 


As "Funções personalizadas", dentro da estrutura do projeto, são as funções criadas para tratar as variáveis de treinamento cujos insights foram obtidos na fase de EDA. Dessa forma, por exemplo, foram criadas funções para criar novas variáveis para serem usadas no treinamento de modelos de ML.

Como o projeto utiliza o PyCaret muitas das tarefas de preparação dos dados foi deixada para o ele resolver, porém, para buscar melhorias de desempenho, alguns tratamentos foram feitos fora do Pycaret pelas funções personalizadas. Contudo ao tratar variáveis de treinamento antes de rodar o Pycaret é necessários alguns cuidados, pois os mesmos tratamentos precisam ser feitos nas variáveis de testes, então para tornar o processo mais transparente ao longo do código criado, foram criadas algumas funções de suporte para tratar as particularidades do tratamento dos dados. Essas funções têm características semelhantes a das pipelines, que não foram usadas no projeto devido a algumas dificuldades de adequação do código para trabalhar em conjunto com o Pycaret. 

> PyCaret é uma biblioteca de auto-ML que permite fazer tratamento nas variáveis, fazer treinamentos de modelos de ML, fazer otimizações de modelos, analisar os modelos e ainda preparar o modelo para *deploy*. Por ter o uso bastante simples e apresentar diversos recursos, essa biblioteca foi usada como motor de geração dos modelos de ML.


A "Procura por um candidato", próxima etapa da estrutura do projeto, consiste em varrer uma lista de "modelos de configuração", rodar comandos de iniciação do Pycaret e listar os primeiros resultados de desempenho.

> Modelos de configuração é uma lista que contém dados sobre como o PyCaret deve ser configurado para rodar uma sessão. Cada item da lista, ou seja um modelo, indica como os dados de treinamento devem ser tratados, quais parâmetros do setup do PyCaret devem ser carregados, dentre outras configurações possíveis. Para tratar os modelos foi criada uma função que varre a lista e roda o Pycaret com os dados indicados para cada modelo (*setup* e *compare_models*), gerando como resultado uma tabela com as métricas obtidas para cada modelo. 

Da análise dos resultados dessa etapa já é possível identificar os modelos de configuração que apresentaram melhores desempenhos. Dessa foram é possível saber quais estratégias apresentam melhores desempenhos e quais devem ser descartadas. Por exemplo, dessa análise foi percebido que retirar variáveis dos modelos de configuração geraram desempenho pior, ou seja, no mínimo todas as variáveis deveriam ser usadas no treinamento dos modelos de ML.

Identificado um bom modelo de configuração, passa-se para a etapa de "Tunagem do modelo", que é a etapa de escolha dos hiperparâmetros dos algoritmos de machine learning. Contudo, antes da tunagem propriamente dita é necessário executar o modelo de configuração novamente, porém dessa vez apenas para o modelo escolhido, pois dessa forma os dados dessa configuração ficarão carregados na memória do Pycaret para execução de outras etapas.  

> Para a Tunagem do modelo foi criada uma função que encapsula a função *tune* do PyCaret e a executa para os modelos selecionadas na fase inicial de carga da configuração do Pycaret. Nos modelos testados optou-se por avaliar os cinco melhores modelos.

Da etapa de tunagem também foram capturadas as métricas para avaliação dos resultados de cada algorítmo de machine learning. Dessa análise foram os escolhidos algorítmos com melhores desempenho para serem usados em misturas de algorítmos, como técnicas de *ensemble* e *stacking*. Em geral, os melhores algorítmos foram *Light GMB* e o *GradientBoosting*, então esses algorítmos foram usados nas misturas na próxima etapa.

Em paralelo aos passos descritos na construção do modelo de machine learning, sempre que uma etapa gerasse dados de desempenho do modelo, esses dados foram armazenados para avaliação posterior. A ideia dessa avaliação foi identificar o modelo que tivesse o melhor desempenho de métrica na validação cruzada e que não apresentasse evidências de overfitting. Essa análise foi feita no módulo "Análise de Métricas", onde além das tabelas com dados, também foram gerados alguns índices e gráficos para ajudar na identificação do melhor modelo.

Para armazenar os dados da análise posterior, foi necessário recorrer a alguns recursos do Pycaret para recuperar dados gerados nas análises do próprio Pycaret. Dentre os dados guardados, foram usados as métricas de RMSE gerada pela validação cruzada, o RSME obtido ao aplicar o algoritmo no dados de testes, reservados pelo Pycaret e não usados para treinamento do modelo e dados do R², que foi a forma usada para avaliar como os modelos estavam acertando, ou errando, as predições dos dados de teste.

Em particular à análise do R² foi percebido que os modelos gerados apresentavam deficiência em acertar valores nos extremos dos dados, ou seja, os modelos erravam bastante com preoições altas para valores que deveriam ser baixos e também erravam as predições com valores baixos para valores que deveriam ter valores mais altos. O gráfico abaixo ilustra esse problema.

*** Gráficos enviados ao Rafa com o R2 com distorções

Somente no final da competição, eliminando um tratamento para outliers os erros no extremo de valores mais altos foi corrigido. Desse ganho foi possível melhorar o desempenho do modelo e ganhando algumas posições na classificação geral da competição.

*** Gráficos enviados ao Rafa com o R2 com distorções


Comentando isso depois de finalizada a competição parece "óbvia" a conclusão que os dados de treinamento não poderiam ter tratamento de outliers, pois os valores "fora do padrão" eram exemplos de exceções, então deveriam ser usados no treinamento dos algoritmos. Contudo, essa constatação não foi óbvia durante a competição, pois sempre os que os outliers eram tratados as métricas na validação cruzada apresentavam melhores desempenhos.


Finalizando a estrutura do projeto, também foram construídas funções para salvar as configurações dos modelos construídos para uso furuto, sem a necessidade de haver novos treinamentos dos algoritmos e também com todos os passos de transformações nos dados (pipeline). Contudo, dependendo do caso, ainda havia a necessidade de tratar dados de teste antes de rodar o modelo gerado pelo Pycaret, mas isso ficou encapsulado por funções que avaliavam a necessidade de rodar algum função personalizada.

Com o modelo salvo em arquivo do tipo peckle, os dados de treinamento e teste novamente eram usados para uma última análise nas métricas de desempenho. Também nessa etapa foi feita uma avaliação de predições erradas com número de aluguéis negativos. Como os algoritmos não sabiam que os números deveriam ser positivos, eventualmente algumas predições apareciam com valores negativos. Para corrigir esses valores, que eram poucos, adotou-se uma tática simples de trocar esses valores pelo menor valor observado nos dados de treinamento. Certamente essa não foi a melhor técnica, pois poderia-se avaliar os menores valores por estações, dias da semana e faixas horárias.

O código montado no projeto muito provavelmente não é típico de quem desenvolve projetos para machine learning. Na verdade, ele foi construído de forma *ad hoc* à medida que surgiam necessidades ao longo do projeto. Boa parte do código foi reutilizado em uma outra competição da FLAI, ou seja, nada saiu do zero nessa competição. Foram incorporadas novas funções e o código sofreu uma reorganização para agrupar as funções por características. 

Apesar do código parecer complexo, a ideia de fazê-lo dessa forma é exatamente oposto. Como o código foi montado, para análise de projetos futuros, bastará dar foco no EDA, nas funções personalizadas derivadas do EDA e nos modelos de configuração. Como todo o restante está pronto, bastará rodar as próximas etapas e avaliar os resultados. Foi exatamente isso já aconteceu nessa competição!


<div id='insights'/>  

## Principais insights

A dinâmica da 9ª competição foi bastante curiosa, pois havia pouco dados para analisar e o problema não era complexo. Contudo, melhorar o desempenho na classificação geral não foi uma tarefa simples.

Ao desenvolver as análise para tratamento de variáveis, o tempo todo levou-se em consideração as "boas práticas" estudadas nos cursos da FLAI e também por experiência em outros projetos. Um dos tratamentos aplicados, que apresentou melhorias nas métricas obtidas na validação cruzada (CV), e que na prática estava piorando a métrica com os dados de teste, era o tratamento de outliers. 

Em nenhum momento na competição foi percebido que o tratamento feito para os outliers estava "prejudicando" o desempelho com os dados de teste. Somente na última submissão, já sem muitas expectativas de melhorias, o tratamento de outliers foi retirado do "pipeline" de processamento dos dados. Para grande surpresa essa "pequena modificação" levou o modelo final, que gerava RMSE na casa de 235, passou a gerar o RMSE na casa de 222, em ambos os casos considerando apenas 1/3 da base de testes.

Outros fatores que também melhoraram o modelo:
Transformação de dados: na fase de EDA percebeu-se que os feriados tinham características semelhantes aos dias de domingo. Então foram alterados os registros de feriados para que os dias fossem domingo.

Criação de novas variáveis: baseado num notebook encontrado no Kaggle, foram criadas variáveis para conter informações de dias úteis e horas de pico que resultassem em maior número de aluguéis. Essas novas variáveis proporcionaram pequenas melhorias na métrica.

Outras variáveis criadas e que surtiram foi geradas pelo Pycaret com a inclusão do parâmetro *interaction_xpto*. De forma controlado pelo Pycaret, foram criadas variáveis cujo resultado sera a multiplicação de valores de outras variávies. Por exemplo, foram criadas as variáveis asfsaf_multiply_asfasf e asfsaf_multiply_asfasf 

citar arquivo do Kaggle  

Outras tentativas, inclusive mais complexas, foram tentatas para melhorar o desempenho na competição, porém o resultado, a princípio bom na validação cruzada, não foi constatado na base de teste da competição. 

Como no EDA percebeu-se que havia falta de dados em determinados grupos de dados, tentou-se criar uma heurística para preeencher os dados faltantes com a expectativa de melhoria do treinamento do modelo de machine learning. Por exemplo, foi detectado que não havia registros de dados, ou havia poucos registros, para uma determinada estação num dia da semana e isso possivelmente gerava uma distorção na curva de alguéis do perído. Como havia dados na "vizinhaça", tentou-se inserir registros a partir dos dados vizinhos para completar a falta de dados. 


>>> colocar gráfico de inverno com falta de dados e com distorção - análise para imputação de dados

Outra tentativa frustrada foi tentar tratar outlier do número de alguéis segmentando os dados por variáveis, ou seja, além de avalia os outliers na variável dependente, também foi feita uma avaliação considerando as variáveis. As figuras abaixo ilustram os dados antes e depois do tratamento de outliers.

>>> gráficos de outliers por variável. 

<div id='ml_model'/>  

## Modelo de Machine Learning

Dentre as diversas configurações avaliadas durante a competição, o conjunto qeu gerou melhores resultados foi o seguinte:

<<< incluir modelo de configuração vencedor

Dessa configuração os algoritmos com melhores resultados foram o *Light GBM* e o *GradientBoosting* misturados pela técnica de *voting*.



<div id='learnings'/>  
 
## Principais aprendizados

Na fase de EDA foram estudados com mais detalhes como criar gráficos usando as bibiliotecas Matplolib e seaborn. Mais do escolher características dos gráficos, foi explorado como obter os dados a serem usados pelos gráficos. O aprofundamento na construção de gráficos permitiu a detecção de algumas características que ajudaram a construir novas variáveis que contribuiram na melhoria da métrica.

A estruturação do código e a criação de diversas funções permitiu o aprofundamento em programação Python. Também surgiram *insights* para melhorias futuras como a criação de módulos, ou bibliotecas, em Python que possam simplificar ainda mais a construções de projetos futuros.

Melhor entendimento de como avaliar se um algoritmo está enviesado e apresentando overfitting com a memorização dos dados de treinamento e como gerar possíveis soluções para minimizar esse tipo de problema. 

Para melhorias no desempenho das métricas geradas pelo modelos de machine learning, deve-se evitar criar novos dados a partir dos dados existentes, pois o esforço para implementar a ideia é grande, há riscos da ideia ser implementada de forma incorreta, há risco da implementação criar viéses que levará as predições serem erradas. Em resumo, há muitos riscos a serem controlados e ainda pode acontecer que alguns não sejam identificados.

Muito provavelmente o tratamento de outliers aplicado nesse projeto não foi feito de forma adequada. Por ora, ainda não foi possível identificar com clareza o que foi feito de forma incorreta. Fica como "lição de casa" estudar mais sobre como outliers podem interferir nos modelos de machine learning e quando devem ser tratados. 



Referências:
https://www.kaggle.com/competitions/bike-sharing-demand
Kaggle https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile
Kaggle https://www.kaggle.com/code/mohitsital/top-10-bike-sharing-rf-gbm

pandas-profiling
https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/PandasProfiling.html#overview

SweetViz
https://raw.githubusercontent.com/mkunyosi/FLAI/learning/DDS9/SweetVizCompare.html

