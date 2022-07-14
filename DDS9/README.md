# 9ª Competição de Machine Learning FLAI
Este repositório contém arquivos relacionados à 9ª Competição de Machine Learning realizada pela FLAI em junho de 2022. 

Nesta competição obtvie a 3ª posição na classificação geral.

A competição em questão era sobre um problema de previsão de demanda de aluguéis de bicicletas (problema de regressão). Foi fornecido um conjunto de dados com o histórico de características e as respectivas demandas (dados de treinamento). O desafio era predizer demandas para um outro conjunto de daos (dados de teste). Para avaliação na competição e comparação de resultados entre os demais participantes foi usada a métrica de RMSE (raiz quadrada da média dos erros quadráticos).

*******
Conteúdo
 1. [Detalhes da competição](#comp_details)
 2. [Como o trabalho foi estruturado](#work_structure)
 3. [Principais insights](#insights)
 4. [Modelo de Machine Learning](#ml_model)
 5. [Principais aprendizados](#learnings)
 
*******
<div id='comp_details'/>  

## Detalhes da competição
**Problema** : Trata-se de um conjunto de dados para previsão da quantidade de aluguéis de bicicleta a partir de variáveis do dia e do clima!

**Detalhes técnicos**
* Dados de treinamento (4500, 11): variável resposta “aluguéis”
<br></br>
* hora	dia	feriado	estação	temperatura	chuva	umidade	sol	visibilidade	vento	aluguéis
* Dados de teste (3000, 10): não contém a variável resposta
* Métrica alvo: o modelo com o menor RMSE

**Dinâmica da competição**
* Envio de até 10 submissões, respeitando a data limite da competição. 
* O ranking da competição era atualizado à medida em que novas submissões fossem feitas. 
* A única submissão que conta é a que tiver o melhor desempenho.

<div id='work_structure'/>  

## Como o trabalho foi estruturado
A figura "Estrutura geral" ilustra como foi estruturado o trabalho durante a competição. Inicialmente, foi feita a análise exploratória dos dados de treinamento (EDA -  Exploratory Data Analysis) com o objetivo de se buscar quais variáveis continham mais informações para predição da variável resposta.

A partir de dados 

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




A dinâmica da 9ª competição foi bastante curiosa, pois trabalhei o tempo todo considerando "boas práticas" no tratamento dos dados visando a melhoria do score na validação cruzada (CV), porém, para minha frustação, ao enviar as subs o resultado não melhorava.

Na última sub "chutei o balde" e retirei um tratamento de outliers que estava fazendo e enviei a sub considerando todos os dados, ou seja, incluindo os outliers. Obtive o meu melhor score, porém descobri isso somente com o anúncio dos vencedores da competição!

Depois de finalizada a competição ainda fiz outras avaliações para tentar entender o motivo da melhoria, porém ainda não fechei as conclusões.

O código aqui apresentado não é usual em exemplos de códigos de machine learning. Boa parte das tarefas foram encapsulada em funções, assim o entendimento do código pode não ser simples para usuários com menos experiência em python, ou em linguagens de programação de forma geral.

Para faciliar o entendimento do código, considere a estrutura abaixo ilustrada.



- Módulo: Funções personalizadas
	```mermaid 
    flowchart TB

		subgraph SUB_A [Funções personalizadas]
			direction TB;
			A1[Transformações em variáveis] --> A2[Engenharia de variáveis];
			A2 --> A3[Seleção de variáveis]			
			A3 --> A4[Tratamento de dados faltantes]
			A4 --> A5[Tratamento de outliers]			
			A5 --> A6[Transformação na variável alvo]
		end            
			
			
- Módulo: Procura por um candidato
	```mermaid 
    flowchart TB
		subgraph SUB_B [Procura por um candidato]
			direction TB;
			B1[Modelos de tratamento] --> |setup + compare_models| B2[CV Inicial];
		end

- Módulo: Tunagem do modelo
	```mermaid 
    flowchart TB
		subgraph SUB_C [Tunagem do modelo]
			direction TB;          
			C1[Modelo base] --> |setup + compare_models| C2[CV Inicial]
			C2 --> |tune| C3[CV - Otimizado]
			C3 --> |Blend|C4[CV-Blend]
			C3 --> |Stack|C5[CV-Stack]
		end

- Módulo: Finalização do modelo
	```mermaid 
    flowchart TB
		subgraph SUB_D [Finalização do modelo]
			direction TB;          
			D1[Modelo final] --> |finalize| D2[Metricas finais]
			D2 --> |ajustes finais| D3[Arquivo para submissão]
		end

- Module: Análise das métricas
	```mermaid 
    flowchart TB

        subgraph SUB_E [Análise das métricas]
            direction TB;          
            E1[métricas] --> |exportação| E2[Arquivos Excel]
            E1[métricas] --> |plotagem| E3[Gráficos]
        end



<div id='insights'/>  

## Principais insights

<div id='ml_model'/>  

## Modelo de Machine Learning



<div id='learnings'/>  
 
## Principais aprendizados
