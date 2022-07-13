Este repositório contém arquivos relacionados à 9ª Competição de Machine Learning realizada pela FLAI em junho de 2022.

A competição em questão era sobre um problema de previsão de demanda de aluguéis de bicicletas. Foi fornecido um conjunto de treinamento com o histórico de características e as respectivas demandas. O desafio era predizer demandas para um conjunto de teste. Para avaliação na competição a métrica adotada foi a RMSE (raiz quadrada da média dos erros quadráticos)


A dinâmica da 9ª competição foi bastante curiosa, pois trabalhei o tempo todo considerando "boas práticas" no tratamento dos dados visando a melhoria do score na validação cruzada (CV), porém, para minha frustação, ao enviar as subs o resultado não melhorava.

Na última sub "chutei o balde" e retirei um tratamento de outliers que estava fazendo e enviei a sub considerando todos os dados, ou seja, incluindo os outliers. Obtive o meu melhor score, porém descobri isso somente com o anúncio dos vencedores da competição!

Depois de finalizada a competição ainda fiz outras avaliações para tentar entender o motivo da melhoria, porém ainda não fechei as conclusões.

O código aqui apresentado não é usual em exemplos de códigos de machine learning. Boa parte das tarefas foram encapsulada em funções, assim o entendimento do código pode não ser simples para usuários com menos experiência em python, ou em linguagens de programação de forma geral.

Para faciliar o entendimento do código, considere a estrutura abaixo ilustrada.

- Como o notebook está estruturado
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
