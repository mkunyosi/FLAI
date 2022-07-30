
# 9ª Competição de Machine Learning da FLAI - Análise Exploratória dos dados (EDA)

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

**Tipos de variáveis**

* Variáveis categóricas: estação, dia, feriado
* Variáveis numéricas: temperatura, hora, sol, visibilidade, vento, umidade, chuva

**Dados faltantes**
* Não há dados faltantes nos dados de treinamento e teste


**Outliers**
* Analisando os dados de aluguéis foram identificados alguns outliers
* Agrupando os dados por estação e dia, também foram identificados outliers para número de aluguéis dentre dos agrupamentos.
* Houve tentativa de tratar os outliers majorando os valores de aluguéis nos grupos analisados.

**Correlação entre variáveis**
* Correlação positiva: temperatura, hora, sol, visibilidade, vento, estação, dia, feriado
* Correlação negativa: umidade, chuva
* Variáveis com maior correlação com a variável dependente: temperatura, hora, sol, umidade, visibilidade, estação

**Transformações na variável resposta**

* Avaliação de transformação dos valores da variável resposta com função logarítmica. Pela análise gráfica, não se percebeu  melhorias evidentes. 

* Como os dados da variável target apresentam uma curva diferente de uma normal, inclusive com uma "cauda longa", avaliou-se a possibilidade de aplicação de uma transformação logarítmica na variável resposta.

* Pela análise gráfica, não se percebe melhorias evidentes. Contudo, foram consideradas essas transformações com modelos de configurações a serem  testadas no treinamento dos modelos de machine learning.



**Cruzamento de dados: Análises**

* Distribuição média de aluguéis ao longo do dia mostra que há aluguéis nas 24 horas do dia.   
    * A partir das 5h00 há um aumento do número de aluguéis, chegando num primeiro pico às 8h00. 
    * Depois há uma leve queda permanecendo quase que constante até às 12h00, quando o número volta subir levemente.
    * A partir das 16h00 o número aumenta significativamente até às 18h00, quando, então o número de aluguéis começa a cair de forma constante. Esse padrão final se mantém até o dia seguinte.
* Distribuição de aluguéis por estação, considerando os valores médios, apresenta uma pequena diferença, sendo que há mais aluguéis no verão, seguidos por outono e primavera. No inverno o número de aluguéis é nitidamente mais baixo.
* Na distribuição de aluguéis por estação, considerando os valores máximos, praticamente não há diferença entre os dados de verão, outono e primavera. No inverno o número máximo de aluguéis novamente é nitidamente mais baixo.

> 💡 Números mostram que os dados foram colhidos em uma região onde é comum o uso de bicicletas durante todo as 24 horas do dia.  
> 💡 O pico às 8h00 pode indicar a saída de casa para ir ao trabalho e à escola. Os picos às 16h00 e 18h00 podem indicar horário de saída de escola e trabalho, respectivamente.  
> 💡 Os dados variam pouco entre as estações de verão, outono e primavera, isso pode indicar que o uso de bicicleta segue o mesmo padrão ao longo do ano, exceto no inverno.  
> ❓ Quais outros fatores poderiam influenciar o número de aluguéis mais baixo no inverno? 


* Distribuição média de aluguéis por temperatura, separada por quartis, mostra curvas semelhantes às curvas das estações.
* Distribuição máxima de aluguéis por temperatura, separada por quartis, mostra que no pico matinal (8h00) os números de aluguéis é semelhante para qualquer temperatura, porém o pico no final do dia (18h00) varia de acordo com a temperatura, sendo que quanto mais quente, há mais aluguéis.
* As temperaturas registradas variam de 6.5 a 44.6º C.

> 💡 No inverno temperatura mais baixas pode influenciar negativa o número de aluguéis.   
> ❓ Como o número máximo de alguéis no inverno é semelhante em torno das 8h00, porém cai significativamente nos demais horários, qual seria o motivo?

* Distribuição de alguéis por temperatura no inverno mostra comportamento semelhante entre 3h00 e 10h00. Nos demais períodos há indícios que o número de aluguéis é mais baixo quando a temperatura está mais baixa.
* Distribuições de alguéis analisadas nas demais variáveis (sol, vento, chuva, umidade, visibilidade e feriado) não permitem extrair insights significativos.


* Distribuição de temperaturas médias por estação ao longo do dia mostra temperaturas em faixas bem definidas
	* Inverno: próxima de 15º entre 0h00 e 9h00, elevação até 19º às 17h00 e queda nos demais horários
	* Outono e primavera: próxima de 25º/26º entre 0h00 e 7h00, elevação até 30º às 15h00 e queda nos demais horários
	* Verão: próxima de 35º entre 0h00 e 6h00, elevação até 39º às 16h00 e queda nos demais horários
	
* Distribuição de intensidades solares médias (sol) por estação ao longo do dia mostra dias mais curtos no inverno e mais longos no verão. Na primavera a intensidade solar é mais baixa comparada ao período de verão, e no outono é um pouco mais baixa do que na primavera. No inverno a intensidade solar é significativamente mais baixa.
	* Inverno: próxima de 15º entre 0h00 e 9h00, elevação até 19º às 17h00 e queda nos demais horários
	* Outono e primavera: próxima de 25º/26º entre 0h00 e 7h00, elevação até 30º às 15h00 e queda nos demais horários
	* Verão: próxima de 35º entre 0h00 e 6h00, elevação até 39º às 16h00 e queda nos demais horários

> 💡 No inverno, além da temperatura influir negativamente no número de alguéis, a intensidade solar também parece influenciar negativamente o número de aluguéis na estação.


* Distribuição de quantidade média de vento por estação ao longo do dia apresenta curvas parecidas para todas as estações. No inverno e na primavera os valores são levemente superiores aos dados do outono e verão.

* Distribuição de quantidade de chuva por estação ao longo do dia mostra que no inverno quase não chuvas e nas demais estações chove de forma desigual durante o dia. Percebe-se que no outono há um pouco menos de chuva do que na primavera e no verão.

* Distribuição de umidade por estação ao longo do dia mostra que no inverno o tempo fica mais seco e no verão mais úmido. Na primavera e no outono a umidade apresenta características semelhantes. Em todas as estações, o dia amanhece mais úmido e se torna mais cedo até à 14h00, depois volta a umidade volta a aumentar.

* Distribuição do índice de visibilidade por estação ao longo do dia mostra que na primavera a visibilidade é um pouco menor do que nas demais estações. Talvez a combinação umidade, vento e temperatura explique  essa característica. Nas demais estações a visibilidade se mostra semelhante do longo do dia.

* Distribuição de aluguéis por feriado mostra comportamentos diferentes no número de aluguéis. Nos feriados há mais alguéis entre 11h00 e 17h00, quando não é feriado há mais alguéis entre 6h00 e 11h00 e entre 17h00 e 23h00. No perído da madrugada, o número de alguéis é semelhante em todos os dias.

* Distribuição de aluguéis por dia útil mostra curvas semelhantes aos do gráfico de feriados. Dias com feriado tem comportamento semelhante de dias em finais de semana e dias sem feriado são semelhantes aos dias úteis.


* Distribuição de aluguéis por estação em feriados que ocorrem na quarta mostra que, provavelmente, o dataset de treinamento contém dados de três feriados, sendo dois ocorridos no verão e um na primavera. No gráfico de quantidades de registros é possível observar que faltam vários dados.
	* Observando outros gráficos com as quantidades de registros, nota-se que em vários horários o número de registros é baixo, em relação aos números mais altos de registros.
	* O número máximo de registros por estação foi de 56, ou seja, no dataset de treinamento há registros de pelo menos 8 semanas diferentes o que equivale a 2 meses. Como uma estação do ano tem 3 meses, conclui-se que faltam pelo dados de pelo menos 1 mês.

> ❓ Qual seria o motivo de vários dados não estar no dataset? Algum problema na coleta dos dados, ou os dados foram omitidos?  
> 💡 E se usando uma interpolação numérica os dados faltantes observados fossem preenchidos? Será que não seria possível reconstruir dados e ter um dataset completo daí para a previsão de novos não seria apenas uma consulta a esses dados dataset?   
> 💡 Observando a distribuição de aluguéis por hora no gráfico com os valores de mínimos e de máximos, percebe-se que há uma faixa que poderia limitar a previsão de novos valores. Esses limites de mínimos e máximos, então, poderiam ser usados para garantir que as predições não extrapolassem os valores conhecidos e, assim, contribuissem com valores menores no cálculo do RMSE.

