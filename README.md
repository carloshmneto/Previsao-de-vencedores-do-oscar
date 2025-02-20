# Previsão de Vencedores do Oscar

Esse trabalho consiste no desenvolvimento e ajuste de um modelo de classificação que tem como objetivo prever o vencedor da categoria de Melhor Filme na premiação de cinema mais famosa do mundo, o Oscar. Para isso, o modelo é alimentado com os 10 indicados na categoria em uma edição determinada, com informações específicas de cada um, e devolve como saída a probabilidade calculada de cada um dos longas ser o vencedor.

## Estrutura do Projeto
- `modelagem.ipynb`: Notebook para visualização dos dados, extração de informações e treinamento do modelo em Python;
- `modelo.pkl`: Modelo final treinado com RandomForestClassifier, em formato Pickle;
- `codigo.py`: Código para previsão de filmes dada qualquer lista de filmes inclusos em input.

Para utilização do projeto localmente, é necessário o download apenas do `modelo.pkl` e o `codigo.py`, rodando o arquivo Python em um ambiente de desenvolvimento integrado na máquina e preenchendo informações requisitadas para os filmes a serem previstos. O código retornará a lista dos 10 filmes, em ordem, com suas probabilidades de vitória calculadas com base na previsão feita.

## Processo de modelagem

### Dados utilizados

Para esse projeto, foi necessária a construção de uma base de dados própria, juntando informações de filmes separadas em diversos bancos de dados públicos disponíveis para serem baixadas na internet. Elas foram as seguintes:
- `TMDB_movie_dataset_v11.csv` (The Movie DataBase): Contém dados de todos os filmes registrados na história do cinema. Dessa base, foram retiradas a maior parte das informações utilizadas para treinamento do modelo, como gêneros do filme, duração, orçamento e bilheteria, entre outros
- `golden_globe_awards.csv` (Premiação Globo de Ouro): Representa informações sobre a segunda maior premiação americana de cinema, o Globo de Ouro. Desses, foi utilizado somente uma variável binária que indica vitória ou não na categoria principal do globo de ouro, que tende a ter correlação alta com o Oscar, sempre ocorrendo anteriormente ao mesmo
- `the_oscar_award.csv` (Premiação Oscar): Base com histórico de todos os Oscars anteriores. Foi usada para criação da variável resposta do modelo, mas também para outras variáveis dependentes (como número de indicações totais)
- `title.ratings.tsv.gz` (IMDb): Contém informações de filmes no maior site de avaliação de cinema da internet, o IMDb (Internet Movie Database). Foi utilizada para as variáveis de popularidade e recepção (número de avaliações e nota média das mesmas no site)

Para junção de tais bases, foram utilizadas como pilares as colunas indicativas de nome e ano de lançamento dos filmes, apesar de que, em alguns casos, certas divergências de caligrafia e tradução de nomes tiveram que ser contornados por meios alternativos. Dentre eles, foi utilizada uma técnica de processamento da biblioteca fuzzywuzzy, na qual títulos de certa base tiveram que ser comparados com todos os títulos da base maior (TMDb) para se encontrar os dados matematicamente mais próximos em caligrafia, a fim de se encontrar seus "pares" na outra lista.

### Exploração dos dados

Para análise de variáveis mais importantes para realização da previsão, foram desenvolvidas visualizações que poderiam indicar relações entre a vitória de um filme no Oscar e seus outros fatores:

![Texto](analises/globodeouro.png)

Em primeira análise, nota-se que mais de 56% dos vencedores do Oscar também foram vencedores do Globo de Ouro, o que pode indicar esse como um fator determinante para se prever a variável resposta.

![Texto](analises/nom.png)

![Texto](analises/main_nom.png)

![Texto](analises/pop.png)

![Texto](analises/aprov.png)

![Texto](analises/runtime.png)

Para todas essas análises, o resultado é similar, mudando apenas em proporção em cada um deles: a média dos valores em filmes vencedores do Oscar (representados pela linha verde) são maiores em comparação a filmes não vencedores (linha vermelha). Isso é especialmente perceptível nas categorias de indicações (em que a média é 9 para vencedores e 6 para outros indicados) e popularidade, em que o número de avaliações médio é mais do que o dobro para filmes vencedores.

Por outro lado, apesar de ainda ter uma vantagem, a diferença média de notas é menor comparativamente, o que contradiz a intuição de que os vencedores do prêmio são sempre os filmes mais aclamados.

### Modelo

Como citado anteriormente, foi utilizado o modelo de Floresta Aleatória da biblioteca sklearn (RandomForestClassifier). Para o treinamento, foram separados os filmes da base pelos anos de cerimônia dos quais eles participaram, de forma que todos os filmes do mesmo ano ficassem juntos no treinamento e no teste. A separação foi feita de forma que 77 cerimônias foram selecionadas para o treino, e as outras 19 ficaram para testes (80/20); dessas 19 usadas no teste, foi determinado o escolhido como "vencedor previsto" o filme com maior probabilidade calculada dentre os indicados, independente de seu valor absoluto. Dessa forma, os resultados desse teste foram o seguinte:

![Matriz de confusão](analises/matriz_confusao.png)

Recall: **66.67%** (vencedor certo foi o vencedor previsto em 67% dos casos)

Top-2 Recall: **77.78%** (vencedor certo era um dos dois mais prováveis previstos em 78% dos casos)

### Limitações

Apesar de não ser um modelo perfeito, concluiu-se que um aprimoramento em seu desempenho apenas com dados disponíveis ao público seria demasiadamente complexo, considerando as diversas nuances subjetivas e informações confidenciais que acabam se passando durante uma votação de prêmios como o Oscar. Prova disso é a análise detalhada dos filmes nos quais o modelo não conseguiu prever corretamente o vencedor dentre os dois mais prováveis, sendo casos em que o vencedor da categoria naquele ano foi considerada uma grande surpresa mesmo entre especialistas de cinema (como por exemplo o ano de 2006, com o vencedor sendo o filme divisivo Crash - No Limite).

Além disso, as próprias variáveis utilizadas no treinamento do modelo podem apresentar pequenos problemas que atrapalhariam uma acurácia do modelo para previsões futuras: o indicador de popularidade, por exemplo, provavelmente é de certa forma viesado para essa análise, visto que a popularidade de um filme no IMDb tem tendência de aumentar bastante caso o filme seja o vencedor do Oscar, fazendo com que essa variável para premiações passadas não esteja tão precisa em relação às outras. Outro problema tem relação com o crescimento recente de plataformas de streaming (Netflix, Disney+, entre outros) e filmes exclusivo desses serviços, os quais fazem uma variável como a de bilheteria e lucro menos efetivas; caso houvesse uma maneira de se metrificar a popularidade dos filmes dentro de tais plataformas, esse problema poderia ser contornado, mas novamente esses dados geralmente não são públicos, o que dificulta tais análises para cerimônias recentes do Oscar.

## Previsão do Oscar 2025

Ao final da validação do modelo, este foi utilizado para previsão do Oscar 2025, visto que os indicados já haviam sido indicados ao momento da finalização do projeto, sem definição do vencedor. Preenchidas as informações, os resultados foram os seguintes:

| Posição | Filme               | Probabilidade |
|--------|---------------------|---------------|
| **1**  | **The Brutalist**    | **28.97%**    |
| **2**  | **Emilia Pérez**     | **15.93%**    |
| 3      | Ainda Estou Aqui     | 10.77%        |
| 4      | Dune: Part Two       | 9.59%         |
| 5      | Wicked               | 9.24%         |
| 6      | A Complete Unknown   | 7.92%         |
| 7      | Anora                | 6.68%         |
| 8      | Conclave             | 4.76%         |
| 9      | The Substance        | 3.75%         |
| 10     | Nickel Boys          | 2.40%         |

Utilizando como base os resultados obtidos, espera-se que o vencedor do Oscar 2025 fique entre os filmes "The Brutalist" ou "Emilia Pérez". Para elaboração dos motivos do favoritismo de tais filmes, desenvolveu-se um explicador LIME (Local Interpretable Model-Agnostic Explanations):

![Gráfico LIME descrevendo as variáveis principais para previsão de The Brutalist](analises/lime_brut.png)

![Gráfico LIME descrevendo as variáveis principais para previsão de Emilia Pérez](analises/lime_emillia.png)

![Gráfico LIME descrevendo as variáveis principais para previsão de Ainda Estou Aqui](analises/lime_aea.png)

Com base nessa avaliação, é possível se concluir o principal fator para se definir o favorito ao prêmio do oscar: a variável de vitória no globo de ouro é determinante para que Emilia Pérez esteja a frente do filme Ainda Estou Aqui, por exemplo. Depois dessa categoria principal, entram outros aspectos, como número de indicações totais e em categorias principais, lucro / orçamento dos filmes e duração. Comparando os dois favoritos, alguns dos fatores que colocaram The Brutalist como o favorito isolado são: sua nota média alta (acima de 7.85) e sua duração maior, além do fator de que Emilia Pérez gerou prejuízo para sua distribuidora nos cinemas.

## Contato
- [Meu LinkedIn](https://www.linkedin.com/in/carlos-neto-5668b0265/)
- Email: carloshmneto@usp.br
