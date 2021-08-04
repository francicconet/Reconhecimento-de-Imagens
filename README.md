## Reconhecimento de Imagens

Este algoritmo treina um modelo de rede neural para classificação de imagens de gestos de positivo (:thumbsup:) ou negativo (:thumbsdown:). 

Para isso utiliza-se o [tf.keras](https://keras.io/guides/), uma API de alto nível para treinar modelos no TensorFlow.

## Pré-Requisitos

Para a compilação do código foram utilizadas as seguintes ferramentas:

Python 3.7

Keras 2.3.1

Numpy 1.18.1

Tensorflow 1.14.0

## Parâmetros 

Para a realização dos testes de classificação, os seguintes parâmetros devem ser ajustados:

```
batchSize = 8
learningRate = 0.0001
nIterations = 100
```
O `batchSize` determina o número de imagens que o otimizador irá utilizar em cada passo do treinamento. 
`learningRate` determina o tamanho do passo na direção do mínimo local durante o treinamento e `nIterations` é o número de iterações. 

O tamanho das imagens também é decisiva para a resposta do classificador:

```
imSize = 32
nChannels = 1   
nClasses = 2
nImagesPerClass = [30,30] 
```
O tamanho da imagem é definido pelo parâmetro `imSize` que representa 32x32 pixels. A imagem foi pré processada para tons de cinza, definindo o número de canais `nChannels = 1` (caso fosse mantido o formato RGB: `nChannels = 3` 'red', 'green' e 'blue'). As classes são duas (:thumbsup: ou :thumbsdown:), definidas por `nClasses = 2`. Para o treinamento foi utilizado um banco de dados de 30 imagens de cada classe, definidos em `nImagesPerClass = [30,30]`.

## Modelo

Considerações do modelo:

* Foi utilizado um modelo de rede neural de convolução (CNN) por ser o mais apropriado para processamento de imagens.
* Por ter poucas amostras de treinamento reduziu-se o tamanho das imagens ao máximo e para simplificar a tarefa foram utilizadas escalas de cinza, ao invés de RGB.

## Treinamento e Teste

O treinamento da rede neural segue os seguintes passos:

1. Alimentar o modelo com os dados de treinamento que estão no diretório `im_data/train`.
2. O modelo aprende como associar as imagens às *labels*.
3. O modelo faz previsões sobre o conjunto de testes (diretório `im_data/test`), verificando se as previsões combinam com as *labels* dos testes.

À medida que o modelo treina, as métricas *loss* e *accuracy* são mostradas. O modelo atinge uma acurácia na previsão das imagens de `test acc: 0.850000` (ou 85%). 

## Resultados

No terminal é mostrado o resultado dos testes para 20 imagens (sendo as dez primeiras *thumbsdown* e as dez últimas *thumbsup*). No arquivo *TERMINAL.txt* é possivel verificar as informações de um teste. Ao final, a classificação feita pelo algoritmo é demonstrada:

```
test acc: 0.850000
Imagem | Classificação
0 | thumbsdown
1 | thumbsdown
2 | thumbsdown
3 | thumbsup
4 | thumbsdown
5 | thumbsdown
6 | thumbsdown
7 | thumbsdown
8 | thumbsdown
9 | thumbsdown
10 | thumbsup
11 | thumbsup
12 | thumbsup
13 | thumbsup
14 | thumbsup
15 | thumbsdown
16 | thumbsup
17 | thumbsup
18 | thumbsdown
19 | thumbsup
```

Verifica-se que em alguns casos o modelo não definiu corretamente o tipo de gesto da imagem, mas manteve uma média de acurácia (ou acertos) de 75% das análises.  

## Melhorias

* Incluir mais imagens para o treinamento do algoritmo.
