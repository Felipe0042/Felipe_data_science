#DEFININDO DIRETÓRIO DE TRABALHO
getwd()

install.packages('ISLR')  #dataset
install.packages('caret') #pacote machine learning
install.packages('e1071') #outras funcoes


library(ISLR)
library(caret)
library(e1071)

summary(Smarket)
str(Smarket)
View(Smarket)


#particionar dados
createDataPartition()

index_train <- createDataPartition(Smarket$Direction, p=0.75, list = F)
dados_treino <- Smarket[index_train, ]
dados_teste <- Smarket[-index_train, ]


#distribuicao dos dados
prop.table(table(dados_treino$Direction))*100
prop.table(table(Smarket$Direction))*100

#verificando as correlações
descrCor <- cor(dados_treino[, names(dados_treino) !=  'Direction'])
names(dados_treino)


scale(dados_treino['Volume'])
numeric.variaveis <- names(dados_treino[colnames(dados_treino) != 'Direction'])


#center -> media do atributo
#scale  -> desvio padrao
#calcula a padronização dos dados 

scale.features <- function(df, variaveis){
  for (variavel in variaveis){
    df[[variavel]] <- scale(df[[variavel]], center = T, scale = T)
  }
  return(df)
}

dados_treino_scaled <- scale.features(dados_treino, numeric.variaveis)
dados_teste_scaled  <- scale.features(dados_teste, numeric.variaveis)



#Aplicando o KNN
set.seed(400)

#arquivo de controle
ctrl <- trainControl(method = 'repeatedcv', repeats = 3)

#Criando o modelo
knn_v1 <- train(Direction ~. ,
                data = dados_treino_scaled,
                method = 'knn',
                trControl = ctrl,
                #preProcess = c('center', 'scale') #padronização dos dados
                tuneLength = 20)

knn_v1

#numero de vizinho x Acurácia
plot(knn_v1)


#Fazendo previsoes
knnPredict <- predict(knn_v1, newdata = dados_teste_scaled)
knnPredict


#matriz de confusão
confusionMatrix(knnPredict, dados_teste$Direction)


####### Aplicando outras métricas ROC ###########


ctrl <- trainControl(method = 'repeatedcv',
                     repeats = 3,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

#Treinando modelo
knn_v2 <- train(Direction~.,
                data = dados_treino_scaled,
                method = 'knn',
                trControl = ctrl,
                metric = 'ROC',
                tuneLength = 20)

knn_v2

plot(knn_v2, print.thres = 0.5, type = 'S')

#fazendo previsao 
knnPredict <- predict(knn_v2, newdata = dados_teste_scaled)
confusionMatrix(knnPredict, dados_teste$Direction)


#Prevendo valores
Year <- c(2006,2007,2008)
Lag1 <- c(1.30,0.09,0.114)
Lag2 <- c(1.30,0.29,0.114)
Lag3 <- c(-0.421,-1.9,0.214)
Lag4 <- c(0.548,0.899,-0.254)
Lag5 <- c(0.214,0.105,-0.584)
Volume <- c(1.368,1.0959,1.23124)
Today <- c(0.289,-0.487, 1.649)

novos_dados <- data.frame(Year, Lag1, Lag2, Lag3, Lag4, Lag5, Volume, Today)
View(novos_dados)

#Extraindo os nomes das columas
nomes_colunas <- colnames(novos_dados)
nomes_colunas


#padronizando os dados
novos_dados_scaled <- scale.features(novos_dados, nomes_colunas)
novos_dados_scaled


knnPredict <- predict(knn_v2, newdata = novos_dados_scaled)
cat(sprintf("\n Previsao de \"%s\" é \"%s\"\n", novos_dados$Year, knnPredict))


