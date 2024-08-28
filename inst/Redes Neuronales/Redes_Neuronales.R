#-------------------------------------------------------
# Codigo para implementar Redes Neuronales a un problema
# de regresion.
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------

# Limpiar workspace
rm(list = ls())

#-> Importar las librerias
library("neuralnet")
library(MASS)

#-> Para asegurar replicabilidad de los resultados
set.seed(123)

#-> Importar la base de datos
data = Boston
str(data) # obtener una vista de los datos

#-> Ejercicio: modelar los precios de las viviendas en Boston (miles de dolares)
#-> en terminos de 13 variables, que determinan los atributos del mercado (ver
#-> bloc de notas complementario)

#-> medv es la variable dependiente.

#-> En redes neuronales, se deben normalizar los datos, para mejorar ajuste
#--> datos sin unidades, en la misma escala.
max_data = apply(data, 2, max) 
min_data = apply(data, 2, min)

#--> x_{scaled} = \frac{x-x_{min}}{x_{max}-x_{min}}
data_scaled = scale(data,center = min_data, scale = max_data - min_data) 

#--> Se toma la muestra completa y se rompe en muestra de entrenamiento (70%) y
#--> de evaluacion (30%).
index = sample(1:nrow(data),round(0.70*nrow(data)))
train_data = as.data.frame(data_scaled[index,])
test_data = as.data.frame(data_scaled[-index,])

#--> Se implementa el procedimiento de estimacion de la ANN
#----> Se extraen los nombres de las variables
n = names(data)

#----> Se crear recursivamente la formula de la ANN a estimar dep ~ todas las exp
f = as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

#----> Se estima la red neuronal con 10 nodos ocultos (hidden=10) y resultado lineal
net_data = neuralnet(f,data=train_data,hidden=10,linear.output=T)

#----> Se grafica la red resultante
plot(net_data)

#----> Con esta red es posible hacer prediccion directamente
predict_net_test = compute(net_data,test_data[,1:13])

#----> Se calcula el RMSE para este pronostico
predict_net_test_start = predict_net_test$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test_start = as.data.frame((test_data$medv)*(max(data$medv)-min(data$medv))+min(data$medv))
MSE.net_data = sum((test_start - predict_net_test_start)^2)/nrow(test_start)

#----> Para evaluar la capacidad predictiva de esta ANN, se contrasta contra una
#----> regresion lineal.
#----> Se estima el modelo
Regression_Model = lm(medv~., data=data)
summary(Regression_Model)
#----> Se extrae la muestra de evaluacion
test = data[-index,]
#----> Se predice sobre la muestra de evaluacion
predict_lm = predict(Regression_Model,test)
#----> Se calcula el RMSE
MSE.lm = sum((predict_lm - test$medv)^2)/nrow(test)

#----> Se contrastan los dos RMSE
MSE.net_data
MSE.lm

#--> Notese que la red neural tiene un menor RMSE que la regresion lineal multiple.


