#-------------------------------------------------------
# Codigo para implementar Regression Trees
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------

#- http://uc-r.github.io/regression_trees

# Limpiar workspace
rm(list = ls())

#-> Importar librerias requeridas
library(rsample)      # Dividir muestras 
library(rpart)        # Implementacion basica de la metodologia
library(rpart.plot)   # Graficos de regression trees
library(caret)        # Un paquete que permite implementar varias tecnicas de ML
library(ipred)        # Bagging
library(AmesHousing)  # Base de datos para trabajar
library(ggplot2)
library(tidyr)
library(dplyr) 

#-> Se pretende predecir los precios de la vivienda en terminos de distintos features

#-> Creacion de muestras de entrenamiento (70%) y evaluacion (30%).
set.seed(123)
ames_split = initial_split(AmesHousing::make_ames(), prop = .7)
ames_train = training(ames_split)
ames_test  = testing(ames_split)

#-> Implementacion basica
#-> Para hacer regression trees es necesario usar "anova" en la opcion "method"
m1 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova"
)

m1

#--> Resultados del primer regression trees
#n= 2051 

#node), split, n, deviance, yval
#* denotes terminal node

# CADA RAMA TIENE: (#obs, SSE, Precio promedio de vivienda)
#PRIMER NODO  1) root 2051 1.319672e+13 178987.30  
#PRIMERA RAMA  2) Overall_Qual=Very_Poor,Poor,Fair,Below_Average,Average,Above_Average,Good 1725 4.089851e+12 155077.10  
#  4) Neighborhood=North_Ames,Old_Town,Edwards,Sawyer,Mitchell,Brookside,Iowa_DOT_and_Rail_Road,South_and_West_of_Iowa_State_University,Meadow_Village,Briardale,Northpark_Villa,Blueste,Landmark 1039 1.414812e+12 131469.60  
#  8) Overall_Qual=Very_Poor,Poor,Fair,Below_Average 202 1.682257e+11  96476.75 *
#  9) Overall_Qual=Average,Above_Average,Good 837 9.395439e+11 139914.60  
#  18) First_Flr_SF< 1240 647 4.048057e+11 131629.20 *
#  19) First_Flr_SF>=1240 190 3.390731e+11 168128.90 *
#  5) Neighborhood=College_Creek,Somerset,Northridge_Heights,Gilbert,Northwest_Ames,Sawyer_West,Crawford,Timberland,Northridge,Stone_Brook,Clear_Creek,Bloomington_Heights,Veenker,Green_Hills 686 1.218974e+12 190832.40  
#  10) Gr_Liv_Area< 1725 497 5.267647e+11 176577.80  
#  20) Total_Bsmt_SF< 1217.5 317 2.021716e+11 163165.00 *
#  21) Total_Bsmt_SF>=1217.5 180 1.671280e+11 200199.30 *
#  11) Gr_Liv_Area>=1725 189 3.256624e+11 228316.80 *
#SEGUNDA RAMA  3) Overall_Qual=Very_Good,Excellent,Very_Excellent 326 2.902384e+12 305506.30  
#  6) Overall_Qual=Very_Good 235 9.229383e+11 270486.30  
#  12) Gr_Liv_Area< 1956.5 146 3.185953e+11 241512.80 *
#  13) Gr_Liv_Area>=1956.5 89 2.807253e+11 318015.80 *
#  7) Overall_Qual=Excellent,Very_Excellent 91 9.469774e+11 395942.60  
#  14) Gr_Liv_Area< 1958 38 6.559161e+10 337195.90 *
#  15) Gr_Liv_Area>=1958 53 6.562129e+11 438062.80  
#  30) Neighborhood=Edwards,Somerset,Veenker 7 9.951422e+10 290836.40 *
#  31) Neighborhood=College_Creek,Old_Town,Northridge_Heights,Timberland,Northridge,Stone_Brook 46 3.818802e+11 460466.80  
#  62) Total_Bsmt_SF< 2394 39 1.996090e+11 437638.20 *
#  63) Total_Bsmt_SF>=2394 7 4.870803e+10 587655.30 *

#-> Es posible ver esta misma informacion, pero visualmente
rpart.plot(m1)

#-> Noten que inicialmente se cuentan con 80 variables para trabajar, pero el arbol
#-> tiene solamente 11 nodos internos, para 12 nodos terminales.
#-> Esto es el resultado de PODAR el arbol.

#-> Si se desea ver como se disminuye el error a medida que el arbol va creciendo
#-> se puede usar la siguiente instruccion
plotcp(m1)

#-> Noten como el error no mejora ante cada nuevo nodo.
#-> Si se quiere evaluar la totalidad del arbol, sin podar, y ver como se comporta
#-> este indicador, se debe hacer lo siguiente:
#-> CLAVE: cp=0 desactiva el podado del arbol.
m2 = rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(cp = 0, xval = 10)
)

plotcp(m2)
abline(v = 12, lty = "dashed")
m1$cptable

#-> Igual, se puede tratar de refinar el arbol, mediante la calibracion de los
#-> hiperparametros del modelo: minsplit es el numero minimo de datos en cada nodo
#-> maxdepth es el numero maximo de nodos internos.
#-> Por ejemplo, se va a crecer un arbol con 10 obs por nodo y maximo 12 nodos internos
m3 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(minsplit = 10, maxdepth = 12, xval = 10)
)

m3$cptable


#-> Por supuesto, este procedimiento puede hacerse automaticamente, sin tener que
#-> repetir el codigo para cada combinacion de hiperparametros.
hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)

head(hyper_grid)

#minsplit maxdepth
#1        5        8
#2        6        8
#3        7        8
#4        8        8
#5        9        8
#6       10        8

#--> numero total de combinaciones
nrow(hyper_grid)
# [1] 128

#-> Se aplican los experimentos
models = list()

for (i in 1:nrow(hyper_grid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit = hyper_grid$minsplit[i]
  maxdepth = hyper_grid$maxdepth[i]
  
  # train a model and store in the list
  models[[i]] = rpart(
    formula = Sale_Price ~ .,
    data    = ames_train,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

#-> Con esta funcion se puede hallar la combinacion optima de hiperparametros
#-> de todos los experimentos hechos.
#--> Se halla el minimo numero de nodos
get_cp = function(x) {
  min    = which.min(x$cptable[, "xerror"])
  cp = x$cptable[min, "CP"] 
}

#--> Se halla el minimo error
get_min_error = function(x) {
  min    = which.min(x$cptable[, "xerror"])
  xerror = x$cptable[min, "xerror"] 
}

hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

#-> Si estos resultados nos convencen, podemos usar este arbol para predecir
#minsplit maxdepth         cp     error
#1        6       10 0.01000000 0.2331321
#2        7       12 0.01012094 0.2335002
#3       15       14 0.01012094 0.2352524
#4        6        9 0.01000000 0.2356500
#5       17       14 0.01000000 0.2357567

optimal_tree = rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova",
  control = list(minsplit = 6, maxdepth = 10, cp = 0.01)
)

#-> Hagamos prediccion con este modelo
pred = predict(optimal_tree, newdata = ames_test)

#-> Calculo del RMSE
RMSE_rpart = RMSE(pred = pred, obs = ames_test$Sale_Price)

#-> Grafico observado vs pronosticado
#--> Grafico de comparacion
plot(pred, type = "l", lty=1)
lines(ames_test$Sale_Price, type = "l", lty=1, col=3)

#-------------------------------------------------------------------------------
# Bagging
# Recuerden que este procedimiento esta entre Regression Tree y Random Forest
#-------------------------------------------------------------------------------
#-> Semilla para hacer reproducible el ejercicio
set.seed(123)

# Entrenar el modelo
bagged_m1 = bagging(
  formula = Sale_Price ~ .,
  data    = ames_train,
  coob    = TRUE
)

bagged_m1

#Bagging regression trees with 25 bootstrap replications 

#Call: bagging.data.frame(formula = Sale_Price ~ ., data = ames_train, 
#                         coob = TRUE)

#Out-of-bag estimate of root mean squared error:  35884.95 

#-> En este acercamiento, mientras mas arboles se tenga, mejor. Sin embargo, se puede
#-> hallar un valor optimo de estos
#--> Evaluaremos una senda de entre 10 y 50 arboles
ntree = 10:50

#--> espacio para guardar los valores de OOB RMSE
rmse = vector(mode = "numeric", length = length(ntree))

#--> Implementando el experimento
for (i in seq_along(ntree)) {
  
  set.seed(123)
  
  #---> Implementar el modelo
  model = bagging(
    formula = Sale_Price ~ .,
    data    = ames_train,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  #---> obtener el OOB error
  rmse[i] = model$err
}

#--> Graficamos el resultado
plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 25, col = "red", lty = "dashed")

#- 25 arboles parece ser un numero adecuado.

#-> el paquete caret permite hacer lo mismo, mas analizar graficamente las variables
#-> mas importantes, mas validacion cruzada

#-> Se especifica el espacio para hacer validacion cruzada
#-> 10-fold significa partir la muestra en 10 partes, 9 de entrenamiento, una de
#-> evaluacion y hacer 10 veces cada iteracion y guardar los residuales
ctrl = trainControl(method = "cv",  number = 10) 

#-> implementar el modelo
bagged_cv = train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)

#-> Revisar los resultados 
bagged_cv
## Bagged CART 
## 
## 2051 samples
##   80 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 1847, 1845, 1846, 1845, 1846, 1847, ... 
## Resampling results:
## 
##   RMSE      Rsquared   MAE     
##   36083.63  0.8061187  24109.43

#-> Podemos graficar cuales son las variables mas importantes
plot(varImp(bagged_cv), 20)  

#-> Podemos implementar prediccion y compararlo con el metodo anterior
pred_cv = predict(bagged_cv, ames_test)

#--> Grafico de comparacion
plot(pred, type = "l", lty=1)
lines(pred_cv, type = "l", lty=1, col=4)
lines(ames_test$Sale_Price, type = "l", lty=1, col=3)


#--> RMSE para este arbol
RMSE_cv = RMSE(pred_cv, ames_test$Sale_Price)

RMSE_rpart
RMSE_cv
