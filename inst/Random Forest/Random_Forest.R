#-------------------------------------------------------
# Codigo para implementar Random Forest
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------
#- Fuente http://uc-r.github.io/random_forests

# Limpiar workspace
rm(list = ls())

#-> Importar librerias requeridas
library(rsample)      # Dividir muestras 
library(randomForest) # Implementacion basica de la metodologia
library(ranger)       # Una implementacion mas rapida de randomForest
library(caret)        # Un paquete que permite implementar varias tecnicas de ML
library(AmesHousing)  # Base de datos para trabajar
library(ggplot2)
library(tidyr)
library(dplyr) 


#-> Se utiliza la base de datos que caracteriza las viviendas en Ames, Iowa, EEUU.
#-> Fuente: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

#-> Se pretende predecir los precios de la vivienda en terminos de distintos features

#-> Creacion de muestras de entrenamiento (70%) y evaluacion (30%).
set.seed(123)
ames_split = initial_split(AmesHousing::make_ames(), prop = .7)
ames_train = training(ames_split)
ames_test  = testing(ames_split)

#-> Implementacion del modelo estandar
set.seed(123)
m1 <- randomForest(
  formula = Sale_Price ~ .,
  data    = ames_train,
  ntree   = 600
)

m1

##-> m1 muestra el resultado: 
## Call:
##  randomForest(formula = Sale_Price ~ ., data = ames_train) 
##                Type of random forest: regression
##                      Number of trees: 500 (Numero de regression trees en el bosque)
## No. of variables tried at each split: 26 (26 de las 79 variables disponibles)
## 
##           Mean of squared residuals: 659550782
##                     % Var explained: 89.83

#-> Graficar el comportamiento del error a medida que se incorporan arboles
#-> Se usa OOB error como el metodo de calculo del error
plot(m1)

#-> Se determina el numero optimo de arboles a entrar en la regresion
which.min(m1$mse)

#-> RMSE del random forest con el numero optimo de arboles
sqrt(m1$mse[which.min(m1$mse)])

#-> Si se desea utilizar validacion cruzada para hallar el mejor modelo, se debe hacer:
#-> Se crea las muestras de estimacion y de evaluacion 
set.seed(123)
valid_split = initial_split(ames_train, .7)

#-> Muestra de entrenamiento
ames_train_v2 = analysis(valid_split)

#-> Muestra de evaluacion
ames_valid = assessment(valid_split)
x_test = ames_valid[setdiff(names(ames_valid), "Sale_Price")]
y_test = ames_valid$Sale_Price

#-> Se estima el modelo con las muestras propuestas arriba
set.seed(123)
rf_oob_comp = randomForest(
  formula = Sale_Price ~ .,
  data    = ames_train_v2,
  ntree   = 800,
  xtest   = x_test,
  ytest   = y_test
)

#-> Extraer errores OOB y de pronostico con la muestra de evaluacion
oob = sqrt(rf_oob_comp$mse)
validation = sqrt(rf_oob_comp$test$mse)

#-> Comparar las tasas de error
tibble::tibble(
  `Out of Bag Error` = oob,
  `Error validación` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = scales::dollar) +
  xlab("Número de árboles")

#-> Hasta aqui, podriamos usar este modelo para pronosticar. 
#-> Implementacion del modelo estandar
set.seed(123)
rf_md = randomForest(
  formula = Sale_Price ~ .,
  data    = ames_train,
  ntree   = 501
)

#-> Sin embargo, se puede calibrar mas, con el fin de refinar el modelo y sus predicciones
#--> Se plantea la busqueda de la combinacion de los hiperparametros, para encontrar
#--> una configuracion optima
#--> Se plantea una grilla de valores para los hiperparametros relecantes:
#--> 1. Numero de variables que se evaluan aleatoriamente en cada bifurcacion
#--> 2. Cantidad minima de observaciones en los nodos terminales.
#--> 3. El numero de muestras para realizar el entrenamiento.
#--> Por supuesto, se pueden hacer combinaciones diferentes o solo evaluar de a 
#--> un hiperparametro al tiempo, usando el mismo comando.

hyper_grid = expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sample_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

#--> Numero total de cominaciones (modelos a evaluar)
nrow(hyper_grid)
## [1] 96

#--> Se implementan todos los modelos con las combinaciones propuestas
#--> Notese que aqui se usa libreria ranger, hace Random Forest mucho mas rapido
for(i in 1:nrow(hyper_grid)) {
  
  # Modelo de entrenamiento
  model = ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 590,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sample_size[i],
    seed            = 123
  )
  
  #--> Agregar OOB error a la grilla
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)
#mtry node_size sample_size OOB_RMSE
#1    26         3       0.800 25562.07
#2    26         5       0.800 25642.20
#3    30         3       0.800 25643.97
#4    30         5       0.800 25692.39
#5    26         7       0.800 25703.74
#6    30         7       0.800 25714.03
#7    22         3       0.800 25750.93
#8    26         9       0.800 25769.61
#9    26         3       0.632 25782.45
#10   24         3       0.800 25808.47

#-> Se ha dicho que Random Forest es capaz de lidiar con variables categoricas
#-> sin tener que construir variables indicadoras asociadas. Averiguemos si
#-> aplica a este caso.
#--> Para las variables categoricas, crea indicadoras (dummies)
one_hot = dummyVars(~ ., ames_train, fullRank = FALSE)
ames_train_hot = predict(one_hot, ames_train) %>% as.data.frame()

#--> Se compatibilizan los nombres de las variables transformadas, para
#--> que la herramienta sepa cual variables se esta incluyendo
names(ames_train_hot) = make.names(names(ames_train_hot), allow_ = FALSE)

#--> Dado que una variable categorica ahora es determinada por c-1 variables 
#--> dicotomas (c es el numero de categorias de cada variable), ahora se tienen
#--> 353 variables predictoras (noten como puede incrementarse facilmente el espacio de
#--> features)
hyper_grid_2 <- expand.grid(
  mtry       = seq(50, 200, by = 25),
  node_size  = seq(3, 9, by = 2),
  sample_size = c(.55, .632, .70, .80),
  OOB_RMSE  = 0
)

# perform grid search
for(i in 1:nrow(hyper_grid_2)) {
  
  # train model
  model <- ranger(
    formula         = Sale.Price ~ ., 
    data            = ames_train_hot, 
    num.trees       = 590,
    mtry            = hyper_grid_2$mtry[i],
    min.node.size   = hyper_grid_2$node_size[i],
    sample.fraction = hyper_grid_2$sample_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid_2$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid_2 %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

#mtry node_size sample_size OOB_RMSE
#1    75         3       0.800 26554.95
#2    75         5       0.800 26629.56
#3    75         7       0.800 26696.37
#4    75         3       0.632 26714.03
#5   175         3       0.800 26722.37
#6   150         7       0.800 26732.72
#7   150         3       0.800 26734.86
#8   175         5       0.800 26735.84
#9    75         3       0.700 26741.84
#10  175         7       0.800 26747.16


#--> Noten que usar variables dummies en lugar de las categoricas originales NO
#--> mejora la estimacion (comparen OOB_RMSE para la primera fila de cada cuadro).

#--> Para los parametros con la menor tasa de error, es posible crear un histograma
#--> para identificar mejor cual puede ser el valor mas probable en la tasa de error

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger = ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 590,
    mtry            = 26,
    min.node.size   = 3,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

#--> Ademas, la opcion importance = 'impurity' permite implementar el procedimiento
#--> de calculo de la importancia de cada variable
#--> Veamos el resultado graficamente.
ggplot(
  enframe(
    optimal_ranger$variable.importance,
    name = "variable",
    value = "importance"
  ),
  aes(
    x = reorder(variable, importance),
    y = importance,
    fill = importance
  )
) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  ylab("Variable Importance") +
  xlab("") +
  ggtitle("Information Value Summary") +
  guides(fill = "none") +
  scale_fill_gradient(low = "red", high = "blue")

#-> Pronostico
#--> Se puede ver y comparar los pronosticos hechos con cada modelo y paquete
#---> Con randomForest
pred_rf = predict(rf_md, ames_test)

#--> Con ranger
pred_ranger = predict(optimal_ranger, ames_test)

#--> Grafico de comparacion
plot(pred_rf, type = "l", lty=1)
lines(pred_ranger$predictions, type = "l", lty=1, col=4)
lines(ames_test$Sale_Price, type = "l", lty=1, col=3)
