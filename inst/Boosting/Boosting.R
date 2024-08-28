#-------------------------------------------------------
# Codigo para implementar Gradient Boosting Machines
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------
#- http://uc-r.github.io/gbm_regression

# Limpiar workspace
rm(list = ls())

#-> Importar librerias
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(AmesHousing)  # Base de datos para trabajar

#-> Se pretende predecir los precios de la vivienda en terminos de distintos features

#-> Creacion de muestras de entrenamiento (70%) y evaluacion (30%).
set.seed(123)
ames_split = initial_split(AmesHousing::make_ames(), prop = .7)
ames_train = training(ames_split)
ames_test  = testing(ames_split)

#-> Implementacion basica, usando gbm 
set.seed(123)

#-> Entrenar el modelo
gbm.fit = gbm(
  formula = Sale_Price ~ .,
  distribution = "gaussian",
  data = ames_train,
  n.trees = 10000,       # numero de arboles
  interaction.depth = 1, # numero de nodos
  shrinkage = 0.001,     # learning rate, step size, shrinkage, equivalentes.
  cv.folds = 5,          # se implementa por validacion cruzada.
  n.cores = NULL,        # usa todo el procesador
  verbose = FALSE
)  

#-> se imprimen los resultados
print(gbm.fit)

# gbm(formula = Sale_Price ~ ., distribution = "gaussian", data = ames_train, 
#    n.trees = 10000, interaction.depth = 1, shrinkage = 0.001, 
#    cv.folds = 5, verbose = FALSE, n.cores = NULL)
# A gradient boosted model with gaussian loss function.
# 10000 iterations were performed.
# The best cross-validation iteration was 10000.
# There were 80 predictors of which 45 had non-zero influence.

#-> Se calcula el RMSE
sqrt(min(gbm.fit$cv.error))

#-> Se grafica la funcion de perdida para los n arboles 
gbm.perf(gbm.fit, method = "cv")

#-> Dada la baja tasa de aprendizaje, el error cae muy poco ante cada nuevo arbol
#-> Esto implica que se deben agregar muchos arboles hasta minimizar el error


#-> Una alternativa es jugar con los hiperparametros del modelo. Por ejemplo, 
#-> a continuacion se estima un modelo con tres ramificaciones por nodo, en lugar
#-> de una.
set.seed(123)

#-> Se ajusta el modelo
gbm.fit2 = gbm(
  formula = Sale_Price ~ .,
  distribution = "gaussian",
  data = ames_train,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, 
  verbose = FALSE
)  

#-> Se halla cual es el modelo con error minimo
min_MSE <- which.min(gbm.fit2$cv.error)

#-> Se calcula el RMSE
sqrt(gbm.fit2$cv.error[min_MSE])

#-> Se grafica, de nuevo, la funcion de perdida
gbm.perf(gbm.fit2, method = "cv")


#-> Por supuesto, y como ya hemos visto, es posible plantear grillas y experimentar
#-> alrededor de todas las combinaciones de hiperparametros propuestas
#---> se crea la grilla de hiperparametros
hyper_grid = expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # espacio para guardar resultados
  min_RMSE = 0                     # espacio para guardar resultados
)

#---> numero total de combinaciones
nrow(hyper_grid)
## [1] 81

#---> Para este experimento no se usara el criterio de validacion cruzada, sino que
#---> Se va a construir muestras de entrenamiento y evaluacion. Esto con el fin de
#---> hacer el proceso mas rapido

#---> Se aleatoriza la informacion para construir las muestras
random_index <- sample(1:nrow(ames_train), nrow(ames_train))
random_ames_train <- ames_train[random_index, ]

#---> implementacion del experimento 
for(i in 1:nrow(hyper_grid)) {
  
  set.seed(123)
  
  #---> Se entrena el modelo
  gbm.tune = gbm(
    formula = Sale_Price ~ .,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, 
    verbose = FALSE
  )
  
  #---> Se adicionan dos criterior de informacion a la grilla: 
  #    min training error y arboles
  hyper_grid$optimal_trees[i] = which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] = sqrt(min(gbm.tune$valid.error))
}

#----> Con estos resultados se puede refinar la busqueda
hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

#shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees min_RMSE
#1       0.01                 5             10         0.80          2591 21335.08
#2       0.01                 5             10         0.65          2404 21446.85
#3       0.01                 5             10         1.00          3979 21588.13
#4       0.01                 5              5         1.00          4994 21620.97
#5       0.01                 5             15         0.80          2696 21671.79
#6       0.01                 3             10         0.65          4001 21702.82
#7       0.01                 3             15         1.00          3997 21713.88
#8       0.01                 3              5         1.00          5000 21717.38
#9       0.10                 3             15         1.00           426 21725.89
#10      0.10                 5              5         1.00           269 21731.89

#----> Se repite el procedimiento de crear una grilla y evaluar su bondad de ajuste
#---> se crea la grilla de hiperparametros
hyper_grid = expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # espacio para guardar resultados
  min_RMSE = 0                     # espacio para guardar resultados
)

#---> numero total de combinaciones
nrow(hyper_grid)

#---> implementacion del experimento 
for(i in 1:nrow(hyper_grid)) {
  
  set.seed(123)
  
  #---> Se entrena el modelo
  gbm.tune = gbm(
    formula = Sale_Price ~ .,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, 
    verbose = FALSE
  )
  
  #---> Se adicionan dos criterior de informacion a la grilla: 
  #    min training error y arboles
  hyper_grid$optimal_trees[i] = which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] = sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

#-> Una vez he refinado las grillas de busqueda, es posible tomar el siguiente 
#-> modelo, usando la combinacion de la primera fila del cuadro.

set.seed(123)

# train GBM model
gbm.fit.final = gbm(
  formula = Sale_Price ~ .,
  distribution = "gaussian",
  data = ames_train,
  n.trees = 424,
  interaction.depth = 7,
  shrinkage = 0.05,
  n.minobsinnode = 7,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, 
  verbose = FALSE
)  

#-> Una vez se tiene el modelo, se puede comenzar a extraer informacion de el.
#--> Importancia de cada variable
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # tambien se puede usar tecnicas de permutacion permutation.test.gbm
  las = 2
)

#--> Graficos de dependencia parcial: pretenden mostrar como cambia la variable
#--> dependiente a partir del aporte de cada una de las variables explicativas.
#--> Hay dos graficos que se pueden analizar: 
#-->    Partial Dependence Plots (PDP)
#-->    Individual Conditional Expectations (ICE)

#--> PDP muestra el comportamiento promedio de la variable dependiente predicha por el modelo
#--> cuando cada feature cambia a lo largo de su distribucion marginal
gbm.fit.final %>%
  partial(pred.var = "Gr_Liv_Area", n.trees = gbm.fit.final$n.trees, grid.resolution = 100) %>%
  autoplot(rug = TRUE, train = ames_train) +
  scale_y_continuous(labels = scales::dollar)

#--> Este grafico analiza el cambio promedio en la variable dependiente cuando se
#--> cambian los valores de la variable "Gr_Liv_Area" (tamaño de la vivienda en pies cuadrados), 
#--> manteniendo todo lo demas constante.

#--> ICE son extensiones de PDP, pero en lugar de graficar el efecto promedio en la
#--> variable respuesta, se grafica el cambio en la variable dependiente para cada obser-
#--> vación.
ice1 = gbm.fit.final %>%
  partial(
    pred.var = "Gr_Liv_Area", 
    n.trees = gbm.fit.final$n.trees, 
    grid.resolution = 100,
    ice = TRUE
  ) %>%
  autoplot(rug = TRUE, train = ames_train, alpha = .1) +
  ggtitle("No centradas") +
  scale_y_continuous(labels = scales::dollar)

ice2 = gbm.fit.final %>%
  partial(
    pred.var = "Gr_Liv_Area", 
    n.trees = gbm.fit.final$n.trees, 
    grid.resolution = 100,
    ice = TRUE
  ) %>%
  autoplot(rug = TRUE, train = ames_train, alpha = .1, center = TRUE) +
  ggtitle("Centradas") + # los puntos se corte de la y se normalizan a cero
  scale_y_continuous(labels = scales::dollar)

gridExtra::grid.arrange(ice1, ice2, nrow = 1)

#--> Utilidad de grafico: ver si las predicciones individuales siguen de cerca
#--> la tendencia comun (linea roja) o si las predicciones son demasiados dispersas
#--> favoreciendo la presencia de valores atipicos.

#--> Prediccion - se hace con respecto a la muestra de evaluacion
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, ames_test)

#--> Grafico de comparacion
plot(pred, type = "l", lty=1)
lines(ames_test$Sale_Price, type = "l", lty=1, col=3)

#--> RMSE del pronostico
caret::RMSE(pred, ames_test$Sale_Price)
## [1] 21185.75

