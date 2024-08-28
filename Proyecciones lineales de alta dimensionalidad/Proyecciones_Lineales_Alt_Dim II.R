#-------------------------------------------------------
# Codigo para implementar RIDGE, LASSO y ELASTIC NET
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------
# Fuente para las regresiones Ridge, LASSO, Elastic Net
# https://www.datacamp.com/tutorial/tutorial-ridge-lasso-elastic-net

# Limpiar workspace
rm(list = ls())

# Cargar librerias necesarias para estimacion
set.seed(123)
library(glmnet)  # clave para implementar Ridge
library(dplyr)
library(psych)
library(data.table)
library(ggplot2)
library(ggfortify)
library(caret)


#-> Se alistan las variables explicativas
# setwd("D:/Dropbox/BR/Depto_Macro/Cursos/Banco Central de Bolivia/Códigos/Proyecciones lineales de alta dimensionalidad")
gdp.hf <- fread('Proyecciones lineales de alta dimensionalidad/us_gdp_monthly_ch12_sec6.csv',
                select = c('date', 'coinc_indicators_index', 'employees', 'ind_prod', 'pers_inc', 'sales'))
gdp.hf[, date := as.Date(date, format = '%m/%d/%Y')]
gdp.hf[, qtr := rep(1:112, each = 3)]
gdp.hf[, c('CI', 'E', 'IP', 'INC', 'SA') :=
         list(mean(coinc_indicators_index), mean(employees), mean(ind_prod), mean(pers_inc), mean(sales)), by = qtr]

gdp.xs = unique(gdp.hf[date <= '2012-12-01' & date >= '1985-04-01', .(date, CI, E, IP, INC, SA)])

#-> Se alista la variable explicada
gdp.lf = fread('Proyecciones lineales de alta dimensionalidad/usgdp_qtr_ch13_sec6.csv')
gdp.lf[, date := as.Date(date, format = '%m/%d/%Y')]

gdp = unique(gdp.lf[date <= '2012-10-01', .(date, gdp)])

#-> Base de datos a trabajar
gdp_us = merge(gdp, gdp.xs, by="date")

#-> Descripcion base de datos:
#--> Explicada: PIB de EEUU (gdp).
#--> Explicativas:
#--> indice de indicadores coincidentes (CI)
#--> Empleados de la economia, menos agricultura (E)
#--> indice de produccion industrial (IP)
#--> Ingreso personal (INC)
#--> Ventas de la industria y el comercio (SA)

#-> Se estandariza la variable explicada
y = gdp_us %>% select(gdp) %>% scale(center = TRUE, scale = FALSE) %>% as.matrix()

#-> Se transforman los datos para dejarlos como objetos matriciales
#-> En glmnet, las xs se estandarizan automaticamente
X = gdp_us  %>% select(-c(gdp, date)) %>% as.matrix()

#--> Se genera una secuencia de valores de \lambda para evaluar el que minimiza RMSE
lambdas_to_try = 10^seq(-2, 5, length.out = 100)

#--> Clave: alpha=0 implementa ridge en este procedimiento
ridge_cv = cv.glmnet(X, y, alpha = 0, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 10)

#--> Graficar los resultados de la validacion cruzada
plot(ridge_cv)

#--> Se escoge el \lambda que minimiza RMSE - mejor pronostico (validacion cruzada)
lambda_cv = ridge_cv$lambda.min

#-> Se estima el mejor modelo, se estrae RRS y R cuadrado
model_cv = glmnet(X, y, alpha = 0, lambda = lambda_cv, standardize = TRUE)
model_cv$beta

#--> Pronostico por dentro de muestra
y_hat_cv = predict(model_cv, X)
plot(y, type = "l", lty=1)
lines(y_hat_cv, type = "l", lty=1, col=4)

#--> Se computa suma de residuos al cuadrado
ssr_cv = t(y - y_hat_cv) %*% (y - y_hat_cv)

#--> Se calcula el R cuadrado
rsq_ridge_cv = cor(y, y_hat_cv)^2
rsq_ridge_cv

#-> Se puede escoger \lambda con criterios de informacion, también.
#--> Se estandarizan las variables x
X_scaled = scale(X)
aic = c()
bic = c()
for (lambda in seq(lambdas_to_try)) {
  # Correr modelo
  model = glmnet(X, y, alpha = 0, lambda = lambdas_to_try[lambda], standardize = TRUE)
  # Se extraen los coeficientes y residuales (se quita la primera fila, corresponde a constante)
  betas = as.vector((as.matrix(coef(model))[-1, ]))
  resid = y - (X_scaled %*% betas)
  # Se computa la matriz H y los grados de libertad
  ld = lambdas_to_try[lambda] * diag(ncol(X_scaled))
  H = X_scaled %*% solve(t(X_scaled) %*% X_scaled + ld) %*% t(X_scaled)
  df = tr(H)
  # Se computan los criterios de informacion
  aic[lambda] = nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df
  bic[lambda] = nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df * log(nrow(X_scaled))
}

# Grafico de los criterios de informacion vs cada lambda estimado
plot(log(lambdas_to_try), aic, col = "orange", type = "l",
      ylab = "Information Criterion")
lines(log(lambdas_to_try), bic, col = "skyblue3")
legend("bottomleft", lwd = 1, col = c("orange", "skyblue3"), legend = c("AIC", "BIC"))

#--> Se hallan los \lambdas optimos, segun ambos criterios
lambda_aic = lambdas_to_try[which.min(aic)]
lambda_bic = lambdas_to_try[which.min(bic)]

#--> Se ajusta el modelo final, se calcula RSS y R cuadrado
model_aic = glmnet(X, y, alpha = 0, lambda = lambda_aic, standardize = TRUE)
y_hat_aic = predict(model_aic, X)
ssr_aic = t(y - y_hat_aic) %*% (y - y_hat_aic)
rsq_ridge_aic = cor(y, y_hat_aic)^2

model_bic = glmnet(X, y, alpha = 0, lambda = lambda_bic, standardize = TRUE)
y_hat_bic = predict(model_bic, X)
ssr_bic = t(y - y_hat_bic) %*% (y - y_hat_bic)
rsq_ridge_bic = cor(y, y_hat_bic)^2


#--> Notese como incrementar el lambda genera coeficientes cercanos a cero
#--> Cada linea muestra coeficientes para cada variable, para diferentes \lambdas.
#--> Cuanto mayor el lambda, mas coeficientes se vuelven cero.
res = glmnet(X, y, alpha = 0, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(X), cex = .7)


#-------------------------------------------------------------------------------
#-- LASSO
#-------------------------------------------------------------------------------

#- Como se ve en las diapositivas, en LASSO se tiene una forma funcional para la
#- penalizacion. En lo demás, es muy similar a Ridge.
#--> Se genera una secuencia de valores de \lambda para evaluar el que minimiza RMSE
lambdas_to_try = 10^seq(-2, 5, length.out = 100)

#--> Clave: alpha=1 implementa LASSO en este procedimiento
lasso_cv = cv.glmnet(X, y, alpha = 1, lambda = lambdas_to_try,
                     standardize = TRUE, nfolds = 10)

#--> Graficar los resultados de la validacion cruzada
plot(lasso_cv)

# \lambda que minimiza MSE
lambda_cv = lasso_cv$lambda.min
# Se ajusta modelo final, calcula RSS y R cuadrado
model_cv = glmnet(X, y, alpha = 1, lambda = lambda_cv, standardize = TRUE)
y_hat_cv = predict(model_cv, X)
ssr_cv = t(y - y_hat_cv) %*% (y - y_hat_cv)
rsq_lasso_cv = cor(y, y_hat_cv)^2


#--> Notese como incrementar el lambda genera coeficientes cercanos a cero
#--> Cada linea muestra coeficientes para cada variable, para diferentes \lambdas.
#--> Cuanto mayor el lambda, mas coeficientes se vuelven cero.
res = glmnet(X, y, alpha = 1, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(X), cex = .7)

#-> Ridge vs LASSO segun R cuadrado
rsq = cbind("R-squared" = c(rsq_ridge_cv, rsq_ridge_aic, rsq_ridge_bic, rsq_lasso_cv))
rownames(rsq) <- c("ridge validación cruzada", "ridge AIC", "ridge BIC", "lasso Validación cruzada")
print(rsq)

#- Ridge y LASSO, en general, no tienden a mostrar un claro dominador.
#- LASSO puede hacer las veces de evaluador de cuales variables utilizar. Ridge no
#- Ambos metodos dejan usar variables explicativas correlacionadas:
#--- Ridge les ajusta coeficientes de valor similar
#--- LASSO deja uno con un coeficiente grande y el resto cero.

#-------------------------------------------------------------------------------
#- Elastic Net
#-------------------------------------------------------------------------------
#-> En este caso, hay dos parametros a determinar: \lambda y \alpha.
#-> libreria caret ayuda a calibrar \alpha y \lambda.

#-> Se determina los parametros del entrenamiento
train_control = trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "random",
                              verboseIter = TRUE, returnResamp = "all",
                             savePredictions = "all")

#-> Se entrena el modelo
elastic_net_model <- train(gdp ~ .,
                           data = cbind(y, X),
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 25,
                           trControl = train_control)

#--> Exploremos que dice el aviso
elastic_net_model$results

#--> R cuadrado no se calcula cuando lambda es mayor a uno.
#--> Generalmente sucede cuando: Hay missing values en las muestras
#--> El modelo no es realmente bueno, en términos de sus variables.

#--> Este es el resultado.
# Aggregating results
# Selecting tuning parameters
# Fitting alpha = 0.788, lambda = 0.0325 on full training set

#-> Resultado Elastic Net
elastic_net_model

#-> Se analiza el R cuadrado
#--> Prediccion del modelo
y_hat_enet <- predict(elastic_net_model, X)
y_hat_enet

#-> Calculo del R cuadrado
rsq_enet <- cor(y, y_hat_enet)^2

#-> Grafico del pronostico
plot(y, type = "l", lty=1)
lines(y_hat_enet, type = "l", lty=1, col=4)


