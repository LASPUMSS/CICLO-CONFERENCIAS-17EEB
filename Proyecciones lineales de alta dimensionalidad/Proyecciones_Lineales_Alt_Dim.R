#-------------------------------------------------------
# Codigo para implementar BRIDGE, MIDAS y UMIDAS
# Ejemplo con informacion para EE.UU.
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------

# Limpiar workspace
rm(list = ls())

#--> Importar librerias
library(midasr)
library(ggplot2)
library(ggfortify)
library(dplyr)


#-----------------------------------------
#- Regresiones Bridge
#-----------------------------------------
#--> Objetivo: Pronosticar crecimiento de PIB de EEUU (trimestral) con el crecimiento mensual 
#              de los empleados no agricolas.

#--> Importar datos
data("USqgdp")
data("USpayems")

#--> Graficas los datos (¡siempre hacerlo!)
autoplot(USqgdp, xlab='Trimestres', ylab='PIB')
autoplot(USpayems, xlab='Meses', ylab='Empleo')

#--> Como empleados es una variable stock, se promedian datos mensuales para
#--> hallar el dato trimestral con el cual hacer la regresion para la baja frecuencia
#--> Agregar la variable en frecuencia mensual
x_q = aggregate(USpayems, nfrequency = 4)/3

plot(cbind(USqgdp, x_q), yax.flip = TRUE, col = "blue", frame.plot = TRUE, 
main = expression("Series para Bridge"), xlab = "Trimestres")

#--> Estimar regresion Bridge para la baja frecuencia
#----> 1. Volver las series estacionarias
yg <- diff(log(USqgdp))*100
xg <- diff(log(x_q))*100

#----> 2. Estimando la regresion
dlgdp  = window(yg, end=c(2011,2))
dlpayr = window(xg, start=c(1947,2), end=c(2011,2))
bridge.lf = lm(dlgdp~dlpayr)
summary(bridge.lf)

#--> Modelo de pronostico para serie en alta frecuencia
bridge.flash = rep(0,3)
x_h = diff(log(USpayems))*100

#----> Idea: se va añadiendo de a un mes adicional observado del trimestre a pronosticar
#----> y se recalcula el pronostico de la serie trimetral objetivo.
for (n in 1:2) {
  #----> Se estima un modelo AR(1), el mas simple
  ar.fit     = ar(window(x_h, end=c(2011,6+n)), order.max=1, method='ols')
  ar.predict = as.numeric(predict(ar.fit, n.ahead=3-n)$pred)
  bridge.flash[n] = bridge.lf$coefficients[1]+
         bridge.lf$coefficients[2]*sum(window(x_h, start=c(2011,7), end=c(2011,7+(n-1))), sum(ar.predict))/3
}
bridge.flash[3] = bridge.lf$coefficients[1]+
  bridge.lf$coefficients[2]*window(xg, start=c(2011,3), end=c(2011,3))  

window(yg, start=c(2011,2), end=c(2011,4))
bridge.flash

#--> Extensiones:
#----> Estimar diferentes modelos para la serie de alta frecuencia.
#----> Añadir mas variables a la ecuacion de la serie de baja frecuencia.


#-----------------------------
#- MIDAS y UMIDAS
#-----------------------------
# Para mucha mas informacion, consultar 
# https://github.com/mpiktas/midasr-user-guide/blob/master/midasr-examples.R
# https://www.jstatsoft.org/article/view/v072i04

#--> Importar datos (otra vez)
data("USqgdp")
data("USpayems")

#--> Definir las fechas para los ultimos datos de cada serie
y <- window(USqgdp, end = c(2011, 2))
x <- window(USpayems, end = c(2011, 7))

#--> Calcular las diferencias logaritmicas de los datos
yg <- diff(log(y))*100
xg <- diff(log(x))*100

#--> Darle propiedad de series de tiempo a los datos
nx <- ts(c(NA, xg, NA, NA), start = start(x), frequency = 12)
ny <- ts(c(rep(NA, 33), yg, NA), start = start(x), frequency = 4)

#--> Graficas los datos (¡siempre hacerlo!)
plot.ts(nx, xlab = "Tiempo", ylab = "Porcentajes", col = 4, ylim = c(-5, 6))
lines(ny, col = 2)


#--> Homogenizar las muestras para implementar el ejercicio
xx <- window(nx, start = c(1985, 1), end = c(2009, 3))
yy <- window(ny, start = c(1985, 1), end = c(2009, 1))

#--> Estimar los modelos
#--> Especificaciones del modelo:
#----> 1. PIB se modela con un AR(1).
#----> 2. Salarios se modelan con rezagos desde 3 al 11.

#----> mls es el pedazo de codigo clave: mls(serie, rezagos, frecuencia de la serie, polinomio)
#----> En este caso se restringe MIDAS a 2 parametros y pol - almon
almon0 <- midas_r(yy ~ mls(yy, 1, 1) + mls(xx, 3:11, 3, nealmon), start = list(xx = c(2, 0.5)))
summary(almon0)

#----> ¿Cual deberia ser el mejor modelo?
#----> Se usa la instruccion midas_r_ic_table para usar criterios de informacion
#----> La usaremos con polinomios Almon y Beta.

#----> 1. Se construye un cuadro con todos los experimentos a proponer
set_x1 <- expand_weights_lags(weights = c("nealmon", "nbeta"), from = 3, 
                              to = c(11, 11), m = 1, start = list(nealmon = c(1, 1), 
                                                                nbeta = rep(0.5, 3)))

#----> 2. Se construye el cuadro con todos los modelos estimables
eqs_ic <- midas_r_ic_table(yy ~ mls(yy, 1, 1) + mls(xx, 3:11, 3), table = list(xx = set_x1))

#----> 3. Se hallar el mejor modelo, usando AIC
modsel(eqs_ic, IC = "AIC", type = "restricted")

#----> Por supuesto, puede evaluarse la seleccion del mejor modelo con validacion
#----> cruzada. Por ejemplo, se pueden poner a prueba este modelo con dos alternativas
#----> escogidas por el investigador: usando beta y sin restricciones (estilo U-MIDAS)

#----> a. Modelo escogido con AIC
mod_aic = midas_r(yy ~ mls(yy, 1, 1) + mls(xx, 3:11, 3, nealmon), start = list(xx = c(2, -0.5)))
summary(mod_aic)

#----> b. Modelo con polinomio beta
betam <- midas_r(yy ~ mls(yy, 1, 1) + mls(xx, 3:11, 3, nbeta),
                  start = list(xx = c(2, 1, 5)))
summary(betam)

#----> c. Modelo sin restricciones polinómicas
um <- midas_r(yy ~ mls(yy, 1, 1) + mls(xx, 3:11, 3), start = NULL)
summary(um)

#--> Evaluar pronostico.
#----> Se tiene una muestra de evaluacion, desde 2009:T3 a 2011:T2
#----> Se elegira el modelo que mejor pronostique para esos 9 trimestres

#----> Estructura de la muestra completa
fulldata <- list(xx = window(nx, start = c(1985, 1), end = c(2011, 6)),
                    yy = window(ny, start = c(1985, 1), end = c(2011, 2)))

#----> Estructura de la muestra de estimacion
insample <- 1:length(yy)

#----> Estructura de la muestra de evaluacion
outsample <- (1:length(fulldata$yy))[-insample]

#----> Instruccion para evaluar la habilidad de pronostico de cada modelo
avgf <- average_forecast(list(mod_aic, betam, um), data = fulldata,
                             insample = insample, outsample = outsample)

#----> Se usa el Error Cuadratico Medio (RMSE) para realizar la evaluacion
eval_pron = sqrt(avgf$accuracy$individual$MSE.out.of.sample)

#----> En la consola aparecen los RMSE de cada modelo, en orden dado en avgf:
#----> (mod_aic, betam, um). En este caso, el mejor es el modelo um.
eval_pron

#-> Unrestricted MIDAS
#--> Ya hemos hecho un U-MIDAS sin quererlo. Sin embargo, la libreria tiene otra forma de hacerlo
#--> Se necesita tener los objetos en forma de "base de datos"
xumidas <- data.frame(window(nx, start = c(1985, 1), end = c(2009, 3)))
yumidas <- data.frame(window(ny, start = c(1985, 1), end = c(2009, 1)))

#--> Se usa comando midas_u
umidas_ex = midas_u(yy ~ mls(yy, 1, 1) + fmls(xx, 11, 3), list(yumidas, xumidas) )

#---> Sin embargo, esto es equivalente a hacer
umidas_ex_r = midas_r( yy ~ mls(yy, 1, 1) + mls(xx, 0:11, 3), start = NULL )

#---> En este caso, es conveniente seguir con el comando "midas_r" para implementar
#---> U-MIDAS.

