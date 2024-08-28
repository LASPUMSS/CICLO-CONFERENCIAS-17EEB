#-------------------------------------------------------
# Codigo para implementar Evaluacion de pronostico
#
# Esta version: Agosto 2024
# Autor: Fernando Arias-Rodriguez
#-------------------------------------------------------
#-> Fuente:
#---> https://pure.au.dk/ws/files/164671164/Catania_2018_The_model_confidence_set_package_for_R.pdf
#---> https://cran.r-project.org/web/packages/multDM/multDM.pdf

# Limpiar workspace
rm(list = ls())

#-> Se estudian dos metodos:
#--> Diebold-Mariano: evaluar dos modelos a partir de su poder para pronosticar
#--> Model confidence interval: Evalua en un grupo de modelos el que mejor pronostica

# Limpiar workspace
rm(list = ls())

#-> Importar librerias
library(multDM)
library(rugarch)
library(MCS)

#-> Prueba de Diebold-Mariano
#--> Series artificiales para mostrar el funcionamiento de la prueba
Actual = c(1.22884,2.6684,3.41773,2.2392,2.12256,0.4638,-0.55081,1.18295,-2.4133,0.97947,0.55088,1.22792,-0.92351,-0.09028,1.68379,-0.61077,1.28104,-0.92225,-0.57811,0.7687)
fore1  = c(0.902837,2.4492678,3.2075581,2.4383221,2.7751086,0.5931617,0.1085186,0.8785177,-1.165313,0.5937193,-0.003627,0.9943153,0.5194248,0.285099,0.5713786,0.2233359,0.327581,0.0846889,-0.083991,-0.073104)
fore2  = c(0.8945434,2.3213521,2.5208157,1.908075,0.9507821,-0.610665,-1.11545,1.1116309,-2.777648,1.51728,0.4897679,1.047002,-1.344792,0.0019235,1.7465681,-1.063168,1.2719837,-1.289334,-0.464421,0.9785314)

#--> Grafico de comparacion
plot(Actual, type = "l", lty=1)
lines(fore1, type = "l", lty=1, col=4)
lines(fore2, type = "l", lty=1, col=3)

#--> test aplicado a los dos pronosticos (h=n/3+1)
DM1 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=4,c=FALSE,H1="same")

#--> Aplicando correccion de muestra pequenha
DM2 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=4,c=TRUE,H1="same")

#--> Cambiando el sentido de la prueba de hipotesis alternativa
DM3 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=4,c=TRUE,H1="more")

DM1
DM2
DM3

#-> Usando datos reales
#--> Pronostico precios del petroleo de muchos modelos
#--> Fuente: Juvenal y Petrella (2015)
data(oilforecasts)

#-> Tomamos el pronostico NAIVE (caminata aleatoria)
fore1 = oilforecasts[15,]

#-> Tomamos modelo LASSO
fore2 = oilforecasts[12,]

#-> Tomamos serie observada
Actual = oilforecasts[1,]

#-> Prueba de Diebold-Mariano
#--> test aplicado a los dos pronosticos
DM1 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=18,c=FALSE,H1="same")

#--> Aplicando correccion de muestra pequenha
DM2 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=18,c=TRUE,H1="same")

#--> Cambiando el sentido de la prueba de hipotesis alternativa
DM3 = DM.test(f1=fore1,f2=fore2,y=Actual,loss="SE",h=18,c=TRUE,H1="more")

DM1
DM2
DM3

#-> Li, Liao and 
#--> Material para replicar el documento: https://zenodo.org/records/4884813
#--> Prueba implementada en Stata: https://journals.sagepub.com/doi/abs/10.1177/1536867X221141014


#-> Model confidence sets
#--> Se usa la misma informacion de pronosticos para el precio del petroleo
pronos = as.matrix(oilforecasts) # Se deben poner los datos como matriz

perdida = (pronos[-1,]-pronos[1,])^2 # Se calcula funcion de perdida (cuadratica como en casos anteriores)

#--> Se aplica la prueba statistic puede ser "Tmax" o "TR"
MCS = MCSprocedure(Loss=perdida,alpha=0.1,B=5000,statistic='TR',cl=NULL)
MCS
