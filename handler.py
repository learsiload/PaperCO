import pandas as pd
from flask import Flask, request, render_template, url_for, redirect
from flask_bootstrap import Bootstrap 
import pickle
import numpy as np
import os
import lightgbm as lgb


## Carregar  o modeelo na memoria para toda vez que inciar a API o modelo carregar, antes da solcitacao
# carregar o modelo
model_cl = pickle.load( open('model/modelo_Machine_Learning_PaperCO_CL_8.pkl', 'rb'))
model_re = pickle.load( open('model/modelo_Machine_Learning_PaperCO_RE_10.pkl', 'rb'))

# instanciar flask
app = Flask(__name__)
Bootstrap(app)

# endpoint
@app.route('/')
def pagina():
    return render_template("paperco.html")

@app.route('/predict', methods=['POST','GET'])
def predict():

#--------------------------------------------
# Coleta de dados da pagina
    #---- tudo em uma lista (1,4), poré apenas uma feature.
    lista = [float(x) for x in request.form.values()]
    print("lista handler",lista)
    print("lista handler",type(lista))
    print("lista shape",np.array(lista).shape)

    lista1 = np.array(lista)
    lista1 = lista1.reshape(-1,1)
    print("lista1 size: ",lista1.size)
    #print('lista1 reshape',lista1.shape)
    #----------SCALER------------------------
    #---Para esse método, o shape precisa ser: lista1 reshape (4, 1)-----#
    #--ou seja os dados devem estar em linhas....

    range_min_max = [[641.2100,1382.2500,549.8500,9021.7300,46.8500,518.6900,8099.9400,8.3249,38.1400,22.8942],[644.5300,1441.4900,556.0600,9244.5900,48.5300,523.3800,8293.7200,8.5848,39.4300,23.6184]]
    x_new = []
    for i in range(lista1.size):
        x_new.append((lista1[i] - range_min_max[0][i]) / (range_min_max[1][i] - range_min_max[0][i]))
    #print(x_new)

    df = pd.DataFrame(x_new)
    indices_cl = [0,1,2,4,5,7,8,9]
    indices_re = [0,1,2,3,4,5,6,7,8,9]
    #print('df: ',df)
    #print('df.T: ',df.T)
    df_cl = df.iloc[indices_cl,:]
    df_re = df.iloc[indices_re,:]
#---------------------------------------------
    pred_pagina_cl = model_cl.predict(df_cl.T)
    pred_pagina_re = np.int(model_re.predict(df_re.T))
    
    #print('classificacao: {}'.format(pred_pagina_cl))
#---------------------------------------------

    if pred_pagina_cl == 1:
        pred='Ops! Máquina precisando de manutenção. Dentro dos últimos 20 Ciclos se espera uma parada.'
        bhai='Próxima quebra será daqui {}'.format(pred_pagina_re) +' dias. Acione a equipe de manutenção.'
    else:
        pred='Muito bem, a máquina e suas variáveis estão cooperando para um bom funciomanento.' 
        bhai='O Sistema de previsão diz que ainda restam {}'.format(pred_pagina_re) + ' dias.'
    return render_template('paperco.html',predx = pred_pagina_cl, bhaix = pred_pagina_re, pred = pred , bhai = bhai ) 

#--------------------------------------------
if __name__ == '__main__':
    app.run(host  = '0.0.0.0', port = '5000', debug=False)

