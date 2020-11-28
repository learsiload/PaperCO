import pandas as pd
from flask import Flask, request, render_template, url_for, redirect
import pickle
import numpy as np
import os
import lightgbm as lgb

from pred_scaler import pred_scaler

## Carregar  o modeelo na memoria para toda vez que inciar a API o modelo carregar, antes da solcitacao
# carregar o modelo
model_cl = pickle.load( open('model/modelo_Machine_Learning_PaperCO_CL.pkl', 'rb'))
model_re = pickle.load( open('model/modelo_Machine_Learning_PaperCO_RE.pkl', 'rb'))
# instanciar flask
app = Flask(__name__)

# endpoint
@app.route('/')
def pagina():
    return render_template("paperco.html")

@app.route('/predict', methods=['POST','GET'])
def predict():

#--------------------------------------------
# Coleta de dados da pagina
    lista = [x for x in request.form.values()]

    #----------SCALER------------------------
    pipeline = prep_scaler()
    lista = pipeline.data_preparacao(data_preparacao)


    df = pd.DataFrame(lista)
#---------------------------------------------
    pred_pagina_cl = model_cl.predict(df.T)
    pred_pagina_re = np.int(model_re.predict(df.T))
#---------------------------------------------
# Exibindo na pagina
    if pred_pagina_cl == 1:
        return render_template('paperco.html',
                               #pred='Máquina precisando de manutenção.\nDentro dos últimos 20 Ciclos.\nResultado da Predição: {}'.format(pred_pagina_cl),
                               pred='Máquina precisando de manutenção.\nDentro dos últimos 20 Ciclos',
                               bhai='Próxima quebra será daqui {}'.format(pred_pagina_re) +' dias')
    else:
        return render_template('paperco.html',
                               pred='Máquina fora do risco de quebra.\n Ainda faltam {}'.format(pred_pagina_re),
                               bhai="dias.")
#--------------------------------------------
if __name__ == '__main__':
    # star flask
    port = os.environ.get('PORT', 5000)
    app.run(host  = '0.0.0.0', port = port)

