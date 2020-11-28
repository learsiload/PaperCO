import pickle

class prep_scaler( object):
	def __int__(self):
		self.prep = pickle.load(open('/Users/israeljose/OneDrive/Desafio_Comunidade_Ciencia_de_Dados_Suzano/modelos/prep_scaler.pkl', 'rb'))

	def data_preparacao(self, array):
		array = self.prep.transform(array)
		return array

