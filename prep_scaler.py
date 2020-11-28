import pickle

class prep_scaler( object):
	def __int__(self):
		self.prep = pickle.load(open('model/prep_scaler.pkl', 'rb'))

	def data_preparacao(self, array):
		array = self.prep.transform(array)
		return array

