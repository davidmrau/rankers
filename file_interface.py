import numpy as np
		
class File():
	def __init__(self, fname, encoded=True):
		self.file = {}
		count_empty_docs = 0
		with open(fname, 'r', encoding='utf-8') as f:
			for line in f:
				delim_pos = line.find('\t')
				id_ = line[:delim_pos]
				id_ = id_.strip()
				if encoded:
					# in case the encoded of the line is empy, and the line only contains the id, then we return None
					# example of line with empy text: line = "1567 \n" -> len(line[delim_pos+1:]) == 1
					if len(line[delim_pos+1:]) < 2:
						self.file[id_] = None
						#print('empty doc', id_)
						count_empty_docs += 1
					else:
						# extracting the token_ids and creating a numpy array
						self.file[id_] = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
				else:
					self.file[id_] = line[delim_pos+1:].strip()
		print(f'{fname} num empty docs:', count_empty_docs)
	def __getitem__(self, id_):
		if id_ not in self.file:	
			print(f'"{id_}" not found!')
			return ' '
		return self.file[id_]

	def __len__(self):
		return len(self.file)
