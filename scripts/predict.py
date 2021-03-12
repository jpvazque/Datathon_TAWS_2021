import sys
from clustering import *
lista=sys.argv

filename_cluster=lista[1]
pruebas,model_gm,pruebas_escalada=generate_cluster(filename_cluster)
print("Log-likelihood score for number of cluster {}: {}".format(11,model_gm.score(pruebas_escalada)))

