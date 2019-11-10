# Supposons que nous ayons trois variables aléatoires de rendement. Il faut bien
# comprendre que nous ne souhaitons pas traiter des variables aléatoires
# extérieures, mais bien des rendements qui sont corrélés les uns aux autres.

# On peut utiliser l'approche de la copule, puisque c'est une approche très
# flexible. 

import numpy as np
import numpy.linalg as la

n = 3
X = np.identity(n)

# Quel genre de corrélation veut-on voir?
# Supposons qu'on ait uniquement pour le moment des corrélations iid, donc le
# temps ne vient pas affecter les observations. 

