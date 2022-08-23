> Postulat : à chaque bateau du jeu de donnée correspond une unique forme. Deux bateau ayant la même forme désignent le même bateau.

    Chaque bateau contenus par les images est décrit par un RLE (run-length-encoding)
    Le format RLE porte deux informations :
    1 - la forme du bateau 
    2 - sa position dans l'image
    La première est unique au sein du jeu de donnée (cf. postulat), la deuxième diffère selon les images.Ainsi un même bateau peut avoir un RLE différent s'il est présent sur deux images.Par conséquent, on considère donc que la première de ces deux informations identifie de façon unique un bateau du jeu de données. En tirant du RLE de chaque bateau la forme désignée, on identifie donc ce bateau de façon unique. Cet identifiant, un entier généré à partir de cette information à l'aide d'une fonction de hachage, sera nommé BoatHash.


* boats_hash.CSV :
Généré par la fonction hash_boat_rle du module img_hash.py
Ce fichier CSV associe à chaque image contenant au moins un bateau et pour chacun des bateaux contenus :
- son identifiant (BoatHash)
- son RLE, spécifique l'image concernée
- sa largeur (W) et sa hauteur (H) correspondant à celles du rectangle d'aire minimale qui englobe l'intégralité du bateau. 

* imgs_per_boats.CSV :
Généré par la fonction imgs_per_b_csv du module img_hash.py
Ce fichier csv associe à chaque bateau, identifié par son BoatHash, les images qui le contiennent. 

* clusters_h.pkl : 
Généré par la fonction find_clusters du module img_hash.py
Sauvegarde au format binaire à partir du module pickle de python de la liste des clusters généré par find_clusters.

* big_clust.csv : 
Généré par la fonction big_clust_csv du module reassemble_cluster.py
Fichier csv qui associe à chaque cluster de clusters_h.pkl, c'est-à-dire les clusters trouvés par la méthode réseau en ne se basant que sur les images avec bateau, son "big_clust", c'est-à-dire le cluster trouvé par la méthode mosaïque (en reconstruisant les images satellites) qui contient ce cluster.
Chaque "big_clust" est identifié par un index. Cet index est l'index du cluster dans la liste des clusters généré par la méthode mosaïque.

* cluster_reassembled.pkl :
Généré par la fonction reassemble_clusters du module reassemble_cluster.py.
Sauvegarde au format binaire à partir du module pickle de python de la liste des clusters de cluster_h.pkl où deux clusters ont été reassemblés dans le cas où ils appartiendrait à la même image satellite.

* cluster.csv :
Généré par la fonction cluster_csv du module reassemble_cluster.py
Fichier csv qui décrit les clusters de clusters_reassembled.pkl, c'est-à-dire les clusters finaux qui seront utilisés pour le split train/test. Associe à chaque cluster les images qu'il contient ainsi que les bateaux contenus (identifiés par leur BoatHash).

* train_ship_segmentation_OD.csv : 
Généré par la fonction generate_od_csv du module generate_od_csv.py
Fichier csv au format pascal VOC pour la détection d'objets.

* cluster_sizes.csv : 
Généré par la fonction cluster_sizes_csv du module sizes.py.
Fichier CSV qui associe à chaque cluster les largeur et hauteur moyennes des bateaux qu'ils contient ainsi que leur nombre. 

* clusters.pkl
Généré par la fonction main du module cluster.py
Sauvegarde au format binaire à partir du module pickle de python de la liste des clusters généré par l'approche mosaïque, c'est-à-dire celle qui tente de reconstruire les images satellites.
Beaucoup de duplicatas de clusters présents, dus à l'approche multiprocessing où certain cluster ont été trouvés en parallèle par plusieurs coeurs de calcul.

* clusters_clean.pkl 
Généré par le module clean_cluster.py
Sauvegarde au format binaire à partir du module pickle de python de la liste des clusters généré par l'approche mosaïque, version "propre" où les duplicatas décrits ci-dessus ont été supprimés.








