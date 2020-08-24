from karateclub.node_embedding.neighbourhood import Walklets
from karateclub.node_embedding.neighbourhood import HOPE
from karateclub.node_embedding.neighbourhood import DeepWalk
from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps
from karateclub.node_embedding.neighbourhood import Diff2Vec
from karateclub.node_embedding.neighbourhood import GraRep

def hope(dim):
   return  HOPE(dimensions=dim)

def deepwalk(dim):
    return DeepWalk(dimensions=dim)

def walklets(dim):
    return Walklets(dimensions=dim)

def le(dim):
    return LaplacianEigenmaps(dimensions=dim)

def diff2vec(dim):
    return Diff2Vec(dimensions=dim)

def grarep(dim):
    return GraRep(dimensions=dim)

def load(filename):
    emb = []
    with open(filename, "r") as f:
        next(f)
        for line in f:
            # first item == node
            emb.append(line.split(' ')[1:])
    return emb
