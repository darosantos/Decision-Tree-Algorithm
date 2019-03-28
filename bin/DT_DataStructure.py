#####
# Filename: DT_DataStructure.py
# Description: Contém a definição dos tipos de dados utilizados no projeto
###
from typing import List, Dict, Any

# Alias for type
DataFrequency = Dict[str, Any]

Frequency = List[DataFrequency]

DataIdColumn = List[str]

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
class StructNode(Struct):
    pass