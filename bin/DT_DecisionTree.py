#####
# Filename: DT_DecisionTree.py
# Description: Contém a classe para arvore de decisão
###

# import DT_DataStructure
import logging

class DecisionTree:
    """Decision Tree Class"""
    def __init__(self, data_trainning, column_label_class: str, target_class: str):
        from pandas import DataFrame
        
        if not isinstance(data_trainning, DataFrame):
            raise TypeError('Expected value should descend from pandas.core.frame.DataFrame')
        self.__data_trainning = data_trainning.iloc[:,0:]
        
        self.__total_instances = self.__data_trainning.shape[0]
        
        #if (id_column_class >= data_trainning.shape[1]):
        #    raise TypeError('Expected value integer in limit for data frame')
        #self.__id_column_class = id_column_class
        
        self.__entropy_global = 0.0
        
        self.__root_tree = None
        
        self.__logger = True
        
        self.__target_class = target_class
        
        self.__column_label_class = column_label_class
        
        self.__columns = None
        
        self.__id_classifier = 'ClassifierTree'
        
        # self.__type_discret = (str)
        
        # self.__type_continuous = ('int', 'float')
        
        self.__method_continuos_supported = ('media', 'mediana', 'quantil', 'moda')
        
        self.__method_continuos_selected = 'media'
        
        self.__method_continuos_args = {'quantil': 0.5}

        
    def __entropy(self, class_positive: int, class_negative: int) -> float:
        from math import log2
        """
            Esta função calcula a entropia total de um conjunto de dados, observe 
                que o valor da entropia varia em função da precisão de ponto flutuante de python
            Input:  class_positive = numero de instâncias com classe positiva
                    class_negative = número de instâncias com classe negativa
            Output: (float) que representa a entropia total do conjunto
        """
        if (class_positive < 0) or (class_negative < 0):
            return -1
        if (class_positive == 0) or (class_negative == 0):
            return 0
        total_instance = class_positive + class_negative
        p = (class_positive/total_instance) * log2(class_positive/total_instance)
        q = (class_negative/total_instance) * log2(class_negative/total_instance)
        r__entropy = (-1 * p) - q
        
        msg = "[Entropy for calculate]:\tClass positive = {}\tClass negative = {}\tTotal instances = {}\tEntropy = {}"
        msg = msg.format(class_positive, class_negative, (class_positive+class_negative), r__entropy)
        logging.debug(msg)
        
        return r__entropy
    
    
    def __prepareCalcEntropy(self, ref_trainning: list) -> float:
        """
             Esta função lida com os dados do subconjunto para o cálculo da entropia, desta maneira 
                 de um lado temos a função que calcula a entropia e de outro uma função que prepara
                 os dados para este cálculo
            Input:  ref_trainning = uma lista com os nomes das colunas (subconjunto) para o cálculo
                        da entropia
            Output: (float) que representa a entropia total do conjunto
        """
        if (not self.__column_label_class in ref_trainning):
            ref_trainning.append(self.__column_label_class)
        
        msg = "[Prepare calc entropy]:\tColumn Label Trainning = {}"
        msg = msg.format(ref_trainning)
        logging.debug(msg)
        
        df_selected = self.__data_trainning.loc[:,ref_trainning]
        total_intances = self.__data_trainning[self.__column_label_class].count()
        mask = self.__column_label_class + ' == ' + '"' + self.__target_class + '"'
        class_positive = df_selected.query(mask)[self.__column_label_class].count()
        class_negative = total_intances - class_positive
        r__prepare_calc_entropy = self.__entropy(class_positive, class_negative)
        
        return r__prepare_calc_entropy
        
        
    def __gaugeStopRecursion(ref_trainning: list) -> bool:
        """ 
            Esta função determina quando a recursividade da arvore deve parar para a expansão
                Neste caso termina caso seja o úlitmo nó da lista
            Input: Dados de treinamento
            Output: (Bool) True = Stop, False = Continue
        """
        
        msg = "[Gauge Stop Recursion]:\t Label Trainning = {}"
        msg = msg.format(ref_trainning)
        logging.debug(msg)
        
        if len(ref_trainning) == 1:
            return True
        
        return False
    
    
    def __gainInformation(self, frequencies: Frequency, total_instances: int) -> float:
        """
            Esta função calcula o ganho de informação de um conjunto de dados, observe 
                que o valor do ganho de informação varia em função da precisão de ponto 
                flutuante de python
            Input:  class_positive = numero de instâncias com classe positiva
                    class_negative = número de instâncias com classe negativa
                    frequencies = uma lista de dicionário que representa os dados do conjunto
            Output: (float) que representa o ganho de informação total do conjunto
        """
        
        msg = '[Gain Information]:\tFrequencies = {}\t Total instances = {}'
        msg = msg.format(frequencies, total_instances)
        logging.debug(msg)
        
        gain_local = 0
        
        for frequency in frequencies:        
            frequency_entropy = self.__entropy(frequency['class_positive'], frequency['class_negative'])
            frequency_relative = frequency['class_positive'] + frequency['class_negative']
            gain_local += (frequency_relative / total_instances) * frequency_entropy

        r__gainInformation =  self.__entropy_global - gain_local
        
        msg = "[Gain Information]\tEntropy Global = {}\tEntropy Local = {}\tGain Information = {}"
        msg = msg.format(self.__entropy_global, gain_local, r__gainInformation)
        logging.debug(msg)
        
        return r__gainInformation
    
    
    def __get_point_cut(self, df_column) -> float:
        """
            Esta função calcula o ponto de divisão em uma série de acordo com as pŕedefinições da classe
            Input:  df_column = Coluna do dataframe
            Output: (float) que representa o valor do ponto de divisão
        """
        r__point_cut = 0
        if (self.__method_continuos_selected == 'media'):
            r__point_cut = df_column.mean()
        elif (self.__method_continuos_selected == 'mediana'):
            r__point_cut = df_column.median()
        elif (self.__method_continuos_selected == 'quantil'):
            r__point_cutter = df_column.quantile(q=self.__method_continuos_args['quantil'])
        elif (self.__method_continuos_selected == 'moda'):
            r__point_cut = df_column.mode()
        else:
            r__point_cut = df_column.mean()
        
        return r__point_cut;
    
    
    def __mountFrequencySeriesDiscrete(self, df_work, column_name: str) -> list:
        """
            Esta função monta a estrutura de dados de uma série discreta para ser usado no cálculo
                de ganho de informação
            Input:  df_work = DataFrame formado pela coluna de dados e pela coluna de classes
                    column_name = nome da coluna de dados do dataframe
            Output: (list) uma lista de dicionário com a estrutura (value, class_positive, class_negative)
        """
        column_frequency = []
        partitions = list(df_work[column_name].unique())
        for partition in partitions:
            mask_up = column_name + '==' + '"' + partition + '"'
            df_up = df_work.query(mask_up)
            total_instances = df_up[column_name].count()
            mask_target = self.__column_label_class + ' == ' + '"' + self.__target_class + '"'
            class_positive = df_up.query(mask_target)[column_name].count()
            class_negative = total_instances - class_positive
            column_frequency.append({'value': partition, 
                                     'class_positive': class_positive, 
                                     'class_negative': class_negative
                                    })
        return column_frequency
        
        
        
    def __mountFrequencySeriesContinuos(self, df_work, column_name: str) -> list:
        """
            Esta função monta a estrutura de dados de uma série continua para ser usada no cálculo
                de ganho de informação
            Input:  df_work = DataFrame formado pela coluna de dados e pela coluna de classes
                    column_name = nome da coluna de dados do dataframe
            Output: (list) uma lista de dicionário com a estrutura (value, class_positive, class_negative)
        """
        point_cut = self.__get_point_cut(df_work[column_name])
        column_frequency = []
        for unique_partition in ['<', '>=']:
            mask_up = column_name + unique_partition + str(point_cut)
            df_up = df_work.query(mask_up)
            total_instances = df_up[column_name].count()
            mask_target = self.__column_label_class + ' == ' + '"' + self.__target_class + '"'
            class_positive = df_up.query(mask_target)[column_name].count()
            class_negative = total_instances - class_positive
            column_frequency.append({'value': unique_partition, 
                                     'class_positive': class_positive, 
                                     'class_negative': class_negative
                                    })
        return column_frequency
                    
        
    # Deveria haver um parametro column_predecessor para que os dados fossem 
    # filtrados com base no nó anterior cujos valores determinam a classe
    def splitDecisionTree(self, columns_trainning: DataIdColumn) -> str:
        r_split = {}
        
        msg = "[Split Decision Tree]\tColumn Trainning = {}"
        msg = msg.format(columns_trainning)
        logging.debug(msg)
            
        for column in columns_trainning:
            
            msg = "[Split Decision Tree]\tColumn = {}"
            msg = msg.format(column)
            logging.debug(msg)
            
            type_column = type(self.__data_trainning[column][0])
            df_work = self.__data_trainning.loc[:,[self.__column_label_class, column]]
            total_instances_work = df_work[column].count()
            column_frequency = []
            if (type_column != str):
                column_frequency = self.__mountFrequencySeriesContinuos(df_work, column)
            else:
                column_frequency = self.__mountFrequencySeriesDiscrete(df_work, column)
            r_split[column] = self.__gainInformation(column_frequency, total_instances_work)
        
        msg = "[Split Decision Tree]\tGain Information = {}"
        msg = msg.format(r_split)
        logging.debug(msg)
            
        point_division_label = max(r_split, key=r_split.get)
        
        return point_division_label
    
    
    # @ TODO: Reve a criação do nó para o caso de parada de recursão
    def __gender(self, ref_trainning: list) -> Any:
        """
            Esta função gera recursivamente a arvore de decisão
            Input: Dados de treinamento
            Output: (Struct) a arvore de decisão
        """
        if (self.__gaugeStopRecursion(ref_trainning)):
            r__label = ref_trainning[0]
            r__classification = self.identifyClass(ref_trainning)
            r__test_condition = self.trunkCondition(ref_trainning)
            r__leaf_end = StructNode(label = r__label, 
                                     test_condition = r__test_condition, 
                                     leaf = True, 
                                     classification = r__classification)
            return r__leaf_end
        # else
        r__attribPointDivision = self.__splitDecisionTree(ref_trainning)
        
        
    def build(self) -> None:
        """ Esta função inicia o processo para criar a arvore de decisão"""
        if self.__logger:
            logging.basicConfig(filename=self.__getNameLog(),
                                level= logging.DEBUG, 
                                format='%(asctime)s - %(levelname)s - %(message)s')
            
        column_df = list(self.__data_trainning.columns)
        self.__columns = column_df
        self.__entropy_global = self.__prepareCalcEntropy(column_df)
        
        column_df.pop(self.__id_column_class)
        self.__root_tree = self.__gender(column_df)
        
    
    def setProperties(propertie: str, value: Any) -> None:
        pass
    
    
    def getProperties(propertie: str) -> Any:
        pass

    
    def __getNameLog(self) -> str:
        import time as lib_tm
        coded_time = str(lib_tm.localtime().tm_hour) + '_' + str(lib_tm.localtime().tm_min) + '_' + str(lib_tm.localtime().tm_sec)
        filename = self.__id_classifier + coded_time + '.log'
        return filename