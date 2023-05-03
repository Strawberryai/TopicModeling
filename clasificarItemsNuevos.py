# Imports del script
from sys import exit, argv, version_info
from getopt import getopt
import os

import re
import emoji
import string
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from nltk.stem import PorterStemmer

import pickle

from imblearn.under_sampling            import RandomUnderSampler
from sklearn.model_selection            import train_test_split
from sklearn.metrics                    import f1_score
from sklearn.metrics                    import classification_report
from sklearn.metrics                    import confusion_matrix
from sklearn.feature_extraction.text    import CountVectorizer, TfidfVectorizer

# Variables globales
OUTPUT_PATH     = ""                            # Path de los archivos de salida
INPUT_TEST      = "TweetsTestSubSample.csv"     # Path del archivo de datos a clasificar
INPUT_MODEL     = "./models/BNB.sav"            # Path del modelo
TARGET_NAME     = "airline_sentiment"                            # Nombre de la columna que contiene la clase real
ATRIBUTOS       = ['text', 'airline_sentiment'] # Atributos que seleccionamos del dataset | TODOS o lista
TRAIN_DATASET   = "TweetsTrainDev.csv"          # Archivo de train para generar el diccionario

TWEET_ATRIB     = "text"                        # Atributo de entrada de tweets

DEMOJI          = True                          # Tener en cuenta los emojis transformándolos en texto
CLEANING        = True                          # Limpiar textos: carácteres especiales, abrebiaturas, signos de puntuación...
STOP_WORDS      = False                         # Tratar las stop words
FREQ_WORDS      = True                          # Borramos las palabras más frecuentes. Pueden no aportar demasiada información.
LEMATIZE        = True                          # Lematizamos el texto (realmente hacemos Stemming)
VECTORIZING     = "TFIDF"                       # Sistema de vectorización: BOW | TFIDF

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: entrenar.py <optional-args>")
    print("The options supported by entrenar are:")
    print(f"-h, --help          show the usage")
    print(f"-o, --output        output file path for the weights                DEFAULT: ./{OUTPUT_PATH}")
    print(f"-i, --input         input file path of the data                     DEFAULT: ./{INPUT_TEST}")
    print(f"-m, --model         input file path of the model                    DEFAULT: ./{INPUT_MODEL}")
    print(f"-t --target         target name to predict                          DEFAULT: {TARGET_NAME}")
    print(f"-d --train_dataset  train dataset file path                         DEFAULT: {TRAIN_DATASET}")
    print("")
    
    print(f"Example: clasificarItemsNuevos.py -i TweetsTestSubSample.csv -m ./models/BNB.sav -t airline_sentiment -d TweetsTrainDev.csv")

    # Salimos del programa
    exit(1)

def load_options(options):
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_TEST, INPUT_MODEL, OUTPUT_PATH, PREPROCESADO, TARGET_NAME

    for opt,arg in options:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-o', '--output'):
            OUTPUT_PATH = str(arg)
        elif opt in ('-i', '--input'):
            INPUT_TEST = str(arg)
        elif opt in ('-m', '--model'):
            INPUT_MODEL = str(arg)
        elif opt in ('-t', '--target'):
            TARGET_NAME = str(arg)
        elif opt in ('-d', '--train_dataset'):
            TRAIN_DATASET = str(arg)

def show_script_options():
    # PRE: ---
    # POST: imprimimos las configuración del script
    print("entrenar.py configuration:")
    print(f"-o                  output file path            -> {OUTPUT_PATH}")
    print(f"-i                  input test file path        -> {INPUT_TEST}")
    print(f"-m                  input model file path       -> {INPUT_MODEL}")
    print(f"-t                  target name                 -> {TARGET_NAME}")
    print("")

#######################################################################################
#                               METHODS AND FUNCTIONS                                 #
#######################################################################################
def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    
    # Si no es anterior a la version 3 de python
    return str(x)

def estandarizar_tipos_de_datos(dataset, categorical_features, numerical_features, text_features):
    # PRE: dataset y listas qué atributos son categóricos, numéricos y de texto del dataset
    # POST: devuelve las features categoriales y de texto en formato unicode y las numéricas en formato double o epoch (si son fechas)
    for feature in categorical_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(dataset[feature].dtype, 'base') and dataset[feature].dtype.base == np.dtype('M8[ns]')):
            dataset[feature] = datetime_to_epoch(dataset[feature])
        else:
            dataset[feature] = dataset[feature].astype('double')

def atributos_excepto(atributos, excepciones):
    # PRE: lista completa de atributos y lista de aquellos que no queremos seleccionar
    # POST: devolvemos una lista de atributos
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)

    return atribs

#######################################################################################
#                               PREPROCESADO DE TEXTO                                 #
#######################################################################################
def cleaning(text):
    # Borrar menciones
    text = re.sub(r'@\w+', '', text)

    # removing short form:

    # text = text.replace("u", 'you')
    text = text.replace("isn't ", ' is not ')
    text = text.replace("he's ", ' he is ')
    text = text.replace("wasn't ", ' was not ')
    text = text.replace("there's ", ' there is ')
    text = text.replace("couldn't ", ' could not ')
    text = text.replace("won't ", ' will not ')
    text = text.replace("they're ", ' they are ')
    text = text.replace("she's ", ' she is ')
    text = text.replace("There's ", ' there is ')
    text = text.replace("wouldn' t", ' would not ')
    text = text.replace("haven't ", ' have not ')
    text = text.replace("That's ", ' That is ')
    text = text.replace("you've ", ' you have ')
    text = text.replace("He's ", ' He is ')
    text = text.replace("what's ", ' what is ')
    text = text.replace("weren't ", ' were not ')
    text = text.replace("we're ", ' we are ')
    text = text.replace("hasn't ", ' has not ')
    text = text.replace("you'd ", ' you would ')
    text = text.replace("shouldn't", ' should not ')
    text = text.replace("let's ", ' let us ')
    text = text.replace("they've ", ' they have ')
    text = text.replace("You'll ", ' You will ')
    text = text.replace("i'm ", ' i am ')
    text = text.replace("we've ", ' we have ')
    text = text.replace("it's ", ' it is ')
    text = text.replace("don't ", ' do not ')
    text = text.replace("that´s ", ' that is ')
    text = text.replace("I´m ", ' I am ')
    text = text.replace("it’s ", ' it is ')
    text = text.replace("she´s ", ' she is ')
    text = text.replace("he’s' ", ' he is ')
    text = text.replace('I’m ', ' I am ')
    text = text.replace('I’d ', ' I did ')
    text = text.replace("he’s' ", ' he is ')
    text = text.replace('there’s ', ' there is ')

    # abreviaciones inglesas
    text = text.replace(' lol ', " laughing ")
    text = text.replace(' rofl ', " laughing ")
    text = text.replace(' brb ', " be right back ")
    text = text.replace(' lol ', ' laughing ')
    text = text.replace(' rofl ', ' laughing ')
    text = text.replace(' brb ', ' be right back ')
    text = text.replace(' ily ', ' i love you ')
    text = text.replace(' ty ', ' thank you ')
    text = text.replace(' imy ', ' i miss you ')
    text = text.replace(' yolo ', ' you only live once ')
    text = text.replace(' fomo ', ' fear of missing out ')
    text = text.replace(' idk ', ' i do not know ')
    text = text.replace(' idc ', ' i do not care ')
    text = text.replace(' ffs ', ' for freaks sake ')
    text = text.replace(' smh ', ' shake my head ')
    text = text.replace(' ngl ', ' not going to lie ')
    text = text.replace(' w ', ' with ')
    text = text.replace(' abt ', ' about ')
    text = text.replace(' r ', ' are ')
    text = text.replace(' gtg ', ' going to go ')
    text = text.replace(' nvm ', ' never mind ')
    text = text.replace(' bcoz ', ' because ')
    text = text.replace(' coz ', ' because ')
    text = text.replace(' bcos ', ' because ')
    text = text.replace(' cld ', ' could ')
    text = text.replace(' ez ', ' easy ' )
    text = text.replace(' fbm ', ' fine by me ')
    text = text.replace(' ftw ', ' for the win ')
    text = text.replace(' fyi ', ' for your information ')
    text = text.replace(' ik ', ' i know ')
    text = text.replace(' wfh ', ' work from home ')
    text = text.replace(' lmfao ', ' laughing my freaking ass off ')
    text = text.replace(' lmk ', ' let me know ')
    text = text.replace(' af ', ' as freak ')
    text = text.replace(' aight ', ' alright ')
    text = text.replace(' awol ', ' away without leaving ')
    text = text.replace(' irl ', ' in real life ')
    text = text.replace(' bt ', ' bad trip ')
    text = text.replace(' bb ', ' baby ')
    text = text.replace(' btw ', ' by the way ')
    text = text.replace(' cu ', ' see you ')
    text = text.replace(' idgaf ', " i do not give a freak ")
    text = text.replace(' dgaf ', " do not give a freak ")
    text = text.replace(' df ', ' the freak ')
    text = text.replace(' dis ', ' this ')
    text = text.replace(' dm ', ' direct message ')
    text = text.replace(' dnt ', " do not ")
    text = text.replace(' dw ', ' do not worry ')
    text = text.replace(' enf ', ' enough ')
    text = text.replace(' eta ', ' estimated time of arrival ')
    text = text.replace(' ftw ', ' for the win ')
    text = text.replace(' fu ', ' freak you ')
    text = text.replace(' fwm ', ' fine with me ')
    text = text.replace(' gg ', ' good game ')
    text = text.replace(' gn ', ' good night ')
    text = text.replace(' gm ', ' good morning ')
    text = text.replace(' gr8 ', ' great ')
    text = text.replace(' grl ', ' girl ')
    text = text.replace(' grw ', ' get ready with me ')
    text = text.replace(' h8 ', ' hate ')
    text = text.replace(' hbd ', ' happy birthday ')
    text = text.replace(' hbu ', ' how about you ')
    text = text.replace(' hru ', ' how are you ')
    text = text.replace(' hw ', ' homework ')
    text = text.replace(' idts ', ' i do not think so ')
    text = text.replace(' ig ', ' instagram ')
    text = text.replace(' ilysm ', ' i love you so much ')
    text = text.replace(' imo ', ' in my opinion ')
    text = text.replace(' jk ', ' just kidding ')
    text = text.replace(' k ', ' okay ')
    text = text.replace(' ldr ', ' long distance relationship ')
    text = text.replace(' l2g ', ' like to go ')
    text = text.replace(' ly ', ' love you ')
    text = text.replace(' mfw ', ' my face when ')
    text = text.replace(' m8 ', ' mate ')
    text = text.replace(' nbd ', ' no big deal ')
    text = text.replace(' nsfw ', ' not safe for work ')
    text = text.replace(' nm ', ' nothing much ')
    text = text.replace(' np ', ' no problem ')
    text = text.replace(' nw ', ' no way ')
    text = text.replace(' og ', ' original gangster ')
    text = text.replace(' ofc ', ' ofcourse ')
    text = text.replace(' omg ', ' oh my god ')
    text = text.replace(' omfg ', ' oh my freaking god ')
    text = text.replace(' ootd ', ' outfit of the day ')
    text = text.replace(' otb ', ' off to bed ')
    text = text.replace(' otw ', ' off to work ')
    text = text.replace(' pm ', ' private message ')
    text = text.replace(' ppl ', ' people ')
    text = text.replace(' prob ', ' probably ')
    text = text.replace(' qt ', ' cutie ')
    text = text.replace(' rly ', ' really ')
    text = text.replace(' sh ', ' same here ')
    text = text.replace(' sis ', ' sister ')
    text = text.replace(' bro ', ' brother ')
    text = text.replace(' sry ', ' sorry ')
    text = text.replace(' sup ', ' what is up ')
    text = text.replace(' tbh ', ' to be honest ')
    text = text.replace(' thnk ', ' thank you ' )
    text = text.replace(' thx ', ' thanks ')
    text = text.replace(' ttly ', ' totally ')
    text = text.replace(' ttyl ', ' talk to you later ')
    text = text.replace(' ur ', ' you are ')
    text = text.replace(' wb ', ' welcome back ')
    text = text.replace(' whatevs ', ' whatever ')
    text = text.replace(' wyd ', ' what are you doing ')
    text = text.replace(' wdyk ', ' what do you know ')
    text = text.replace(' wru ', ' where are you ')
    text = text.replace(' wtf ', ' what the freak ')
    text = text.replace(' wtg ', ' way to go ')
    text = text.replace(' wywh ', ' wish you were here ')
    text = text.replace(' XD ', ' laugh ')
    text = text.replace(' xoxo ', ' hugs and kisses ')
    text = text.replace(' xo ', ' hugs and kisses ')
    text = text.replace(' y ', ' why ')
    text = text.replace(' tryna ', ' trying to be ')

    text = text.lower()  # converting to lowercase
    text = text.replace('https?://\S+|www\.\S+', '')  # removing URL links
    text = text.replace(r"\b\d+\b", "")  # removing number
    text = text.replace('<.*?>+', '')  # removing special characters,
    text = text.replace('[%s]' % re.escape(string.punctuation), '')  # punctuations
    text = text.replace('\n', '')
    text = text.replace('[’“”…]', '')
    text = text.replace('  ', ' ')

    # emojis de texto
    text = text.replace(" :D ", ' smile ')
    text = text.replace(re.escape(" :)"), ' smile ')
    text = text.replace(re.escape(" :-)"), ' smile ')
    text = text.replace(re.escape(" =)"), 'smile')
    text = text.replace(' :-D', ' smile ')
    text = text.replace(re.escape(" :("), ' sad ')
    text = text.replace(' :C', 'sad')
    text = text.replace(re.escape(" :-("), ' sad ')
    text = text.replace(re.escape(" :c"), ' sad ')
    text = text.replace(re.escape(" :@"), ' sad ')
    text = text.replace(re.escape(" =("), ' sad ')
    text = text.replace(' :-C', ' sad ')
    text = text.replace(' 0_0', ' surprise ')
    text = text.replace(' o.o', ' surprise ')
    text = text.replace(' o.O', ' surprise ')
    text = text.replace(' O.o', ' surprise ')
    text = text.replace(re.escape(" :‑O"), ' surprise ')
    text = text.replace(re.escape(" :‑o"), ' surprise ')
    text = text.replace(re.escape(" :-0"), ' surprise ')
    text = text.replace(re.escape(" >:("), ' angry ')
    text = text.replace(re.escape(" >:c"), ' angry ')
    text = text.replace(re.escape(" >:["), ' angry ')
    text = text.replace(re.escape(" :'("), ' cry ')
    text = text.replace(re.escape(" :'-("), ' cry ')
    text = text.replace(re.escape(' :"C'), ' cry ')
    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower()  # converting to lowercase
    text = text.replace('https?://\S+|www\.\S+', '')  # removing URL links
    text = text.replace(r"\b\d+\b", "")  # removing number
    text = text.replace('<.*?>+', '')  # removing special characters,
    text = text.replace('[%s]' % re.escape(string.punctuation), '')  # punctuations
    text = text.replace('\n', '')
    text = text.replace('[’“”…]', '')

    # removing emoji:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'',text)

    return text

def remove_freqwords(text, freq_words):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in freq_words])

def remove_stopwords(text, stop_words):
    """custom function to remove the stop words"""
    return " ".join([word for word in str(text).split() if word not in (stop_words)])

def get_freqwords(text_col):
    cnt = Counter()
    for text in text_col.values:
        for word in text.split():
            cnt[word] += 1
    cnt.most_common(10)
    return set([w for (w, wc) in cnt.most_common(10)])

def preprocesar_texto(dataset):
    # Tratamos los emojis
    if DEMOJI:
        dataset['demoji'] = dataset[TWEET_ATRIB].apply(lambda text: emoji.demojize(text))
    else:
        dataset['demoji'] = dataset[TWEET_ATRIB]

    # Limpiamos el texto
    if CLEANING:
        dataset['cleaned'] = dataset['demoji'].apply(cleaning) # Limpiamos
    else:
        dataset['cleaned'] = dataset['demoji']

    # Tratamos las stop words
    if STOP_WORDS:
        stop_words = stopwords.words()
        dataset['no_sw'] = dataset['cleaned'].apply(lambda text: remove_stopwords(text, stop_words)) # Eliminamos las stop words
    else:
        dataset['no_sw'] = dataset['cleaned'] # sin borrar stop words

    # Tratamos las palabras más frecuentes
    if FREQ_WORDS:
        freq_words = get_freqwords(dataset['no_sw'])
        dataset["wo_stopfreq"] = dataset["no_sw"].apply(lambda text: remove_freqwords(text, freq_words)) # Eliminamos las más frecuentes
    else: 
        dataset["wo_stopfreq"] = dataset["no_sw"]

    # Lematizamos (hemos probado stemming pero conseguimos peores resultados)
    if LEMATIZE:
       #wordnet_lem = WordNetLemmatizer()
       #dataset['wo_stopfreq_lem'] = dataset['wo_stopfreq'].apply(wordnet_lem.lemmatize) # Lematizamos las palabras

        porter = PorterStemmer()
        dataset["wo_stopfreq_lem"] = dataset["wo_stopfreq"].apply(porter.stem)
    else:
        dataset['wo_stopfreq_lem'] = dataset['wo_stopfreq']

#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main():
    # Entrada principal del programa
    print("---- Iniciando main...")
    print(f"---- Cargando el modelo {INPUT_MODEL}")
    clf = pickle.load(open(INPUT_MODEL, 'rb'))

    print(f"---- Cargando los datos a clasificar {INPUT_TEST}")
    # Cargamos los datos de entrada y seleccionamos los atributos
    train_dataset = pd.read_csv(TRAIN_DATASET)
    test_dataset = pd.read_csv(INPUT_TEST)
    if ATRIBUTOS == "TODOS":
        atributos = test_dataset.columns
    else:
        atributos = ATRIBUTOS
    test_dataset = test_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    categorical_features = []
    text_features = [TWEET_ATRIB]
    numerical_features = atributos_excepto(test_dataset.columns, [TARGET_NAME] + categorical_features + text_features)

    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(test_dataset, categorical_features, numerical_features, text_features)

    # Preprocesado del texto
    print("-- PREPROCESADO DE TEXTO")
    preprocesar_texto(train_dataset)
    preprocesar_texto(test_dataset)

    print(train_dataset.head(5))

    # Crear Bag Of Words or TFIDF
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
    tf = TfidfVectorizer(min_df=7, max_df=0.5, ngram_range=(1, 2), stop_words=stopwords.words('english'))
    lda = LatentDirichletAllocation(n_components = 10000, max_iter=5, learning_method='online',learning_offset=50., random_state=0)

    lda.fit(tf)

# making LDA TOP MATRIX USING CORPUS TF
    lda_topic_modelling = lda.fit_transform(tf)
    '''bow     = cv.fit_transform(train_dataset['wo_stopfreq_lem'])
    tfidf   = tf.fit_transform(train_dataset['wo_stopfreq_lem'])
    
    bow = cv.transform(test_dataset['wo_stopfreq_lem'])
    tfidf = tf.transform(test_dataset['wo_stopfreq_lem'])'''

    # Escalamos el texto -> NO CONSEGUIMOS MEJORES RESULTADOS
    # print("-- ESCALADO DE TEXTO")
    # scaler = MinMaxScaler()
    # tfidf = scaler.fit_transform(tfidf.toarray())

    # Creamos el dataframe después de aplicar todos los preprocesos necesarios
    if VECTORIZING == "BOW":
        dataframe = pd.DataFrame(bow.toarray())
    elif VECTORIZING == "TFIDF":
        dataframe = pd.DataFrame(tfidf.toarray()) #Si se escala hay que quitar el .toarray()

    # Añadimos los atributos seleccionados al dataset
    # . . .

    print("---- Dataset a clasificar:")
    print(dataframe.head(5))

    # Si la variable TARGET_NAME está en blanco, suponemos que no tenemos la clase objetivo en los datos
    # Por ello, solo realizaremos predicciones sin comprobar si son correctas o no
    if TARGET_NAME == "":
        # Suponemos que todos los atrobutos pertenecen al conjunto TestX
        print("---- Realizando predicciones sin TestY")
        testX = dataframe
        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)
        
        print("---- Resultados predicciones: ")
        print(predictions)
    
    else:
        print("---- Tratamos el TARGET: " + TARGET_NAME)    
        # Tratamos el Target attribute
        target_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        test_dataset['__target__'] = test_dataset[TARGET_NAME].map(str).map(target_map)
        test_dataset = test_dataset[~test_dataset['__target__'].isnull()]
        testY = test_dataset[['__target__']].copy() # Creamos testY con las clasificaciones luego
        testX = dataframe
        
        print(f"testX -> len: {len(testX.index)}")
        print(testX.head(5))
        print(f"testY -> len: {len(testY.index)}")
        print(testY.head(5))

        print("---- Realizando predicciones")
        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)
       
        predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
        cols = [
            u'probability_of_value_%s' % label
            for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
        ]
        
        report = classification_report(testY,predictions)
        testY['preds'] = predictions

        print("---- Resultados predicciones: ")
        print(testY)
        print(report)

if __name__ == "__main__":    
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt(argv[1:], 'ho:i:m:t:d:', ['help', 'output', 'input', 'model', 'target', 'train_dataset'])
        
    except getopt.GetoptError as err:
        # Error al parsear las opciones del comando
        print("ERROR: ", err)
        exit(1)

    print(options)
    # Registramos la configuración del script
    load_options(options)
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    main()
    

    
