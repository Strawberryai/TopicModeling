# Imports del script
from sys import exit, argv, version_info
from getopt import getopt
import os
import json

import re
import emoji
import string
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

import pickle

from sklearn.metrics                    import classification_report
from sklearn.preprocessing              import MinMaxScaler
from sklearn.model_selection            import train_test_split
from imblearn.over_sampling             import RandomOverSampler
from imblearn.under_sampling            import RandomUnderSampler
from sklearn.linear_model               import LogisticRegression
from sklearn.feature_extraction.text    import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes                import GaussianNB, MultinomialNB, BernoulliNB

# Train LDA model.
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from gensim.models import TfidfModel

# Variables globales
INPUT_FILE      = "./data/TweetsTrainDev.csv"               # Path del archivo de entrada
OUTPUT_PATH     = "./models"                                # Path de los archivos de salida
TARGET_NAME     = "airline_sentiment"                       # Nombre de la columna a clasificar
ATRIBUTOS       = ['text', 'airline_sentiment', 'airline']  # Atributos que seleccionamos del dataset | TODOS o lista
DEV_SIZE        = 0.2                                       # Indice del tamaño del dev. Por defecto un 20% de la muestra
RANDOM_STATE    = 42                                        # Seed del random split
MESSAGE         = ""                                        # Test message
AIRLINE         = "Delta"                                   # Aerolinea a filtrar
SENTIMIENTO     = "positive"                                # Sentimiento a filtrar


DEBUG           = True                                      # Flag para mostrar el archivo de debug con el dataset preprocesado
DEBUG_FILE      = "debug.csv"                               # Archivo que muestra el dataframe preprocesado

TWEET_ATRIB     = "text"                                    # Atributo de entrada de tweets

DEMOJI          = True                                      # Tener en cuenta los emojis transformándolos en texto
CLEANING        = True                                      # Limpiar textos: carácteres especiales, abrebiaturas, signos de puntuación...
STOP_WORDS      = False                                     # Tratar las stop words
FREQ_WORDS      = True                                      # Borramos las palabras más frecuentes. Pueden no aportar demasiada información.
LEMATIZE        = True                                      # Lematizamos el texto (realmente hacemos Stemming)
VECTORIZING     = "BOW"                                   # Sistema de vectorización: BOW | TFIDF

SAMPLING        = "NONE"                                    # Método de muestreo de nuestro dataset: OVERSAMPLING \ UNDERSAMPLING | NONE

# Downloads necesarios
stop_words = stopwords.words()

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: sentiment.py <optional-args>")
    print("The options supported by sentiment are:")
    print(f"-h, --help          show the usage")
    print(f"-i, --input         input file path of the data                     DEFAULT: ./{INPUT_FILE}")
    print(f"-o, --output        output file path for the weights                DEFAULT: ./{OUTPUT_PATH}")
    print(f"-t, --target        target name to predict                          DEFAULT: {TARGET_NAME}")
    print(f"-d, --debug         debug preprocess                                DEFAULT: {DEBUG}")
    print(f"-g, --debugfile     debug file                                      DEFAULT: ./{DEBUG_FILE}")
    print(f"-m, --message       test message                                    DEFAULT: {MESSAGE}")
    print(f"--airline           airline filter                                  DEFAULT: {AIRLINE}")
    print(f"--sentiment         sentiment filter                                DEFAULT: {SENTIMIENTO}")
    print("Text preprocessing:")
    print(f"-w                  tweet atribute                                  DEFAULT: {TWEET_ATRIB}")
    print(f"-e                  emoji to text                                   DEFAULT: {DEMOJI}")
    print(f"-c                  clean text                                      DEFAULT: {CLEANING}")
    print(f"-s                  remove stop words                               DEFAULT: {STOP_WORDS}")
    print(f"--stopw             remove stop words                               DEFAULT: {STOP_WORDS}")
    print(f"-f                  remove freq words                               DEFAULT: {FREQ_WORDS}")
    print(f"-l                  lematize text                                   DEFAULT: {LEMATIZE}")
    print(f"-v                  vectorizing function                            DEFAULT: {VECTORIZING}")
    print("Other preprocessing strategies:")
    print(f"-u                  sampling strategy                               DEFAULT: {SAMPLING}")
    print("")
    
    print(f"Example: sentiment.py")
    print(f"Example: entrenar.py -i input.csv -o models -a DecisionTree -z 6,5 -l 1")
    print(f"Example: entrenar.py -i iris.csv -t Especie --no-preprocesing -a DecisionTree -z 5")

    # Salimos del programa
    exit(1)

def load_options(options):
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_FILE, OUTPUT_PATH, TARGET_NAME, DEBUG, DEBUG_FILE, TWEET_ATRIB, DEMOJI, CLEANING, STOP_WORDS, FREQ_WORDS, LEMATIZE, VECTORIZING, SAMPLING, MESSAGE, AIRLINE, SENTIMIENTO

    for opt,arg in options:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-i', '--input'):
            INPUT_FILE = str(arg)
        elif opt in ('-o', '--output'):
            OUTPUT_PATH = str(arg)    
        elif opt in ('-t', '--target'):
            TARGET_NAME = str(arg)
        elif opt in ('-d', '--debug'):
            DEBUG = bool(arg)
        elif opt in ('-g', '--debugfile'):
            DEBUG_FILE = str(arg)
        elif opt in ('-m', '--message'):
            MESSAGE = str(arg)
        elif opt in ('--airline'):
            AIRLINE = str(arg)
        elif opt in ('--sentimiento'):
            SENTIMIENTO = str(arg)

        elif opt == '-w':
            TWEET_ATRIB = str(arg)
        elif opt == "-e":
            DEMOJI = bool(arg) 
        elif opt == "-c":
            CLEANING = False 
        elif opt == "-s":
            STOP_WORDS = bool(arg) 
            print('prueba')
        elif opt == "--stopw":
            STOP_WORDS = True
        elif opt == "-f":
            FREQ_WORDS = bool(arg) 
        elif opt == "-l":
            LEMATIZE = bool(arg) 
        elif opt == "-v":
            VECTORIZING = str(arg)
            if VECTORIZING not in ("BOW", "TFIDF"):
                print(f"ERROR: Vectorizing value {VECTORIZING} not known")
        
        elif opt == "-u":
            SAMPLING = str(arg)
            if SAMPLING not in ("OVERSAMPLING", "UNDERSAMPLING", "NONE"):
                print(f"ERROR: Sampling value {SAMPLING} not known")

def show_script_options():
    # PRE: ---
    # POST: imprimimos las configuración del script
    print("sentiment.py configuration:")
    print(f"-i                  input file path             -> {INPUT_FILE}")
    print(f"-o                  output models path          -> {OUTPUT_PATH}")
    print(f"-t                  target name                 -> {TARGET_NAME}")
    print(f"                    selected atributes          -> {ATRIBUTOS}")
    print(f"                    dev size                    -> {DEV_SIZE}")
    print(f"                    ramdom state seed           -> {RANDOM_STATE}")
    print(f"-d                  debug preprocess            -> {DEBUG}")
    print(f"-g                  debug file                  -> {DEBUG_FILE}")
    print(f"-m, --message       test message                -> {MESSAGE}")
    print(f"--airline           airline filter              -> {AIRLINE}")
    print(f"--sentiment         sentiment filter            -> {SENTIMIENTO}")
    print("Text preprocessing:")
    print(f"-w                  tweet atribute              -> {TWEET_ATRIB}")
    print(f"-e                  emoji to text               -> {DEMOJI}")
    print(f"-c                  clean text                  -> {CLEANING}")
    print(f"-s                  remove stop words           -> {STOP_WORDS}")
    print(f"-f                  remove freq words           -> {FREQ_WORDS}")
    print(f"-l                  lematize text               -> {LEMATIZE}")
    print(f"-v                  vectorizing function        -> {VECTORIZING}")
    print("Other preprocessing strategies:")
    print(f"-u                  sampling strategy           -> {SAMPLING}")
    print()

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

def crear_directorio_modelos():
    # PRE: ---
    # POST: Crea el directorio de modelos. Si ya existe, borra su contenido
    
    # Si no existe, creamos el directorio
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Borramos el contenido del directorio
    for filename in os.listdir(OUTPUT_PATH):
        file_path = os.path.join(OUTPUT_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def guardar_modelo(clf, nombre):
    file_path = os.path.join(OUTPUT_PATH, nombre + ".sav")
    saved_model = pickle.dump(clf, open(file_path,'wb')) 
    print(f'Modelo {nombre} guardado correctamente')

def atributos_excepto(atributos, excepciones):
    # PRE: lista completa de atributos y lista de aquellos que no queremos seleccionar
    # POST: devolvemos una lista de atributos
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)

    return atribs

def guardar_resultadosLDA(configuracion, resultado, topic_coherence):
    #crear carpeta para guardar resultados 
    if not os.path.exists('results'):
        os.makedirs('results')

    num_topics = configuracion['n_topics']
    chunksize = configuracion['chunksize']
    passes = configuracion['passes']
    iterations = configuracion['iterations']
    alpha = configuracion['alpha']

    filename = "results/"+AIRLINE+"_"+SENTIMIENTO+"_t_"+str(num_topics)+"_ch_"+str(chunksize)+"_p_"+str(passes)+"_it_"+str(iterations)+"_a_"+str(alpha)+"_"+VECTORIZING+".txt"
    with open(filename, "w") as file:

        # Escribir los resultados en el archivo de texto
        file.write(f"\nNumero de topicos \t: {num_topics}\n")
        file.write(f"chunksize \t: {chunksize}\n")
        file.write(f"passes \t: {passes}\n")
        file.write(f"iterations \t: {iterations}\n")
        file.write(f"alpha \t: {alpha}\n")
        file.write(f"airline \t: {AIRLINE}\n")
        file.write(f"sentiment \t: {SENTIMIENTO}\n")
        file.write(f"menssage \t: {MESSAGE}\n")
        file.write("\n")
        file.write(f"topic coherence \t: {topic_coherence}\n")
        file.write("\n")
        for topic in resultado:
            #no se que es este numero
            file.write(str(topic[1]))
            #palabras del topic
            file.write("\n Palabras del topic: \n")
            palabras = []
            palabras = topic[0]
            for palabra in palabras:
                file.write("Palabra "+str(palabra[1])+" \t %"+ str(palabra[0])+"\n")
            file.write("\n")

    #guardar registro de las pruebas realizadas
    with open("pruebasRealizadas.txt", "a") as archivo:
        archivo.write(f"Numero de topicos \t: {num_topics}\n")
        archivo.write(f"chunksize \t: {chunksize}\n")
        archivo.write(f"passes \t: {passes}\n")
        archivo.write(f"iterations \t: {iterations}\n")
        archivo.write(f"alpha \t: {alpha}\n")
        archivo.write("Vectorizing \t: "+VECTORIZING+"\n")
        archivo.write("------------------------------------------ \n")
        

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

#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main():
    # Entrada principal del programa
    print("-- INICIANDO MAIN")
    crear_directorio_modelos()

    # Cargamos los datos de entrada y seleccionamos los atributos
    ml_dataset = pd.read_csv(INPUT_FILE)
    if ATRIBUTOS == "TODOS":
        atributos = ml_dataset.columns
    else:
        atributos = ATRIBUTOS
    ml_dataset = ml_dataset[atributos]

    # FILTRADO
    print("--- FILTRADO")
    ml_dataset = ml_dataset[ml_dataset['airline'] == AIRLINE]
    ml_dataset = ml_dataset[ml_dataset['airline_sentiment'] == SENTIMIENTO]
    ml_dataset = ml_dataset.drop('airline', axis=1)
    print(ml_dataset.head(5))

    categorical_features = []
    text_features = [TWEET_ATRIB]
    numerical_features = atributos_excepto(ml_dataset.columns, [TARGET_NAME] + categorical_features + text_features)

    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)

    # Tratamos el Target attribute
    #target_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    #ml_dataset['__target__'] = ml_dataset[TARGET_NAME].map(str).map(target_map)
    #ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    #del ml_dataset[TARGET_NAME]

    # Preprocesado del texto
    print("-- PREPROCESADO DE TEXTO")
    # Tratamos los emojis
    if DEMOJI:
        ml_dataset['demoji'] = ml_dataset[TWEET_ATRIB].apply(lambda text: emoji.demojize(text))
    else:
        ml_dataset['demoji'] = ml_dataset[TWEET_ATRIB]

    # Limpiamos el texto
    if CLEANING:
        ml_dataset['cleaned'] = ml_dataset['demoji'].apply(cleaning) # Limpiamos
    else:
        ml_dataset['cleaned'] = ml_dataset['demoji']

    # Tratamos las stop words
    if STOP_WORDS:
        stop_words = stopwords.words()
        ml_dataset['no_sw'] = ml_dataset['cleaned'].apply(lambda text: remove_stopwords(text, stop_words)) # Eliminamos las stop words
    else:
        ml_dataset['no_sw'] = ml_dataset['cleaned'] # sin borrar stop words

    # Tratamos las palabras más frecuentes
    if FREQ_WORDS:
        freq_words = get_freqwords(ml_dataset['no_sw'])
        ml_dataset["wo_stopfreq"] = ml_dataset["no_sw"].apply(lambda text: remove_freqwords(text, freq_words)) # Eliminamos las más frecuentes
    else: 
        ml_dataset["wo_stopfreq"] = ml_dataset["no_sw"]

    # Stemming da mejores resultados. Dejamos Lematización comentado
    if LEMATIZE:
        wordnet_lem = WordNetLemmatizer()
        ml_dataset['wo_stopfreq_lem'] = ml_dataset['wo_stopfreq'].apply(wordnet_lem.lemmatize) # Lematizamos las palabras
        
        #porter = PorterStemmer()
        #ml_dataset["wo_stopfreq_lem"] = ml_dataset["wo_stopfreq"].apply(porter.stem)

    else:
        ml_dataset['wo_stopfreq_lem'] = ml_dataset['wo_stopfreq']

    #TODO AÑADIR LO DE LOS N-GRAMAS ?

    # Para ver el preprocesado de texto
    if DEBUG:
        ml_dataset.to_csv(DEBUG_FILE, index = True)

    # Crear Bag Of Words or TFIDF
    #token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    #cv = CountVectorizer(stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
    #tf = TfidfVectorizer(min_df=7, max_df=0.5, ngram_range=(1, 2), stop_words=stopwords.words('english'))
    tf = TfidfVectorizer(stop_words=stopwords.words('english'))

    #bow     = cv.fit_transform(ml_dataset['wo_stopfreq_lem'])
    tfidf   = tf.fit_transform(ml_dataset['wo_stopfreq_lem'])

    # Escalamos el texto -> NO CONSEGUIMOS MEJORES RESULTADOS
    # print("-- ESCALADO DE TEXTO")
    # scaler = MinMaxScaler()
    # tfidf = scaler.fit_transform(tfidf.toarray())

    # Creamos el dataframe después de aplicar todos los preprocesos necesarios
    if VECTORIZING == "BOW":
        # Tokenize the documents.
        

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        for row in ml_dataset.itertuples():
            ml_dataset['wo_stopfreq_lem'][row.Index] = ml_dataset['wo_stopfreq_lem'][row.Index].lower()  # Convert to lowercase.
            ml_dataset['wo_stopfreq_lem'][row.Index] = tokenizer.tokenize(ml_dataset['wo_stopfreq_lem'][row.Index])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        ml_dataset['wo_stopfreq_lem'] = [[token for token in doc if not token.isnumeric()] for doc in ml_dataset['wo_stopfreq_lem']]

        # Remove words that are only one character.
        ml_dataset['wo_stopfreq_lem'] = [[token for token in doc if len(token) > 1] for doc in ml_dataset['wo_stopfreq_lem']]

        #dataframe = pd.DataFrame(bow.toarray())
        #Remove rare and common tokens.
        from gensim.corpora import Dictionary

        # Create a dictionary representation of the documents.
        dictionary = Dictionary(ml_dataset['wo_stopfreq_lem'])

        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        dictionary.filter_extremes(no_below=20, no_above=0.80)

        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in ml_dataset['wo_stopfreq_lem']]
        # Make an index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

    # Añadimos los atributos seleccionados al dataset
    #dataframe['__target__'] = ml_dataset['__target__']

    # Entrenando modelos
    print("-- TRAINING MODELS")
    

    ## gensim LDA model

    #parametros para el training
    # Set training parameters.
    num_topics = 5
    chunksize = 80000
    passes = 50
    iterations = 700
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    alpha=0.00001


    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=alpha,
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    configuracion = {'n_topics': num_topics,
                     'chunksize': chunksize,
                     'passes':passes,
                     'iterations':iterations,
                     'alpha':alpha
                     }

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)
    guardar_resultadosLDA(configuracion=configuracion, resultado=top_topics, topic_coherence=avg_topic_coherence)

if __name__ == "__main__":
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt(argv[1:], 'h:i:o:t:d:g:w:e:c:s:f:l:v:u:m', ['help', 'input', 'output', 'target', 'debug', 'debugfile', 'message=', 'airline=', 'sentimiento=', 'stopw'])
        
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