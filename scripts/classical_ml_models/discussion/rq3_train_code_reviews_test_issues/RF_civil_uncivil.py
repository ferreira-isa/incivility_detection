import numpy as np
import pandas as pd
import xlrd as xl
from nltk.tokenize import word_tokenize
from pandas import ExcelWriter
from pandas import ExcelFile
import pprint
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import re
import pickle
from operator import itemgetter
import time, datetime
from functools import partial, update_wrapper
from copy import deepcopy
from sklearn.metrics import matthews_corrcoef
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from imblearn.pipeline import Pipeline as Imb_Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, make_scorer, confusion_matrix

pp = pprint.PrettyPrinter(indent=4)
import nltk
nltk.download('punkt')

# EDA
import random
from random import shuffle
random.seed(1)
import re
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
import pandas as pd

## Ignore warnings
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
np.warnings.filterwarnings('ignore')

"""#### Code to tokenize:"""
def tokenize (text):
	return word_tokenize(text)

"""### Read incivility data"""
def read_csv(filename, file_type, dataset):
	df = pd.read_csv(filename, encoding='utf-8', sep=',', index_col=0)

	if dataset == "lkml":
	# Deleting columns not useful for the model
		if (file_type == "technical") or (file_type == "non_technical"):
			del df ["original_text"]
			del df ["author_name"]
			del df ["author_email"]
			del df ["author_role"]
			df['email_classification'] = [0 if x=='not_technical' else 1 for x in df['email_classification']]
			df.rename(columns={"ahlaam_preprocessed_text": "text_content"}, inplace=True)
			df.rename(columns={"email_classification": "label"}, inplace=True)
		elif (file_type == "civil") or (file_type == "uncivil"):
			del df ["author_name"]
			del df ["author_email"]
			del df ["author_role"]
			del df["email_final_code"]
			df['quotation_code'] = [0 if x=='uncivil' else 1 for x in df['quotation_code']]
			df.rename(columns={"quotation": "text_content"}, inplace=True)
			df.rename(columns={"quotation_code": "label"}, inplace=True)

		#df['author_role'] = [0 if x=='developer' else 1 for x in df['author_role']]
		df['is_first_author_thread'] = [0 if x==False else 1 for x in df['is_first_author_thread']]
	elif dataset == "gh":
		if (file_type == "technical") or (file_type == "non_technical"):
			del df ["original_text"]
			del df ["author_name"]
			df['email_classification'] = [0 if x=='not_technical' else 1 for x in df['email_classification']]
			df.rename(columns={"ahlaam_preprocessed_text": "text_content"}, inplace=True)
			df.rename(columns={"email_classification": "label"}, inplace=True)
		elif (file_type == "civil") or (file_type == "uncivil"):
			del df ["author_name"]
			del df ["issue_final_code"]
			df['quotation_code'] = [0 if x=='uncivil' else 1 for x in df['quotation_code']]
			df.rename(columns={"quotation": "text_content"}, inplace=True)
			df.rename(columns={"quotation_code": "label"}, inplace=True)

		df['is_first_author_thread'] = [0 if x==False else 1 for x in df['is_first_author_thread']]

	return df

print("Reading datasets...")
civil_dataset = "../../data/lkml/3_features_dataset/civil.csv"
uncivil_dataset = "../../data/lkml/3_features_dataset/uncivil.csv"

civil_data = read_csv(civil_dataset, "civil", "lkml")
uncivil_data = read_csv(uncivil_dataset, "uncivil", "lkml")

dataframe = pd.concat([civil_data, uncivil_data], ignore_index=True)

X = dataframe.reset_index(drop=True)
del X["thread_id"]
del X["email_id"]
del X["quotation_tbdf"]
y = dataframe["label"].reset_index(drop=True)

gh_civil_dataset = "../../data/gh/3_features_dataset/civil.csv"
gh_uncivil_dataset = "../../data/gh/3_features_dataset/uncivil.csv"

civil_data_gh = read_csv(gh_civil_dataset, "civil", "gh")
uncivil_data_gh = read_csv(gh_uncivil_dataset, "uncivil", "gh")

dataframe_gh = pd.concat([civil_data_gh, uncivil_data_gh], ignore_index=True)

X_gh = dataframe_gh.reset_index(drop=True)
del X_gh["issue_id"]
del X_gh["comment_id"]
del X_gh["quotation_tbdf"]
y_gh = dataframe_gh["label"].reset_index(drop=True)

"""# Easy Data Augmentation (EDA)"""

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
def get_only_chars(line):
    clean_line = ""
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm.?!.;:\'\*" ':
            clean_line += char
        else:
            clean_line += ' '
    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')
	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, p):
	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
# Augmentation arguments:

# input_file = input file of unaugmented data
# output_file = output file of unaugmented data
# n_aug = number of augmented sentences per original sentence (default 9)
# alpha_sr = % of words in each sentence to be replaced by synonyms (default 0.1)
# alpha_ri = % of words in each sentence to be inserted (default 0.1)
# alpha_rs = % of words in each sentence to be swapped (default 0.1)
# alpha_rd = % of words in each sentence to be deleted (default 0.1)
########################################################################
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word != '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences

def gen_eda(orig_df_train, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
	output_train_df = pd.DataFrame(columns=["text_content", "label", "is_first_author_thread", "nr_characters", "ratio_words_email_thread", "ratio_words_email_comment", "position_sentence_comment", "position_sentence_thread", "is_last_comment", "time_start_to_email", "time_email_to_end", "time_previous_to_email", "time_email_to_next"])

	for idx in orig_df_train.index:
		label = orig_df_train.loc[idx, 'label']
		sentence = orig_df_train.loc[idx, 'text_content']
		# author_role = orig_df_train.loc[idx, 'author_role']
		is_first_author_thread = orig_df_train.loc[idx, 'is_first_author_thread']
		nr_characters = orig_df_train.loc[idx, 'nr_characters']
		ratio_words_email_thread = orig_df_train.loc[idx, 'ratio_words_email_thread']
		ratio_words_email_comment = orig_df_train.loc[idx, 'ratio_words_email_comment']
		position_sentence_comment = orig_df_train.loc[idx, 'position_sentence_comment']
		position_sentence_thread = orig_df_train.loc[idx, 'position_sentence_thread']
		is_last_comment = orig_df_train.loc[idx, 'is_last_comment']
		time_start_to_email = orig_df_train.loc[idx, 'time_start_to_email']
		time_email_to_end = orig_df_train.loc[idx, 'time_email_to_end']
		time_previous_to_email = orig_df_train.loc[idx, 'time_previous_to_email']
		time_email_to_next = orig_df_train.loc[idx, 'time_email_to_next']

		aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
		d = {}
		d["text_content"] = aug_sentences
		d["label"] = [label for _ in range(len(aug_sentences))]
		# d["author_role"] = [author_role for _ in range(len(aug_sentences))]
		d["is_first_author_thread"] = [is_first_author_thread for _ in range(len(aug_sentences))]
		d["nr_characters"] = [nr_characters for _ in range(len(aug_sentences))]
		d["ratio_words_email_thread"] = [ratio_words_email_thread for _ in range(len(aug_sentences))]
		d["ratio_words_email_comment"] = [ratio_words_email_comment for _ in range(len(aug_sentences))]
		d["position_sentence_comment"] = [position_sentence_comment for _ in range(len(aug_sentences))]
		d["position_sentence_thread"] = [position_sentence_thread for _ in range(len(aug_sentences))]
		d["is_last_comment"] = [is_last_comment for _ in range(len(aug_sentences))]
		d["time_start_to_email"] = [time_start_to_email for _ in range(len(aug_sentences))]
		d["time_email_to_end"] = [time_email_to_end for _ in range(len(aug_sentences))]
		d["time_previous_to_email"] = [time_previous_to_email for _ in range(len(aug_sentences))]
		d["time_email_to_next"] = [time_email_to_next for _ in range(len(aug_sentences))]
		df = pd.DataFrame(d)
		output_train_df = pd.concat([output_train_df, df], ignore_index=True)
    
		# print("Generated", num_aug, "sentences per input sentence with EDA")
		# print("output_train_df: ", len(output_train_df))
	return output_train_df

"""# Nested Cross-Validation"""

# To be used within GridSearch
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# To be used in outer CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

non_transform_features = [col for col in X.columns.values if col not in ['text_content', 'label']]

"""## Define pandas-compatible feature unificator

This is just a pandas compatible feature unificator.

This is necessary because we have two types of features: textual + conversational

Hence, we first use tf-idf to vectorize the textual content and then append the conversational features to this vector using the feature unificator.
"""
class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())
        if not result:
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

class PandasTransform(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None, copy=None):
        return X.loc[:, self.columns].astype(float)

"""## Define Pipelines:
1. Pipeline1: Textual + Conversational + RandomOverSampling
2. Pipeline2: Textual + Conversational + RandomUnderrSampling
3. Pipeline3: Textual + Conversational + SMOTE
"""

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, x, y=None):
        return self
    def transform(self, data_array):
        return data_array[:, self.columns]

### Simply select the conversational features, and tfidf-vectorize the textual content
ling_features = PandasTransform(non_transform_features)
tfidf_text_features = Pipeline([('extract_field',FunctionTransformer(lambda x: x['text_content'],validate=False)),('vect',TfidfVectorizer(tokenizer=tokenize))])

feature_union1 = PandasFeatureUnion([
    ('selector',ling_features),
    ('text_vectorizer',tfidf_text_features)
])

### Combine the two types of feature vectors
pipeline1 = Imb_Pipeline([
    ('features', feature_union1),
    ('RandomOverSampler', RandomOverSampler(sampling_strategy='minority')),
    ('clf', RandomForestClassifier())
])
pipeline2 = Imb_Pipeline([
    ('features', feature_union1),
    ('RandomUnderSampler', RandomUnderSampler(sampling_strategy='majority')),
    ('clf', RandomForestClassifier())
])

pipeline3 = Imb_Pipeline([
    ('features',feature_union1),
    ('smote', SMOTE()),
    ('clf', RandomForestClassifier()),
])

### Hyperparameters to search
parameters = {
    'features__text_vectorizer__vect__ngram_range': [(1,2)],  # unigrams or bigrams or trigrams
    'clf__n_estimators': [50],
    'clf__min_samples_split': [2],
}

### Define and create the scoring functions
def compute_mcc (y_true, y_pred):
  return matthews_corrcoef(y_true, y_pred)

### Perform Nested cross-validation and EDA on Pipelines
def run_pipelines (pipeline, parameters, pipeline_id):
	start = time.time()

	outer_f1 = list()
	outer_precision = list()
	outer_recall = list()
	outer_mcc = list()
	results = list()

	EDA_parameters = [{'alpha_sr': 0.05, 'alpha_ri': 0.05, 'alpha_rs': 0.05, 'alpha_rd': 0.05, 'num_aug': 8}]

	with open ('output/RF_civil_uncivil_pipeline'+pipeline_id+'.csv', "a+") as f:
		f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dataset", "data", "model", "thread_id_email_id", "issue_id", "comment_id", "label_correct", "label_predicted", "quotation_tbdf"))
		for parameter in EDA_parameters:
			
			X_train_augmented = gen_eda(X, parameter["alpha_sr"], parameter["alpha_ri"], parameter["alpha_rs"], parameter["alpha_rd"], parameter["num_aug"])  

			y_train_augmented = X_train_augmented["label"].astype(str).astype(int).to_list()
			X_train_augmented = X_train_augmented.drop(['label'], axis=1)

			clf1 = GridSearchCV(pipeline, parameters, cv=None)
			result = clf1.fit(X_train_augmented, y_train_augmented)
			best_model = result.best_estimator_
			best_model_score = result.best_score_
			best_model_params = result.best_params_
			yhat = best_model.predict(X_gh)

			for index, row in y_gh.iteritems():
				correct_label = "uncivil" if row==0 else "civil"
				predicted_label = "uncivil" if yhat[index]==0 else "civil"
				thread_id_email_id = str(dataframe_gh.loc[index, "issue_id"]) + "_" + str(dataframe_gh.loc[index, "comment_id"])

				f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("rq3_train_lkml_test_gh_civil_uncivil", "Train on code reviews and test on issues", "RF", thread_id_email_id, dataframe_gh.loc[index, "issue_id"], dataframe_gh.loc[index, "comment_id"], correct_label, predicted_label, dataframe_gh.loc[index, "quotation_tbdf"]))

		print(">>>>>>> Completed Pipeline " + pipeline_id + " scenario in "+ str(datetime.timedelta(seconds=(time.time()-start))))

"""### Run pipeline1"""
#run_pipelines (pipeline1, parameters, "1")

"""### Run pipeline2"""
run_pipelines (pipeline2, parameters, "2")

"""### Run pipelin3"""
#run_pipelines (pipeline3, parameters, "3")
