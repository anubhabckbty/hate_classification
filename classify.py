import pandas as pd
import os
import elmo_model as em

#configuration
path = 'data/'
in_filename = 'tweets.csv'
out_filename = 'classified_tweets.csv'
#header_list = ['user_id', 'screen_name', 'tweet_id', 'tweet_text', 'rt_flag', 'rt_text'] 
header_list = ['user_id','screen_name','tweet_id','tweet_text','tweet_creation','tweet_fav','tweet_rt','rp_flag','rp_status','rp_user','qt_flag','qt_status_id','qt_user_id','qt_text','qt_creation','qt_fav','qt_rt','rt_flag','rt_status_id','rt_user_id','rt_text','rt_creation','rt_fav','rt_rt']
chunksize = 10**5 
model = em.get_model() #initializing classifier model

def ishateful(tweets):
    text = em.preprocess_text(tweets)
    pred = em.get_predictions(text, model, "Experiment4_hatespeech_weights.hdf5")
    return pred

def process(df, output_file):
    #preview the first 5 lines of the loaded data
    print(df.head())
    df['text'] = df['tweet_text'].fillna('') + df['rt_text'].fillna('')
    df['hateful'] = ishateful(df['text'])
    df.drop(labels='text', axis=1, inplace=True) #remove text column
    #output to file
    if os.path.isfile(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, mode='w', index=False)

#fileprocessing
output_file = path + out_filename
#existing output file is deleted to prevent repeated rows
if os.path.isfile(output_file):
    p = input("\nAn output file already exists, do you want to overwrite it?(Y/n) ")
    if p == 'Y' or p == 'y' or p == '':
        os.remove(output_file)
    else:
        print("Exiting program, please make a backup and try again.\n")
        exit()
for chunk in pd.read_csv(path + in_filename, usecols=header_list, chunksize=chunksize):
    process(chunk, output_file)
