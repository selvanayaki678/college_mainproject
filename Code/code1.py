#!/usr/bin/env python
# coding: utf-8

# In[12]:


from __future__ import print_function
import nltk
import numpy as np
import pandas as pd
import random
import pprint, time
import re
import os.path
from datetime import date
from datetime import datetime as dt, timedelta
today = date.today()
import os
from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer
from google_auth_oauthlib.flow import InstalledAppFlow
scopes = ['https://www.googleapis.com/auth/calendar']
flow = InstalledAppFlow.from_client_secrets_file("D:\secret_file.json", scopes=scopes)
credentials=flow.run_console()
import pickle
pickle.dump(credentials, open("token.pkl", "wb"))
credentials = pickle.load(open("token.pkl", "rb"))
from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
service = build("calendar", "v3", credentials=credentials)
result = service.calendarList().list().execute()
import datefinder
matches = datefinder.find_dates("5 may 9 PM")
from datetime import datetime, timedelta


# In[13]:


def create_event(start_time_str,summary,description,location,duration=1):
    matches=list(datefinder.find_dates(start_time_str))
    if len(matches):
        start_time=matches[0]
    end_time=start_time+timedelta(hours=duration)

    event={
    'summary':summary,
    'location':location,
    'description':description,
    'start':{
    'dateTime':start_time.strftime("%Y-%m-%dT%H:%M:%S"),
    'timeZone':'Asia/Kolkata',
    },
    'end':{
    'dateTime':end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    'timeZone':'Asia/Kolkata',
    },
    'reminders':{
    'useDefault':False,
    'overrides':[
    {'method':'email','minutes':24*60},
    {'method':'popup','minutes':10},
    ],
    },
    }
    return service.events().insert(calendarId='primary',body=event).execute()


# In[14]:


def create_event(start_time_str,summary,description,location,duration=1):
    matches=list(datefinder.find_dates(start_time_str))
    print(matches)
    if len(matches):
        start_time=matches[0]
    end_time=start_time+timedelta(hours=duration)

    event={
    'summary':summary,
    'location':location,
    'description':description,
    'start':{
    'dateTime':start_time.strftime("%Y-%m-%dT%H:%M:%S"),
    'timeZone':'Asia/Kolkata',
    },
    'end':{
    'dateTime':end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    'timeZone':'Asia/Kolkata',
    },
    'reminders':{
    'useDefault':False,
    'overrides':[
    {'method':'email','minutes':24*60},
    {'method':'popup','minutes':10},
    ],
    },
    }
    return service.events().insert(calendarId='primary',body=event).execute()


# In[15]:


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        # iterate through all file
        text = f.read()
        print(text)
        patterns = [
            [r'(\d+|(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                  eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                  ninety|hundred|thousand))[-](year|day|week|month)[-](after|later)', 'DATE'],  # datename_extract
            [r'(today|tomorrow|tonight|tonite)$', 'DATE'],  # datename_extract
            [r'(coimbatore|chennai|madurai|dindigul|Hyderabad|mumbai|pune|bangalore|kochi|delhi|kolkata)$', 'LOCATION'],
            # location extract
            [r'(interview|party|meeting|conference|registration)$', 'EVENT'],  # event_extract
            [r'(12|1|2|3|4|5|6|7|8|9)(AM|PM)$', 'TIME'],  # Time extract
            [r'(this|next)[-](year|day|week|month|weekend)$', 'DATE'],  # date_extract
            [r'\d{4}[-/]\d{2}[-/]\d{2}', 'DATE'],  # date_extract
            [r'\d{2}/\d{2}/\d{4}', 'DATE'],  # date_extract
            [r'.*ed$', 'VERB'],  # past tense
            [r'(The|the|A|a|An|an)$', 'ARTICLES'],  # articles
            [r'.*es$', 'VERB'],  # verb
            [r'.*ly$', 'ADVERBS'],  # adverbs
            [r'.*\'s$', 'NOUN'],  # possessive nouns
            [r'.*s$', 'NOUN'],  # plural nouns
            [r'\*T?\*?-[0-9]+$', 'X'],  # X
            [r'^-?[0-9]+(.[0-9]+)?$', 'NUM'],  # cardinal numbers
            [r'.*', 'NOUN'],  # nouns
            [r'.*ing$', 'VERB'],  # gerund
        ]

        date_patterns = [
            [r'(today|tomorrow|tonight|tonite)$', 'DATE1'],
            [r'(this|next)[-](year|day|week|month|weekend)$', 'DATE2'],
            [r'(\d+|(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                        eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                        ninety|hundred|thousand))[-](year|day|week|month)[-](after|later)', 'DATE3'],
        ]

        """ Sentence Tokenizing to find the temporal context containing sentence """
        sent_list = sent_tokenize(text)
        print('Sentence tokens', sent_list,'\n')
        sent_dict = {}
        for i in range(0, len(sent_list)):
            event_list = []
            date_list = []
            time_list = []

            tokenizer = MWETokenizer([('this', 'week'), ('this', 'weekend'), ('3', 'year', 'later')], separator='-')
            '''tokenizer.add_mwe(('by', 'tomorrow'))'''
            res = tokenizer.tokenize(sent_list[i].split())

            sentence = ' '.join(map(str, res))
            reg_tagger = RegexpTagger(patterns)
            reg_out = reg_tagger.tag(word_tokenize(sentence))

            tokenized = dict(reg_out)

            ''' grouping keys and values in dictionary'''
            group_tokens = {}
            for key, value in tokenized.items():
                if value in group_tokens:
                    group_tokens[value].append(key)
                else:
                    group_tokens[value] = [key]
            print('\nWord Tokens', group_tokens)

            x = group_tokens.get("EVENT")
            y = group_tokens.get("DATE")
            z = group_tokens.get("TIME")
            print('x y z', x, y, z)
            if x != None:
                for i in range(len(x)):
                    event_list.append(x[i])
            if y != None:
                for i in range(len(y)):
                    date_list.append(y[i])
            if z != None:
                for i in range(len(z)):
                    time_list.append(z[i])

            print('Event,Date,Time :', event_list, date_list, time_list)


            if len(date_list) != 0 and len(event_list) != 0:
                '''Map event to Calender'''

                reg_tagger = RegexpTagger(date_patterns)
                reg_out = reg_tagger.tag(date_list)

                date_format_tokens = dict(reg_out)
                print('Date_format_tokens',date_format_tokens)
                print('')

                final_tokens = {}
                for key, value in date_format_tokens.items():
                    if value in final_tokens:
                        final_tokens[value].append(key)
                    else:
                        final_tokens[value] = [key]
                print('Final_tokens',final_tokens)

                single_val = final_tokens.get('DATE1')
                if single_val is not None:
                    single_date_val= list(single_val)
                    print('val', single_date_val)

                    
                    Date = ''.join(map(str, date_list))
                    if time_list is None:
                        Time= '5 pm'
                    else:
                        Time = ''.join(map(str, time_list)) 
                    for i in range(len(single_date_val)):
                        if single_date_val[i] == 'tomorrow':
                            print("Date calculation")
                            date = today +timedelta(days=1)
                            print('Tomorrow', date)
                        if single_date_val[i] == 'today' or 'tonight' or 'tonite':
                            print("Date calculation")
                            date = today
                            print('single_date_val[i]', date)
                    date=Date+Time
                    print(date)
                    x = datetime.datetime(date) 
                    print(x.strftime("%b %d %Y %H:%M:%S")) 
                create_event(date, event_list, "Publish the paper", "chennai")


# In[16]:


if __name__=="__main__":
    # Folder Path
    path = r"C:\Users\User\Desktop\Main Project\conversations"
    # Change the directory
    os.chdir(path)
    # Read text File
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # call read text file function
            read_text_file(file_path)


# In[ ]:





# In[ ]:




