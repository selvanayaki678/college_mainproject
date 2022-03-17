import nltk
import numpy as np
import pandas as pd
import pprint, time
import re
import datetime
import datefinder
import pickle

from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer
#from google_auth_oauthlib.flow import InstalledAppFlow

nltk.download('treebank')
nltk.download('universal_tagset')

'''
scopes = ['https://www.googleapis.com/auth/calendar']
flow = InstalledAppFlow.from_client_secrets_file("D:\secret_file.json", scopes=scopes)
credentials = flow.run_console()
pickle.dump(credentials, open("token.pkl", "wb"))
credentials = pickle.load(open("token.pkl", "rb"))

from future import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

service = build("calendar", "v3", credentials=credentials)
result = service.calendarList().list().execute()
matches = datefinder.find_dates("5 may 9 PM")
list(matches)


def create_event(start_time_str, summary, description, location, duration=1):
    matches = list(datefinder.find_dates(start_time_str))
    if len(matches):
        start_time = matches[0]
    end_time = start_time + timedelta(hours=duration)

    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'timeZone': 'Asia/Kolkata',
        },
        'end': {
            'dateTime': end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'timeZone': 'Asia/Kolkata',
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }
    return service.events().insert(calendarId='primary', body=event).execute()

'''
#sentence = "John: Hello, Bob!. Bob: Hi, John!. John: Are you free this week ?. Bob: No, I think so, since i have to attend a party tomorrow 5PM. John: i want you to come for a meeting on 20/03/2021."

sentence = "John: Hello, Bob!. Bob: Hi, John!. John: Are you free this week ?. Bob: No, I think so, since i have to attend a party 22/02/2021 tomorrow 5PM. John: Sure. Great!"
patterns = [
    [r'.*ing$', 'VERB'],  # gerund
    [r'(\d+|(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand))[-](year|day|week|month)[-](after|later)', 'DATE1'],  # datename_extract
    [r'(today|tomorrow|tonight|tonite)$', 'DATE'],  # datename_extract
    [r'(coimbatore|chennai|madurai|dindigul|Hyderabad|mumbai|pune|bangalore|kochi|delhi|kolkata)$', 'LOCATION'],
    # location extract
    [r'(interview|party|meeting|conference)$', 'EVENT'],  # event_extract
    [r'(12|1|2|3|4|5|6|7|8|9)(AM|PM)$', 'TIME'],  # Time extract
    [r'(this|next)[-](year|day|week|month|weekend)$', 'DATE1'],  # date_extract
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
]
event_list = []
date_list = []
time_list = []
sent_list = sent_tokenize(sentence)
print('Sentence tokenized', sent_list)
sent_dict = {}
# for j in range()

for i in range(0, len(sent_list)):

    '''sent_list[i] = re.sub(r"[,.:@\'?\.$%_]", "", sent_list[i], flags=re.I)
    print(sent_list[i])'''

    tokenizer = MWETokenizer([('this', 'week'), ('this', 'weekend'), ('3', 'year', 'later')], separator='-')
    #tokenizer.add_mwe(('by', 'tomorrow'))
    res = tokenizer.tokenize(sent_list[i].split())
    print('res',res)

    sentence1 = ' '.join(map(str, res))
    #print(sentence1)
    reg_tagger = RegexpTagger(patterns)
    reg_out = reg_tagger.tag(word_tokenize(sentence1))
    print(list(reg_out))

    tokenized = dict(reg_out)
    # print("token")
    print(tokenized)
    new_tokenized = {}
    for key, value in tokenized.items():
        if value in new_tokenized:
            new_tokenized[value].append(key)
        else:
            new_tokenized[value] = [key]
    print('Tokenized one', new_tokenized)

    key1 = 'DATE1' or 'DATE2'
    for key, value in group_tokens.items():
        if key1 in group_tokens:
            sent_dict[sent_list[i]] = group_tokens[key]
        # else:
        #   sent_dict[sent_list[i]] ='FALSE'


    # y='EVENT'
    # print("Events")
    x = new_tokenized.get("EVENT")
    print('x',x)
    if x != None:
        event_list.append(x[0])
        # print(x)
    print(DATE)
    y = new_tokenized.get("DATE")
    # print(y)
    if y != None:
        date_list.append(y)
    z = new_tokenized.get("TIME")
    if z != None:
        time_list.append(z)
    print(time_list)

    time_list = str(time_list)
    event_list = str(event_list)
    print('e',event_list)
    print('d',date_list)

    '''
    to = str(date_list[0])
    print(to)
    date_time = time_list + to
    print('qq',date_time)
    if to == "tomorrow":
        print("date calculation")
        date = today + datetime.timedelta(days=1)
        print(date)
    '''

for key, value in sent_dict.items():
    print(key, '::::::::', value)
    sentence1 = key
'''

print('date', date_time)
print('event', event_list)

create_event(date_time, event_list, "Publish the paper", "chennai")
'''
'''
import nltk

text = "John: Hello, Bob!. Bob: Hi, John!. John: Are you free this weekend ?. Bob: No, I think so, since i have to attend a party tomorrow. John: Sure. Great!"

tokens = nltk.word_tokenize(text)
print(tokens)

tag = nltk.pos_tag(tokens)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(grammar)
result = cp.parse(tag)
print(result)
result.draw()

'''
'''
import nltk
import numpy as np
import pandas as pd
import random
import pprint, timeh

from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer

def text_to_num(text):
    tokenized = nltk.word_tokenize(text);
    tags = nltk.pos_tag(tokenized)
    print(tags)
    chunkPattern = r""" Chunk0: {((<NN|CD.?|RB>)<CD.?|VBD.?|VBP.?|VBN.?|NN.?|RB.?|JJ>*)<NN|CD.?>} """
    chunkParser = nltk.RegexpParser(chunkPattern)
    chunkedData = chunkParser.parse(tags)
    print(chunkedData)

    for subtree in chunkedData.subtrees(filter=lambda t: t.label() in "Chunk0"):
        exp = ""
        for l in subtree.leaves():
            exp += str(l[0]) + " "
        exp = exp[:-1]
        print(exp)
        try:
            text = text.replace(exp, str(t2n.text2num(exp)))
        except Exception as e:
            print("error text2num ->", e.args)
        print("text2num -> ", text)
    print(text)
    return text
text = "John: Hello, Bob!. Bob: Hi, John!. John: Are you free this weekend ?. Bob: No, I think so, since i have to attend a party tomorrow. John: Sure. Great!"
res=text_to_num(text)
print(res)
'''

'''
from datetime import datetime, timedelta 
  
  
# Get today's date 
presentday = datetime.now() # or presentday = datetime.today() 
  
# Get Yesterday 
yesterday = presentday - timedelta(1) 
  
# Get Tomorrow 
tomorrow = presentday + timedelta(1) 
  
  
# strftime() is to format date according to 
# the need by converting them to string 
print("Yesterday = ", yesterday.strftime('%d-%m-%Y')) 
print("Today = ", presentday.strftime('%d-%m-%Y')) 
print("Tomorrow = ", tomorrow.strftime('%d-%m-%Y')) 
'''
