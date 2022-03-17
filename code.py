from nltk import pos_tag
from nltk import RegexpParser

from __future__ import print_function
import nltk
import numpy as np
import pandas as pd
import random
import pprint, time
import re
import os.path
import datetime
import os
from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer

import datetime
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


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        # iterate through all file
        text = f.read()
        print(text)

        sentences = sent_tokenize(text)
        sentencesBeforeTagging = [word_tokenize(sent) for sent in sentences]
        sentences = [pos_tag(sent) for sent in sentencesBeforeTagging]

        
        patterns = [
            [r'(\d+|(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                  eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                  ninety|hundred|thousand))[-](year|day|week|month)[-](after|later)', 'DATE'],  # datename_extract
            [r'(today|tomorrow|tonight|tonite)$', 'DATE'],  # datename_extract
            [r'(coimbatore|chennai|madurai|dindigul|Hyderabad|mumbai|pune|bangalore|kochi|delhi|kolkata)$', 'LOCATION'],
            # location extract
            [r'(interview|party|meeting|conference)$', 'EVENT'],  # event_extract
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


        overallGrammar = """
            CLAUSE0: {<IN>?<NNP>+<CD><CD>?}
            CLAUSE1: {<DT><CD>}
            DATE: {<CLAUSE0|CLAUSE1>}
            CLAUSE2: {<VBZ>?<TO><CD><CC|NN|VBP|VBZ>?}
            CLAUSE3: {<IN|VB><RB><CD|IN><CD>?<NN|NNS>}
            CLAUSE4: {<IN><IN><CD><NN>?}
            TIME_END: {<CLAUSE2|CLAUSE3|CLAUSE4>}
            CLAUSE5: {<IN><DT>?<NN>*<NNP>+<NNPS>*<NN>?}
            CLAUSE6: {<IN><DT><NN>}
            CLAUSE7: {<TO><NNP>}
            LOCATION: {<CLAUSE5|CLAUSE6|CLAUSE7>}
            TIME_START: {<CD><NN|VBP|VBZ>?}
        """



        date_patterns = [
            [r'(today|tomorrow|tonight|tonite)$', 'DATE1'],
            [r'(this|next)[-](year|day|week|month|weekend)$', 'DATE2'],
            [r'(\d+|(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                        eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                        ninety|hundred|thousand))[-](year|day|week|month)[-](after|later)', 'DATE3'],
        ]



        dateNounGrammar = """
            DATE1: {<JJ><NN>+}
            DATE2: {<DT><NN>+}
            DATE3: {<DT><NNP>+}
            DATE4: {<DT|JJ|NN><VBG>}
            DATE5: {<NN>+}
            DATE6: {<JJ><NNP>}
        """
        eventGrammar = """
            EVENT1: {<DT><NN><VBG><NN>}
            EVENT2: {<DT|VBG><NN>+}
            EVENT3: {<VB|VBG><IN><NN>+}
            EVENT4: {<VBG|VBP><NNP>?<NNS>}
            EVENT5: {<NNS><VBP>}
            EVENT6: {<VB><NN|RP>}
            EVENT7: {<VB><DT><NN>}
            EVENT8: {<DT><NN><VBG><NN>}
            EVENT9: {<NN>}
        """

        dateTimeLocationAndEventList = []
        parser1 = RegexpParser(overallGrammar)
        parser2 = RegexpParser(dateNounGrammar)
        parser3 = RegexpParser(eventGrammar)
        for sentence in sentences:
            result1 = parser1.parse(sentence)
            result2 = parser2.parse(sentence)
            result3 = parser3.parse(sentence)
            dateTimeLocationAndEventResult = cleanTaggedExpressions(result1, result2, result3, sentencesBeforeTagging, emailInput)
            dateTimeLocationAndEventList.append(dateTimeLocationAndEventResult)

        resultString = ""
        for result in dateTimeLocationAndEventList:
            for iter in result:
                if ("undetermined" not in iter) and (iter not in resultString):
                    resultString += iter
                    resultString += ", "
        for info in infoTypes:
            if info not in resultString:
                resultString += info + ": undetermined"
                resultString += ", "

        resultString = resultString.rstrip(" ")
        resultString = resultString.lstrip(" ")
        resultString = resultString.rstrip(",")
        resultString = resultString.lstrip(",")

        # So if multiple dates were found by the tagger then just offer
        # The other date as additional info, this ultimately makes the program more robust!
        # Forget about checking times, because this is already double checked by regex!
        for info in infoTypes:
            if(info == 'DATE'):
                checkerInfo = info + ":"
                count = resultString.count(info)
                if(count > 1):
                    newString = resultString.rsplit(info, resultString.count(info) - 1)
                    new = info + "_ADDITIONAL_INFO_FOUND"
                    resultString = new.join(newString)

        print(resultString)
        return resultString
        create_event(date, event_list, "Publish the paper", "chennai")


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

