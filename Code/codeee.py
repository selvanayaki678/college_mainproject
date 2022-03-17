#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import nltk
nltk.download('averaged_perceptron_tagger')
import os
from nltk import tree
from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer


# In[2]:


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


# In[4]:


infoTypes = ["LOCATION", "DATE", "TIME_START", "TIME_END", "EVENT"]

helpfulWords = ["this", "next", "tomorrow", "tonight", "evening",
                "morning", "autumn", "fall", "spring", "winter",
                "afternoon", "dawn", "dusk", "later", "soon", "weekend",
                "twilight", "whenever", "night", "sunset", "sunrise" "daytime",
                "daybreak", "nightfall", "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday", "month",
                "week", "year", "day", "soonish", "monthly", "weekly", "annually",
                "daily", "occasional", "perennial", "hourly", "january", "february",
                "march", "april", "may", "june", "july", "august", "september",
                "october", "november", "december"]

meals = ["breakfast", "brunch", "lunch", "dinner", "supper", "dessert"]

numbers = "(one|two|three|four|five|six|seven|eight|nine|ten|           eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|           eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|           ninety|hundred|thousand|noon|midnight)"

timeExpression = re.compile("((2[0-3](:)?[0-5][0-9]|[0-1][0-9](:)?[0-5][0-9]|24(:)?00)" + "|(" + numbers + ")+)")


# In[5]:


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


# In[64]:


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        # iterate through all file
        text = f.read()

        sentences = sent_tokenize(text)
        sentencesBeforeTagging = [word_tokenize(sent) for sent in sentences]
        sentences = [pos_tag(sent) for sent in sentencesBeforeTagging]


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
            dateTimeLocationAndEventResult = cleanTaggedExpressions(result1, result2, result3, sentencesBeforeTagging, text)
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

        #result_dic=dict(resultString)
        print('res',resultString,'\n')
        
        return resultString
        create_event(date, event_list, "Publish the paper", "chennai")


# In[65]:



def cleanTaggedExpressions(overallParseTree, dateNounParseTree, eventParseTree, rawTokens ,rawEmailString):

    timeStart = ""
    timeEnd = ""
    location = ""
    date = ""
    event = ""

    listOfOverallInfo = getListOfData(overallParseTree)
    listOfDateNounInfo = getListOfData(dateNounParseTree)
    listOfEventInfo = getListOfData(eventParseTree)

    # Here give priority to meals, like "breakfast" etc.
    eventFoundAlready = False
    if (len(listOfEventInfo) > 1):
        for eventIter in listOfEventInfo:
            eventIter = eventIter.replace('EVENT: ', '')
            if eventIter in meals:
                eventFoundAlready = True
                event = 'EVENT: ' + eventIter


    for item in listOfOverallInfo:
        if "TIME_START" in item:
            if timeStart == "":
                timeStart = item
        elif "TIME_END" in item:
            timeEnd = item
        elif "LOCATION" in item:
            location = item
        elif "DATE" in item:
            date = item


    # Now return the first catch w.r.t. the email.  Relying on the fact that the event is normally within the
    # first noun phrase in the email.
    # Also, I am using a ratio:
    #       loc = (location of NP in raw email string)
    #       count = (# of words in NP)
    #       ratio: loc/(3^count)
    # The value with the lowest result gets chosen as the event.
    # Therefore, if the location is 0, then it is automatically chosen.

    currMaxRatio = len(rawEmailString)
    newEventPos = 0
    if ((not eventFoundAlready) and (not (len(listOfEventInfo) == 0))):
        for item in listOfEventInfo:
            itemsLocation = 0
            tempItem = item.replace('EVENT: ', '')
            itemsLocation = rawEmailString.find(tempItem)
            numItems = len(tempItem.split())
            nextMaxRatio = itemsLocation/pow(3, numItems)
            if (nextMaxRatio < currMaxRatio):
                currMaxRatio = nextMaxRatio
                newEventPos = listOfEventInfo.index(item)
        event = listOfEventInfo[newEventPos]

    # If there is no result at all returned for the start and end times, try using a regex to capture the times.
    # If regex couldn't find precisely 2 times either...well then we are S.O.L. so just ignore the start and end times.
    if (timeStart == "" and timeEnd == ""):
        timeStart, timeEnd = doubleCheckStartAndEndTimesUsingRegex(rawTokens)
        if (timeStart == "" and timeEnd == ""):
            listOfOverallInfo.append("TIME_START: undetermined")
            listOfOverallInfo.append("TIME_END: undetermined")
        else:
            listOfOverallInfo.append("TIME_START: " + timeStart)
            listOfOverallInfo.append("TIME_END: " + timeEnd)

    elif(timeStart == "" and (not(timeEnd == ""))):
        timeStart = tryToFindTheOtherTime(rawTokens, timeEnd)
        if (timeStart == ""):
            listOfOverallInfo.append("TIME_START: undetermined")
        else:
            listOfOverallInfo.append("TIME_START: " + timeStart)


    elif(timeEnd == "" and (not (timeStart == ""))):
        timeEnd = tryToFindTheOtherTime(rawTokens, timeStart)
        if (timeEnd == ""):
            listOfOverallInfo.append("TIME_END: undetermined")
        else:
            listOfOverallInfo.append("TIME_END: " + timeEnd)

    #If the tagger couldn't find the location, date, or event, then just return undetermined.
    if(date == ""):
        listOfOverallInfo.append("DATE: undetermined")
    if(location == ""):
        listOfOverallInfo.append("LOCATION: undetermined")
    if (event == ""):
        listOfOverallInfo.append("EVENT: undetermined")
    elif (not(event =="")):
        listOfOverallInfo.append(event)

    # Still make sure to return extra date info if it is found, as well as EVENT INFORMATION!!!!!!!!!
    listOfOverallInfo.extend(listOfDateNounInfo)

    return listOfOverallInfo


# In[66]:


def getListOfData(parseTree):
    #Iterate through elements if the popped element is a tree
    #Then parse further and get data from it
    listOfInfo = []
    tag = parseTree.pop()
    while(tag):
        if (type(tag) is tree.Tree):

            #okay we know it is an inner tree now. Check to see which one it is...LOCATION, DATE, TIME_START, TIME_END, EVENT
            #once we know that then append the children to form the info to return

            stringToForm = ""
            for infoType in infoTypes:
                if infoType in str(tag):
                    firstIter = True
                    tag = tree.Tree.flatten(tag)
                    for childNode in tag:
                        if firstIter:
                            stringToForm += infoType
                            firstIter = False
                            stringToForm += ": "
                        stringToForm += childNode[0]
                        stringToForm += " "
                    stringToForm = stringToForm.rstrip()
                    listOfInfo.append(stringToForm)
                    break
        try:
            tag = parseTree.pop()
        except:
            break

    # So now our list of info should include "LOCATION", "DATE", "TIME_START", "TIME_END", and "EVENT"
    # However, if the tagger didn't work (which is often the case), then we should check
    # if the list only contains the infoType "DATES" and has more than one match, then we must use a regex to
    # check and see if "tomorrow", "evening", etc. are matched and give that as additional date information!

    flag = True
    for info in listOfInfo:
        if "DATE" not in info:
            flag = False
            break

    # Now use the following simple algorithm for catching:
    # DATE1 for "tonight", "tomorrow night", "this afternoon", "this evening", etc.
    # DATE2 for catching things like "friday night" or "thursday night" where the day isn't capitalized and thus is JJ
    # DATE3 for "the evening time" or something like that.
    # DATE4 for "this Friday"

    # Note: Regex wasn't necessary to catch this extra data here, however it was used for time information later on!
    if(flag):
        newList = []
        extraDateInfoString = ""
        for info in listOfInfo:
            info = info.replace("DATE: ", "")
            words = info.split()
            for word in words:
                if word.lower() in helpfulWords:
                    extraDateInfoString += word
                    extraDateInfoString += " "
            if(not ((extraDateInfoString == "this") or (extraDateInfoString == "next") or (extraDateInfoString == "next ") or (extraDateInfoString == "this "))):
                if(not (extraDateInfoString == "")):
                    extraDateInfoString = extraDateInfoString.rstrip()
                    newList.append("EXTRA_DATE: " + extraDateInfoString)
                    extraDateInfoString = ""
        listOfInfo = newList

    # And return the info we found.
    return listOfInfo


# In[67]:


def doubleCheckStartAndEndTimesUsingRegex(rawTokens):
    times = []
    for token in rawTokens[0]:
        result = timeExpression.match(token)
        if(result):
            times.append(result.group())

    times = sorted(times)
    if(len(times) == 2):
        return times[0], times[1]
    else:
        return "", ""


# In[68]:


def tryToFindTheOtherTime(rawTokens, timeFoundByTagger):
    times = []
    taggerTime = re.findall(timeExpression, timeFoundByTagger)
    for token in rawTokens[0]:
        result = timeExpression.match(token)
        if(result):
            times.append(result.group())

    # We can only return the other time value with uncertainty if the value is greater than 2
    if(len(times) >= 2):
        for time in times:
            if (not (time == taggerTime)):
                return time
    elif (len(times) == 1):
        if(not (times[0] == taggerTime)):
            return times[0]
    else:
        return ""


# In[69]:


if __name__=="__main__":
    # Folder Path
    path = r"C:\Users\User\Desktop\Main Project\conversations"
    # Change the directory
    
    
    
    
    
    
    
    3os.chdir(path)
    # Read text File
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # call read text file function
            read_text_file(file_path)


# In[ ]:




