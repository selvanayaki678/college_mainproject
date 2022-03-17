# Importing libraries
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

        #	DA1: {<IN>?<NNP>+<CD><CD>?}
        #	DA2: {<DT><CD>}
        #	DATE: {<DA1|DA2>}
        #	TE3: {<IN><RB><CD><NN|NNS>}
        #	TE4: {<VB><RB><IN><CD>}
        #	TIME_END: {<TE1|TE2|TE3|TE4>}
        #	TS1: {<CD><NN|VBP|VBZ>?}
        #	TS2: {<VBZ><IN><CD>}
        #	TIME_START: {TS1|TS2}
        #   L1: {<IN><DT>?<NN>*<NNP>+<NNPS>*<NN>?}
        #    L2: {<IN><DT><NN>}
        #    L3: {<TO><NNP>}
        #    LOCATION: {<L1|L2|L3>}
        # 	"""

        dateNounGrammar = """
                DATE1: {<JJ><NN>+}
                DATE2: {<DT><NN>+}
                DATE3: {<DT><NNP>+}
                DATE4: {<DT|JJ|NN><VBG>}
                DATE5: {<NN>+}
                DATE6: {<JJ><NNP>}
                """

        # DATE1 for catching things like "friday night" or "thursday night" where the day isn't capitalized and thus is JJ
        # DATE2 for "the evening time" or something like that.
        # DATE3 for "this Friday" (the tagger messes up the classification of capitalized days, etc.)
        # DATE4 for "this evening", or "(t/T)hursday evening"
        # DATE5 for "tonight", "tomorrow night", "this afternoon", "this evening", "lunch", "dinner", etc.
        # DATE6 for "this Friday", etc.

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
        # EVENT1 for "a cake eating contest
        # EVENT2 for "having lunch" or "a meeting", or "curriculum meeting" etc.
        # EVENT3 for "wrestling in space" or "wrestle in space" or "going for ice cream" , etc.
        # EVENT4 for things like "buying Guinness beer"
        # EVENT5 for "doctor's appointment"
        # EVENT6 for "drive home" or "run away"
        # EVENT7 for "running the tap"
        # EVENT8 for "lunch" or "dinner", etc.This is last because the other POS sequences should have priority.
        # EVENT9 for pretty much everything else that could be valid.

        # Extra location grammar
        sentencesBeforeTagging = [word_tokenize(sent) for sent in sent_list]
        # file_object = open( homeDirectory+ "testerDataOutput.txt", "a")
        dateTimeLocationAndEventList = []
        parser1 = nltk.RegexpParser(overallGrammar)
        parser2 = nltk.RegexpParser(dateNounGrammar)
        parser3 = nltk.RegexpParser(eventGrammar)
        for sentence in sent_list:
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
            if (info == 'DATE'):
                checkerInfo = info + ":"
                count = resultString.count(info)
                if (count > 1):
                    newString = resultString.rsplit(info, resultString.count(info) - 1)
                    new = info + "_ADDITIONAL_INFO_FOUND"
                    resultString = new.join(newString)

        return resultString

if __name__=="__main__":
    # Folder Path
    path = r"D:\%%%% Vani %%%%\____Main Project___\Conversations"
    # Change the directory
    os.chdir(path)
    # Read text File
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # call read text file function
            read_text_file(file_path)





























"""
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

                    today = datetime.date.today()
                    Date = ''.join(map(str, date_list))
                    Time = ''.join(map(str, time_list))

                    for i in range(len(single_date_val)):
                        if single_date_val[i] == 'tomorrow':
                            print("Date calculation")
                            date = today + datetime.timedelta(days=1)
                            print('Tomorrow', date)
                        if single_date_val[i] == 'today' or 'tonight' or 'tonite':
                            print("Date calculation")
                            date = today
                            print('single_date_val[i]', date)
            print('---------------------')

if __name__=="__main__":
    # Folder Path
    path = r"D:\%%%% Vani %%%%\____Main Project___\Conversations"
    # Change the directory
    os.chdir(path)
    # Read text File
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # call read text file function
#            read_text_file(file_path)


"""
























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

                    today = datetime.date.today()
                    Date = ''.join(map(str, date_list))
                    Time = ''.join(map(str, time_list))

                    for i in range(len(single_date_val)):
                        if single_date_val[i] == 'tomorrow':
                            print("Date calculation")
                            date = today + datetime.timedelta(days=1)
                            print('Tomorrow', date)
                        if single_date_val[i] == 'today' or 'tonight' or 'tonite':
                            print("Date calculation")
                            date = today
                            print('single_date_val[i]', date)
                    date=Date+Time
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





dateNounGrammar = """
        DATE1: {<JJ><NN>+}
        DATE2: {<DT><NN>+}
        DATE3: {<DT><NNP>+}
        DATE4: {<DT|JJ|NN><VBG>}
        DATE5: {<NN>+}
        DATE6: {<JJ><NNP>}
        """

    # DATE1 for catching things like "friday night" or "thursday night" where the day isn't capitalized and thus is JJ
    # DATE2 for "the evening time" or something like that.
    # DATE3 for "this Friday" (the tagger messes up the classification of capitalized days, etc.)
    # DATE4 for "this evening", or "(t/T)hursday evening"
    # DATE5 for "tonight", "tomorrow night", "this afternoon", "this evening", "lunch", "dinner", etc.
    # DATE6 for "this Friday", etc.

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    #
    # This is now the Grammar that will be used to extract events.
    # Keep in mind that it is often the first noun in the scheduling email that will be found
    # This is a known fact in information extraction.
    #
    # For example see:
    # http://www.iosrjournals.org/iosr-jce/papers/Conf-%20ICFTE%E2%80%9916/Volume-1/12.%2072-79.pdf?id=7557
    #
    # I was also able to come up with a grammar based on all of the random sentences I generate.
    #
    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

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
    # EVENT1 for "a cake eating contest
    # EVENT2 for "having lunch" or "a meeting", or "curriculum meeting" etc.
    # EVENT3 for "wrestling in space" or "wrestle in space" or "going for ice cream" , etc.
    # EVENT4 for things like "buying Guinness beer"
    # EVENT5 for "doctor's appointment"
    # EVENT6 for "drive home" or "run away"
    # EVENT7 for "running the tap"
    # EVENT8 for "lunch" or "dinner", etc.This is last because the other POS sequences should have priority.
    # EVENT9 for pretty much everything else that could be valid.



