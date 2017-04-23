try:
   import cPickle as pickle
except:
   import pickle
import pandas as pd
import numpy as np
import utils.data as utild
import sys


"""
Prepage kaggle Titanic dataset.

Output a dataset of image-like matrices (applying character
quantization) for 'Name', 'Ticket' and 'Cabin' columns.
"""

def halt():
    sys.exit(0)

# Read train and test sets.
train = pd.read_csv('data/train_kaggle.csv')
test = pd.read_csv('data/test_kaggle.csv')


# Add a Survived column with zeros to test data frame.
df = pd.DataFrame({'Survived': [0] * test.PassengerId.size})
test = pd.concat([test, df], axis=1)


# Concatenate test and train data frames.
bind = pd.concat([train, test], ignore_index=True)


# Here we will split and throw away unnecessary parts 
# in each row of strings. thus we will have images with 
# relatively small height and width.

# for example lastnames looks usefull in the Name column but
# we can throw firstnames away.

# get lastnames from Name column.
lastnameL = []
ls = bind.Name.str.split(',', 1).tolist()
for x in xrange(0, len(ls)):
    lastnameL.append(ls[x][0])
lastnameSrs = pd.DataFrame(lastnameL)[0]


# get titles from Name as Master, Mr, Mrs, etc.
titlenameL = []
ls = bind.Name.str.split('\,(.*?)\.').tolist()
for x in xrange(0, len(ls)):
    titlenameL.append(ls[x][1].strip())
titlenameSrs = pd.DataFrame(titlenameL)[0]


# Split ticket codes and ticket numbers.
ticketL = []
ticketnumberL = []
ticketcodeL = []
ls = bind.Ticket.str.split(' ').tolist()
df = pd.DataFrame(ls)
for x in xrange(len(df[0])):
    if df[2][x] != None:
        ticketnumberL.append(df[2][x])
    elif df[1][x] != None:
        ticketnumberL.append(df[1][x])
    elif df[0][x] != None and df[0][x].isdigit():
        ticketnumberL.append(df[0][x])
    else:
        ticketnumberL.append(0)
    if not df[0][x].isdigit():
        ticketcodeL.append(df[0][x])
    else:
        ticketcodeL.append(0)


ticketnumberSrs = pd.DataFrame(ticketnumberL)[0]
ticketcodeSrs = pd.DataFrame(ticketcodeL)[0]


# Remove unnecessary characters to reduce the size of the alphabet
ticketcodeSrs = ticketcodeSrs.str.replace('.','')\
                        .str.replace('/','')\
                        .fillna(' ')

lastnameSrs = lastnameSrs.str.replace('-','')\
                        .str.replace("'",'')\
                        .str.replace(' ', '')

cabinSrs = bind.Cabin.str.replace(' ', '')\
                    .fillna(' ')


# get a unique list of chars from these series
titlename_txt = ''
for element in titlenameSrs:
    titlename_txt = titlename_txt + str(element)

lastname_txt = ''
for element in lastnameSrs:
    lastname_txt = lastname_txt + str(element)

ticket_txt = ''
for element in ticketcodeSrs:
    ticket_txt = ticket_txt + str(element)

cabin_txt = ''
for element in cabinSrs:
    cabin_txt = cabin_txt + str(element)


bind_txt = titlename_txt + lastname_txt + ticket_txt + cabin_txt

# build the alphabet
char_list = sorted(list(set(bind_txt.lower())))

# map each unique character to an integer
char_mapping = {}
for i, item in enumerate(char_list):
    char_mapping[char_list[i]] = i


# get max string lengths for each series
max_list = [
            titlenameSrs.str.len().max(),
            lastnameSrs.str.len().max(), 
            ticketcodeSrs.str.len().max(),
            cabinSrs.str.len().max()
           ]

# alphabet length is the width and max string length is
# the height of images 
height = max(max_list)
width = len(char_list)

# quantize each word as an image with 4 channels 
# each column is another image channel here: 'lastnameSrs', 'titlenameSrs', 
# 'ticketcodeSrs', 'cabinSrs'
series_list = [lastnameSrs, titlenameSrs, ticketcodeSrs, cabinSrs]
img_dataset_x = utild.char2quan(series_list, char_mapping, width, height)

img_dataset_y = np.array(bind.Survived, dtype=np.int32)

# split train and test sets
img_dataset_train_x, img_dataset_test_x = img_dataset_x[:891], img_dataset_x[891:]
img_dataset_train_y, img_dataset_test_y = img_dataset_y[:891], img_dataset_y[891:]


# save datasets
with open('data/img_dataset_train_x.pkl', 'wb') as f:
        pickle.dump(img_dataset_train_x, f)
with open('data/img_dataset_train_y.pkl', 'wb') as f:
        pickle.dump(img_dataset_train_y, f)


with open('data/img_dataset_test_x.pkl', 'wb') as f:
        pickle.dump(img_dataset_test_x, f)
with open('data/img_dataset_test_y.pkl', 'wb') as f:
        pickle.dump(img_dataset_test_y, f)



# Save other unhandled remaining columns as another dataset

# There are missing values on Age and Fare columns, remove nans and write the mean value.
ageSrs = utild.nan2mean(bind.Age)
fareSrs = utild.nan2mean(bind.Fare)


# Create a vector representations for the columns with string values:
sexSrs = utild.uniq2multi(bind.Sex)
embarkedSrs = utild.uniq2multi(bind.Embarked)


bind2 = pd.concat([bind.Pclass, sexSrs, ageSrs, bind.SibSp, bind.Parch, fareSrs, embarkedSrs], axis=1)

dataset_train_additional_x, dataset_test_additional_x = bind2[:891], bind2[891:]

with open('data/train_additional_x.pkl', 'wb') as f:
        pickle.dump(dataset_train_additional_x.as_matrix(), f)
with open('data/test_additional_x.pkl', 'wb') as f:
        pickle.dump(dataset_test_additional_x.as_matrix(), f)