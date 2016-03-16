# vqa-ques-text-mine
Text mining in Questions for common objects

Used python to parse the questions into a list with tokenized format. 
Used NLTK for this purpose. 
Used POS tagger provided by NLTK again and filtered out Nouns and plural Nouns as a rough filter for Common Objects. 
Used Word2vec (gensim library) on a pretrained model of word vectors(GloVe.6B.300d) and identified similar categories of objects to be grouped into the same object. (ex, (man,woman,girl)-> person).
Mapped the no of questions referring to a Common object in two arrays(approximated by a similarity value of atleast 0.5) and also calculated the no. of questions which could not refer to any common object(approximated this by using a similarity value of less than 0.5 for any object category in the list of common objects, filtering out non-objects classified as nouns by the tagger)

Note: i have only used a 30k validation question set(from abstract scenes) due to poor download speed. Some identified nouns(5) were not present in the vocab(in the pretrained model) and gave error while mapping similarity, i had to manually remove them. Using a larger vocab resource should solve the problem.

Results:
Common objects list with the no of questions about those objects (only in the 30k validation set)
 ['tv' '454']
  ['person' '5889']
  ['bicycle' '284']
  ['car' '79']
  ['motorcycle' '5']
  ['airplane' '8']
  ['bus' '4']
  ['train' '17']
  ['truck' '13']
  ['bench' '262']
  ['bird' '557']
  ['cat' '1099']
  ['dog' '2341']
  ['horse' '33']
  ['sheep' '517']
  ['elephant' '2']
  ['bear' '62']
  ['zebra' '0']
  ['giraffe' '1']
  ['backpack' '1']
  ['umbrella' '1']
  ['tie' '0']
  ['frisbee' '49']
  ['sports' '288']
  ['ball' '568']
  ['kite' '1']
  ['skateboard' '78']
  ['surfboard' '1']
  ['bottle' '461']
  ['cup' '150']
  ['fork' '2']
  ['knife' '4']
  ['bowl' '2']
  ['apple' '52']
  ['sandwich' '65']
  ['orange' '43']
  ['pizza' '75']
  ['cake' '153']
  ['chair' '558']
  ['couch' '753']
  ['bed' '740']
  ['laptop' '7']
  ['mouse' '196']
  ['remote' '3']
  ['cell' '2']
  ['phone' '6']
  ['book' '303']
  ['clock' '1']
  ['vase' '15']
  ['scissors' '5']]]
[[['outdoor' '12']
  ['food' '478']
  ['indoor' '32']
  ['sports' '288']
  ['person' '5889']
  ['animal' '2635']
  ['vehicle' '78']
  ['furniture' '363']
  ['accessory' '0']
  ['electronic' '15']
  ['kitchen' '890']]]

 no of questions about no common objects/supercategories : 15626/30000
From the results, 0.5 isnt a good value for supercategories and possibly the pretrained model is not adequate.

