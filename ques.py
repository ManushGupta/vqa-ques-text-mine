import json
import numpy
from pycocotools.coco import COCO
import nltk
no_obj_counter = 0;
import gensim

ques_array = []
common_objects=[ 'tv','person', 'bicycle', 'car', 'motorcycle','airplane', 'bus', 'train', 'truck', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep','elephant', 'bear', 'zebra' ,'giraffe','backpack', 'umbrella','tie', 'frisbee','sports', 'ball', 'kite', 'skateboard', 'surfboard', 'bottle', 'cup', 'fork', 'knife', 'bowl','apple', 'sandwich', 'orange', 'pizza', 'cake', 'chair', 'couch', 'bed', 'laptop' ,'mouse', 'remote' , 'cell', 'phone', 'book', 'clock' ,'vase' ,'scissors']
ques_counter = numpy.zeros([len(common_objects),], dtype = int)
#print common_objects[0]
supercategories=[ 'outdoor', 'food' ,'indoor','sports', 'person', 'animal' ,'vehicle', 'furniture','accessory', 'electronic' ,'kitchen']
ques_counter_sc = numpy.zeros([len(supercategories),] , dtype = int)


with open("OpenEnded_abstract_v002_val2015_questions.json") as json_file:  #using the 30,000 question validation set only
    json_data = json.load(json_file)
    #print(json_data['questions'])
    questions = numpy.array(json_data['questions'])
    #print(questions)
    for question in questions:
    	#ques_array[count]=(question['question'])	
        question = question['question'].lower().replace('/',' ').replace('-',' ')
    	#question=dict((k.lower(), v.lower()) for k,v in question.iteritems())
        #print question
        text = nltk.word_tokenize(question)
        
        ques_array.append(text)
        
print ques_array[:100]
#model = gensim.models.Word2Vec(ques_array, min_count=1, workers = 4)

model= gensim.models.Word2Vec.load_word2vec_format('glove_model.txt', binary = False)


for q in ques_array:

    #print q
    tagged_text = nltk.pos_tag(q)
    #print(tagged_text)
    max_overall_score = 0
    
    for text in tagged_text:
        if(text[1] == 'NN'or text[1] == 'NNS'):
            max_score = 0
            max_score_sc = 0
            max_score_obj = "none"
            max_score_obj_sc = "none"
            for obj in common_objects:
                if(text[0]!='combover' and text[0]!='teeter-totter' and text[0]!='loveseats' and text[0]!= 'smores'  and text[0]!= 'bbqing'):

                    if(model.similarity(text[0],obj) > max_score):
                        max_score = model.similarity(text[0],obj)
                        max_score_obj = obj
                    if(max_score>max_overall_score):
                        max_overall_score = max_score


            if(max_score >= 0.5):
                index = common_objects.index(max_score_obj)
                ques_counter[index] += 1


            for obj in supercategories:
                if(text[0]!='combover' and text[0]!='teeter-totter' and text[0]!='loveseats' and text[0]!= 'smores' and text[0]!= 'bbqing'):

                    if(model.similarity(text[0],obj) > max_score_sc):
                        max_score_sc = model.similarity(text[0],obj)
                        max_score_obj_sc = obj
                    if(max_score_sc > max_overall_score):
                        max_overall_score = max_score_sc


            if(max_score_sc >= 0.5):
                index = supercategories.index(max_score_obj_sc)
                ques_counter_sc[index] += 1
    if(max_overall_score  < 0.5):
        no_obj_counter +=1

common_objects = numpy.asarray(common_objects)
print numpy.dstack((common_objects,ques_counter))
print numpy.dstack((supercategories,ques_counter_sc))
print (no_obj_counter)
