import sys
import os.path
import numpy as np
from collections import defaultdict, Counter


import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """   
    #use defaultdict to avoid key error when encounterin a new word 
    #in the dictionary
    words_dict=Counter()
    #Add a pseudocount to every entry seen/unseen (smoothing)
    words_dict=defaultdict(lambda:1)
    
    for file in file_list:
        #use util function to get a list of all words in a file
        #one occure in each email file is enough
        words_in_file=set(util.get_words_in_file(file))
        
        for word in words_in_file:
            words_dict[word]+=1
            
    return words_dict

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    #Count the number of files
    numFiles=len(file_list)
    #get counts of all words appearing in files in file list
    words = get_counts(file_list)
    
    words_log = {word: util.careful_log(counts) for word, counts in words.items()}

    return words_log

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    #initalize list of dictionary for storing log probabilities of word occurences
    log_probabilities_by_category=[]
    #intialize list of counts for being ham or spam
    log_prior=[]
    total_emails = sum([len(cat) for cat in file_lists_by_category])
    
    for index in range(0, len(file_lists_by_category)):
        #for each element of list get log probabilities of word counts
        log_probabilities_by_category.append(get_log_probabilities(file_lists_by_category[index]))
        #calclogpriors
        class_counts=len(file_lists_by_category[index])
        log_prior.append(util.careful_log(class_counts/total_emails))

    return log_probabilities_by_category, log_prior

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
   
    #labels: 0 us spam, 1:spam, use for return value
    labels={0:'spam', 1:'ham'}
    
    #get unique occurences of word in each file
    new_words=set(util.get_words_in_file(email_filename))
    
    #calcualte posterior distribution for spam and ham
    posterior_dist=[]
    for label in labels:
        posterior_dist.append(log_prior_by_category[label])
        for word in new_words:
            #p_i or q_i
            if word in log_probabilities_by_category[label]:
                posterior_dist[label] += log_probabilities_by_category[label][word]
            #1-p_i or q_i
            else:
                posterior_dist[label] += 1-log_probabilities_by_category[label][word]
    
    #return MAP from log odds
    if posterior_dist[0]/posterior_dist[1] >= 0:
        #return spam
        return labels[0]
    else:    
        return labels[1]

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
