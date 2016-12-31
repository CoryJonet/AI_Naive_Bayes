import java.util.*;
/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
    
    private int HAMInstanceHits = 0; // How many instances of HAM
    private int SPAMInstanceHits = 0; // How many instances of SPAM
    private int trainingDataLength = 0; // Length of the training data
    private int words = 0; // Number of words (directly from v in train())

    private double hamSmoothingDenom = 0.0; // Laplace smoothing denominator for HAM, so p_w_given_l can access
    private double spamSmoothingDenom = 0.0; // Laplace smoothing denominator for SPAM, so p_w_given_l can access

    private List <String> dictionary = new ArrayList <String>(); // All words in the SMS message(s) 
    
    // Links word to how many times it appears (for HAM and SPAM)
    private HashMap <String, Integer> HAMHits = new HashMap <String, Integer>();
    private HashMap <String, Integer> SPAMHits = new HashMap <String, Integer>();

    // Links word to probability it appears given (SPAM or HAM), so p_w_given_l can access
    private HashMap <String, Double> hamSmoothing = new HashMap <String, Double>();
    private HashMap <String, Double> spamSmoothing = new HashMap <String, Double>();
    
    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
	public void train(Instance[] trainingData, int v) {

	/**
	 * "This method should train your classifier with the training data
	 * provided. The integer argument v is the size of the total vocabulary
	 * in your model. Please take this argument and store it as a field,
	 * as you will need it in computing the smooth class-conditional
	 * probabilities."
	 */
	// Make the training data length global
	trainingDataLength = trainingData.length;

	// Make words (v) global
	words = v;
	        
	// Loop through all training data instances (sentences)
	for (int i = 0; i < trainingData.length; i++) {

	    // If this label is HAM, increment HAM instance count
	    if (trainingData[i].label == Label.HAM)
		HAMInstanceHits += 1;

	    // Otherwise, increment SPAM instance count (should only be SPAM and HAM...)
	    else
		SPAMInstanceHits += 1;
	            
	    // Loop through all training data words (within instances (sentences))
	    for (int j = 0; j < trainingData[i].words.length; j++) {
		
		// If HAM
		if (trainingData[i].label == Label.HAM) {

		    // If the word isn't in the Hashmap, put it in with count of 1
		    if (!HAMHits.containsKey(trainingData[i].words[j]))
			HAMHits.put(trainingData[i].words[j], 1);
		    
		    // Otherwise, increment count of word in Hashmap (counter)
		    else
			HAMHits.put(trainingData[i].words[j], 
				    HAMHits.
				    get(trainingData[i].words[j]) + 1);
		}

		// Otherwise should be SPAM (can only have HAM or SPAM)
		else {
		    
		    // If the word isn't in the Hashmap, put it in with count of 1
		    if(!SPAMHits.containsKey(trainingData[i].words[j]))
			SPAMHits.put(trainingData[i].words[j], 1); 
		    
		    // Otherwise, increment count of word in Hashmap (counter)
		    else
			SPAMHits.put(trainingData[i].words[j],
				     SPAMHits.
				     get(trainingData[i].words[j]) + 1);
		}

		// Lastly, add to the dictionary (is the word in the SMS message hasn't been found)
		if (!dictionary.contains(trainingData[i].words[j]))		 
		    dictionary.add(trainingData[i].words[j]);
	    }
	}
	
	/** I was having issues putting the smoothing code in p_w_given_l 
	 * (it was just hanging..so I figured it would be ok to calculate here 
	 * and save contents in global variables)
	 */

	/**
	 * Smoothing: (Laplace smoothing)
	 *
	 * P(w|l) = (C(w) + delta) / (V * delta + summation over v within V of C(w))
	 * where l is Spam, ham, C(w) as representing number of times the token w
	 * appears in messages labeled l in the training data.
	 */
	double delta = 0.00001;
	double CHam = 0.0;
	double hamDenominator = 0.0;
	double CSpam = 0.0;
	double spamDenominator = 0.0;

	// Loop through our accrued dictionary (to compute C(w)) which is "number of times"
	for (int i = 0; i < dictionary.size(); i++) {
	    
	    // Numerator (C(HAM1) + delta)
	    CHam = delta;
	    
	    // Numerator finished (C(HAMn) + delta)
	    if (HAMHits.containsKey(dictionary.get(i)))
		CHam += ((double) (HAMHits.get(dictionary.get(i))));
	            
	    // Denominator (V * delta)
	    hamDenominator = ((double) (words) * delta);

	    // Loop through our accrued dictionary to compute summation in denominator
	    for (int j = 0; j < dictionary.size(); j++) {
		
		// Denominator ((V * delta) + summation(C(v)))
		if(HAMHits.containsKey(dictionary.get(j)))
		    hamDenominator += ((double) (HAMHits.get(dictionary.get(j))));

	    }
	            
	    // Add into hashmap so p_w_given_l can access
	    hamSmoothing.put(dictionary.get(i), ((double) (CHam / hamDenominator)));
	    hamSmoothingDenom = hamDenominator;
	}
	
	// Loop through our accrued dictionary (to compute C(w)) which is "number of times"
	for (int i = 0; i < dictionary.size(); i++) {
	    
	    // Numerator (C(SPAM1) + delta)
	    CSpam = delta;
	    
	    // Numerator finished (C(HAMn) + delta)
	    if(SPAMHits.containsKey(dictionary.get(i)))
		CSpam += ((double) (SPAMHits.get(dictionary.get(i))));
	            
	    // Denominator (V * delta)
	    spamDenominator = ((double) (words) * delta);

	    // Loop through our accrued dictionary to compute summation in denominator
	    for (int j = 0; j < dictionary.size(); j++) {

		// Denominator ((V * delta) + summation(C(v)))
		if(SPAMHits.containsKey(dictionary.get(j)))
		    spamDenominator += ((double) (SPAMHits.get(dictionary.get(j))));

	    }
	    
	    // Add into hashmap so p_w_given_l can access
	    spamSmoothing.put(dictionary.get(i), ((double) (CSpam / spamDenominator)));
	    spamSmoothingDenom = spamDenominator;
	}  
    }

    /**
     * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
     */
    @Override
	public double p_l(Label label) {
	
	// If HAM, P(HAM) = HAM Instance (sentence) hits / number of instances (sentences)
	if (label == Label.HAM)
	    return ((double) HAMInstanceHits) / ((double) trainingDataLength);

	// If SPAM, P(SPAM) = SPAM Instance (sentence) hits / number of instances (sentences)
	else
	    return ((double) SPAMInstanceHits) / ((double) trainingDataLength);
    }

    /**
     * Returns the smoothed conditional probability of the word given the label,
     * i.e. P(word|SPAM) or P(word|HAM)
     */
    @Override
	public double p_w_given_l(String word, Label label) {

	/**
	 * "This method should return the probability of the word conditioned on the label.
	 * In other words, return P(word | label). To compute this probability, you will use smoothing.
	 * Please read the note on how to implement this below."
	 */

	double delta = 0.00001; // "For this assignment, please use the value, delta = 0.00001"

	/**
	 * "One complication for a Naive Bayes classifier is the presence of unobserved
	 * events in test data. Let's be concrete about this in the case of our
	 * classification task...What we do to get around this is that we pretend we actually
	 * did see some (possibly fractionally many) tokens for the word type <not present>"
	 */

	// Otherwise, the word does exist, just return the pre-calculated smoothing from train()
	if (dictionary.contains(word)) {
	    
	    if (label == Label.HAM)
		return hamSmoothing.get(word);
 
	    else
		return spamSmoothing.get(word);

	}

	// So, if the word token doesn't exist in our dictionary from the SMS messages, use delta only
	else {

	     if (label == Label.HAM)
		return ((double) (delta / hamSmoothingDenom)); 

	    else
		return ((double) (delta / spamSmoothingDenom));

	}
    }
    
    /**
     * Classifies an array of words as either SPAM or HAM. 
     */
    @Override
	public ClassifyResult classify(String[] words) {

	// "This method returns the classification result for a single message"

	// Instantiate new ClassifyResult to hold results to return
	ClassifyResult resultsToReturn = new ClassifyResult();
	    
	/**
	 * Log probabilities:
	 * The second gotcha that any implementation of a Naive Bayes classifier
	 * must contend with is underflow. Underflow can occur when we take the 
	 * product of a number of small floating-point values. Fortunately, there
	 * is a workaround. Recall that a Naive Bayes classifier computes:
	 * f(w) = argmax[P(Ham) * PI from i = 1 to k of (P(w | HAM)
	 * f(w) = argmax[P(Spam) * PI from i = 1 to k of (P(w | SPAM)
	 * where l is HAM or SPAM and w is the ith word of your SMS message, numbered
	 * 1 to k.
	 * 
	 * So, for the first part: 
	 * g(w) = argmax[log P(Ham) + ...
	 * g(w) = argmax[log P(Spam) + ...
	 */

	// g(w) = argmax[log P(Ham) + ...
	resultsToReturn.log_prob_ham  = Math.log(((double) HAMInstanceHits) / 
						 (double) (trainingDataLength));

	// g(w) = argmax[log P(Spam) + ...
	resultsToReturn.log_prob_spam = Math.log(((double) SPAMInstanceHits) / 
						 (double) (trainingDataLength));
	
	/**
	 * Now, the summation part:
	 *
         * g(w) = ... + summation from i = 1 to k of (logP(w | ham))]
	 * g(w) = ... + summation from i = 1 to k of (logP(w | spam))]
	 */
	for (int i = 0; i < words.length; i++) {
	    
	    // g(w) = ... + summation from i = 1 to k of (logP(w | ham))]
	    resultsToReturn.log_prob_ham += 
		Math.log(p_w_given_l(words[i], Label.HAM));

	    // g(w) = ... + summation from i = 1 to k of (logP(w | spam))]
	    resultsToReturn.log_prob_spam  += 
		Math.log(p_w_given_l(words[i], Label.SPAM));
	
	}

	// Lastly, label the ClassifyResult as ham if log_prob_ham is greater (argmax)
	if (resultsToReturn.log_prob_ham > resultsToReturn.log_prob_spam)
	    resultsToReturn.label = Label.HAM;

	// Lastly, label the ClassifyResult as spam if log_prob_spam is greater (argmax)
	else
	    resultsToReturn.label = Label.SPAM;

	return resultsToReturn;

    }
}
