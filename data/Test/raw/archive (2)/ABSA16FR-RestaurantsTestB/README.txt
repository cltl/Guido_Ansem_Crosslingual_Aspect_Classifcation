
== SemEval-2016 Task 5 Aspect-Based Sentiment Analysis (ABSA) task for French ==

Test data Phase B (Subtask 1, Sentence-level ABSA): reviews for the Restaurant domain with category and target entity annotations (no polarity values)
--------------------------------------------------

The French review annotations are distributed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Licence (https://creativecommons.org/licenses/by-nc-nd/4.0/). 
The software is distributed under licence CeCILL (http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html)

------------------------
Contents of this package
------------------------

The release package contains four files:

 - ABSA16FR_Restaurants_TestB.xml: an XML file compliant with the ABSA dataset format containing the ids of the reviews used for testing, and the category and target entity annotations. <text> tags in the XML file are empty for copyright reasons.
 - ABSA16FR-download.jar: a jar file for downloading the reviews from the Web and filling the <text> tags in the XML file with the corresponding sentences.
 - ABSA16FR_Restaurants_Test_index.txt: a mapping between review ids and their URLs. The number of sentences in a review is given at the end of each line. 
 - README.txt: this file


-------------------------
How to obtain the dataset
-------------------------

Requirements: java >= 1.5

Open a terminal, move to the directory containing these files, and run:

java -jar ABSA16FR-download.jar ABSA16FR_Restaurants_Test_index.txt ABSA16FR_Restaurants_TestB.xml

This will start the download and filling-in process and will write the output to a file named ABSA16FR_Restaurants_TestB-withcontent.xml     

Please note that:
 - as a courtesy to the web site we get the reviews from, there is a short waiting time between two downloads. As a result, completing the process can take a while.
 - after each download, a <Review> tag in the output file is filled in. You can interrupt the process at any moment without losing the downloaded content. Just choose ‘K’ when the "Should we [O]verwrite, [K]eep already downloaded content or [C]ancel? [O/K/C]" question appears the next time you run the process and it will continue at the point it stopped. To overwrite the already downloaded content choose ‘O’.

Please report any issues (runtime errors, download errors or non coherent data) to the French ABSA-2016 organizers.


-------
Credits
-------
 - Reviews are extracted from HTML with HTMLCleaner.
 - Sentence splitting is done with OpenNLP toolkit.
