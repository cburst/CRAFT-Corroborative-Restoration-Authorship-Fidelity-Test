🧾 these python scripts can address text authorship fidelity by examining claimed authors' familiarity and ability to reconstruct text.  
🧾 each script starts with a students.tsv file that contains student numbers, student names, and text columns.
🧾 each script generates student test PDFs and answer key(s).

**hybrid tests**  
&nbsp;&nbsp;&nbsp;&nbsp;📜 hybrid-intruders.py  
this script uses an LLM to create additional sentences in the original text sample, and the claimed author should be able to identify the impostor sentences.  
this script also shuffles sentences in the original text sample, and the claimed author should be able to recorder the original sentences.   
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 hybrid-assembler-replacer.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, an LLM replaces 5 of those words in the text sample with synonyms, and the claimed author should be able to find the synonyms and identify the original word choices.   
this script also removes a 10-word block from the original text sample, puts those words into an alphabetized word bank, and the claimed author should be able to reassemble the original block.   
*requires weasyprint, wiki_freq.txt (included), and a deepseek API key (compatible with other OpenAI format LLM APIs)


**standalone tests**  
&nbsp;&nbsp;&nbsp;&nbsp;📜 hybrid-assembler-replacer.py  
this script removes a 20-word block from the original text sample, puts those words into an alphabetized word bank, and the claimed author should be able to reassemble the original block.   
*requires weasyprint

&nbsp;&nbsp;&nbsp;&nbsp;📜 sentence-completer.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, replaces them with blanks, and the claimed author should fill-in-the-blanks.  
*requires weasyprint and wiki_freq.txt (included)

&nbsp;&nbsp;&nbsp;&nbsp;📜 sentence-creator.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, and the claimed author should be able to make a new sentence with these words.  
*requires weasyprint and wiki_freq.txt (included)

&nbsp;&nbsp;&nbsp;&nbsp;📜 authorship-recognizer.py  
this script uses an LLM to create two plausible decoy sentences for the 5 longest sentences in the original text sample, and the claimed author should be able to identify the sentence they created.  
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 sentence-intruder.py  
this script uses an LLM to create an additional sentence in the original text sample, and the claimed author should be able to identify the impostor sentence.   
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 sentence-intruders.py  
this script uses an LLM to create additional sentences in the original text sample, and the claimed author should be able to identify the impostor sentences.   
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 sentence-intruders.py  
this script uses an LLM to create additional sentences in the original text sample, the sentence order is shuffled, and the claimed author should be able to identify the impostor sentences.   
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 synonym-replacer.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, an LLM replaces 5 of those words in the text sample with synonyms, and the claimed author should be able to find the synonyms and identify the original word choices.   
*requires weasyprint, wiki_freq.txt (included), and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 summary-recognizer.py  
this script identifiesuses an LLM to create an accurate summary of the original text, as well as two summaries with minor detail changes, and the claimed author should be able to find the accuracy summar.   
*requires weasyprint and a deepseek API key (compatible with other OpenAI format LLM APIs)

**pipeline script**
📜 test_pipeline.py  
example python pipeline for organizing and merging pdf test files and answer keys created to facilitate the creation of a double-sided single page test for each student
