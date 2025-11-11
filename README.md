🧾 these python scripts can verify text authorship by examining claimed authors' familiarity with the text content.  
🧾 each script starts with a students.tsv file that contains student numbers, student names, and text columns.
  
here is the initial batch:
  
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

&nbsp;&nbsp;&nbsp;&nbsp;📜 synonym-replacer.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, an LLM replaces 5 of those words in the text sample with synonyms, and the claimed author should be able to find the synonyms and identify the original word choices.   
*requires weasyprint, wiki_freq.txt (included), and a deepseek API key (compatible with other OpenAI format LLM APIs)
