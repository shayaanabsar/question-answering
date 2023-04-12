# Question-Answering

A simple Question-Answering model that reads in a corpus and is able to answer comprehension-style questions on the data. It makes use of techniques including lemmatization, the tf-idf formula and Part-Of-Speech tagging.

To setup the code, run the following:
```bash
$ pip3 install -r requirements.txt
$ python3 setup.py
```

To run the program:
```bash
$ python3 main.py
```

Examples:
```
>>> How many Olympics took place in the United States?
Eight Olympic Games have taken place in the United States.

>>> Who is the national personification of the United Kingdom?
Britannia is a national personification of the United Kingdom, originating from Roman Britain.

>>> What is the average life expectancy in India?
The average life expectancy in India is at 70 years to 71.5 years for women, 68.7 years for men.
```
