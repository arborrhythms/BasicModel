"""Externally observable presentations for the measured rows in ModelAge.md."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Question:
    text: str
    answer: str
    prompt: str | None = None
    context: tuple = ()


def echo(words):
    return tuple(Question(word, word) for word in words)


def chart():
    corpus = Path(__file__).resolve().parents[1] / 'data/corpus-lessons.json'
    stages = json.loads(corpus.read_text())['stages']
    lesson = stages[0]
    sentences = tuple(sentence for document in lesson['documents']
        for record in document['records'] for sentence in record['sentences'])
    four_word_sentences = tuple(dict.fromkeys(sentence for stage in stages
        for document in stage['documents'] for record in document['records']
        for sentence in record['sentences'] if len(sentence.split()) == 4))
    words = tuple(dict.fromkeys(word for sentence in sentences for word in sentence[:-1].split()))
    counting = tuple(str(i) for i in range(1, 10))
    lesson_words = words + counting[:6]
    two = tuple(dict.fromkeys(' '.join(sentence.split()[:2]) + '.' for sentence in sentences))
    pairs = ('red blue', 'blue red', 'green gold', 'gold green')
    four = ('red blue green gold', 'gold green blue red', 'blue gold red green', 'green red gold blue')
    held_out = ('is This Felix?', 'Felix is here.', 'Ciao is that.', 'is that Ciao?',
                'here is Felix.', 'Felix is there.')
    def count(text, length):
        return Question(text, ' '.join(counting[:length]))
    truth = ('false', 'true')
    boolean = tuple(Question(f'{a} {op} {b}', truth[int((i and j) if op == 'and' else (i or j))])
                    for op in ('and', 'or') for i, a in enumerate(truth) for j, b in enumerate(truth))
    return (
        (.70, 'one word', tuple(Question('What was there?', name, context=(name,))
                                for name in ('Felix', 'Ciao'))),
        (.75, 'recognise a form', (Question('Felix', 'Felix'), Question('Felixx', ''),
                                   Question('cat', 'cat'), Question('catx', ''))),
        (1., 'test_echo_a_word_with_its_own_look', echo(lesson_words)),
        (1.50, 'Felix', (*echo(lesson_words[:10]), Question('This is Felix.', 'Felix', 'What is this?'))),
        (1.70, 'twenty words', echo(lesson_words + ('cat',))),
        (1.75, 'test_echo_a_sentence_of_two_words', echo(two)),
        (2., 'Felix ran', (Question('Felix ran', 'Felix', 'Who ran?'),
                           Question('Ciao ran', 'Ciao', 'Who ran?'))),
        (2.25, 'what is cat ?', (Question('cat is animal', 'animal', 'what is cat ?'),
                                 Question('x is three', 'three', 'what is x ?'))),
        (2.25, 'not', (Question('not true', 'false'), Question('not false', 'true'))),
        (2.50, 'three-word lesson sentences', echo(sentences)),
        (2.50, 'reading_two', echo(pairs)),
        (2.75, 'This is Felix.', (Question('This is Felix.', 'Felix', 'What is this?'),
                                  Question('That is Ciao.', 'Ciao', 'What is that?'))),
        (2.75, 'cat is on mat', (Question('cat is on mat', 'cat', 'what is on mat ?'),
                                 Question('dog is on rug', 'dog', 'what is on rug ?'))),
        (3., 'the cat ran', (Question('the cat ran', 'cat', 'what ran ?'),
                             Question('the dog ran', 'dog', 'what ran ?'))),
        (3., 'four-word lesson sentences', echo(four_word_sentences)),
        (3., 'what sleeps ?', (Question('the cat sleeps', 'cat', 'what sleeps ?'),
                               Question('the dog sleeps', 'dog', 'what sleeps ?'))),
        (3., 'count to four', tuple(count(text, 4) for text in sentences)),
        (3., 'and and or', boolean),
        (3.25, 'he ran', tuple(Question('he ran', name, 'who ran ?', (name + ' slept',))
                              for name in ('Felix', 'Ciao'))),
        (3.25, 'the fast cat ran', (Question('the fast cat ran', 'cat', 'what ran ?'),
                                   Question('the fast dog ran', 'dog', 'what ran ?'))),
        (3.75, 'count every word once', (count('Felix.', 2), count('This is.', 3),
                                        count('This is Felix.', 4), count('Felix is Felix.', 4),
                                        Question('Felix.', '2', 'How many words?'),
                                        Question('This is Felix.', '4', 'How many words?'))),
        (4., 'longest sentences and held out', echo(('Felix is a cat.', 'Who is there is Felix.', *held_out))),
        (4.25, 'adverb', (Question('Felix ran quickly', 'quickly', 'how did Felix run ?'),
                          Question('the fast cat ran quickly', 'quickly', 'how did the cat run ?'))),
        (4.50, 'bigger than', (Question('x is bigger than y', 'x', 'what is bigger than y ?'),
                               Question('y is bigger than x', 'y', 'what is bigger than x ?'))),
        (4.50, 'count to nine', (count('This is Felix and that is Ciao here.', 9),)),
        (5., 'reading_four', echo(four)),
        (6., 'exclusive or', ()),
    )
